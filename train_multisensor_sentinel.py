#!/usr/bin/env python3
"""
Train multi-sensor Sentinel char-level LSTM models for ESP32-S3.

Produces two RILM v1 binary models:
  - H256 (1.6M params, fast profile ~40 chars/s)
  - H320 (2.5M params, accurate profile ~17 chars/s)

The training corpus covers all 16 sensor types with canonical
status phrases that the deterministic policy will feed to the LM
for bounded advisory text generation.

RILM v1 binary format:
  magic: u32 = 0x4d4c4952 ("RILM")
  version: u16 = 1
  reserved: u16 = 0
  tensor_count: u32
  per tensor:
    name_len: u16
    name: bytes[name_len]
    dtype: u8  (0=f32, 1=int8, 3=int8)
    ndim: u8
    dims: u32 * ndim
    scale: f32
    payload_len: u32
    payload: bytes[payload_len]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?'") + ["\n"]
VOCAB_SIZE = len(VOCAB_CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}

# ── Multi-sensor canonical training templates ──────────────────────────
# Each template is (prompt_prefix, completion) where the policy will feed
# the prompt_prefix to the model and expect the model to produce something
# close to completion (stopped at '.' or '\\n').

SENSOR_TEMPLATES = [
    # Temperature / humidity (original domain)
    ("hot room. action is ",        "check airflow."),
    ("humid room. action is ",      "ventilate."),
    ("high heat and humidity. action is ", "escalate."),
    ("cold room. action is ",       "no claim without evidence."),
    ("dry room. action is ",        "no claim without evidence."),
    ("normal room. action is ",     "log receipt."),
    ("stale data. action is ",      "wait."),
    ("missing sensor. action is ",  "no claim."),

    # CO2
    ("co2 is high. action is ",     "ventilate now."),
    ("co2 is normal. action is ",   "log receipt."),
    ("co2 is low. action is ",      "log receipt."),

    # Air quality
    ("air quality is poor. action is ", "ventilate."),
    ("air quality is fair. action is ", "log receipt."),
    ("air quality is good. action is ", "log receipt."),

    # Gas
    ("gas detected. action is ",    "ventilate."),
    ("no gas detected. action is ", "log receipt."),

    # Pressure
    ("pressure is low. action is ", "log receipt."),
    ("pressure is normal. action is ", "log receipt."),

    # Light
    ("light is low. action is ",    "log receipt."),
    ("light is high. action is ",   "log receipt."),
    ("light is normal. action is ", "log receipt."),

    # Motion
    ("motion detected. action is ", "log event."),
    ("no motion. action is ",       "log receipt."),

    # Distance
    ("object is close. action is ", "log event."),
    ("object is far. action is ",   "log receipt."),

    # Soil moisture
    ("soil is dry. action is ",     "water."),
    ("soil is moist. action is ",   "log receipt."),

    # Vibration
    ("vibration is high. action is ", "inspect."),
    ("vibration is normal. action is ", "log receipt."),

    # Door contact
    ("door is open. action is ",    "log event."),
    ("door is closed. action is ",  "log receipt."),

    # Voltage
    ("voltage is low. action is ",  "check power."),
    ("voltage is normal. action is ", "log receipt."),

    # Sound
    ("sound is loud. action is ",   "log receipt."),
    ("sound is quiet. action is ",  "log receipt."),

    # Current
    ("current is high. action is ", "check load."),
    ("current is normal. action is ", "log receipt."),

    # Combinations / sentinel identity
    ("the receipt says ",           "log receipt."),
    ("local first means ",          "the room can think without the cloud."),
    ("the sensor sees ",            "evidence for the receipt."),
    ("the sentinel watches ",       "the room and logs the truth."),
    ("safe action is ",             "no claim without evidence."),

    # Servo / actuator actions
    ("the vent is open. action is ", "ventilate."),
    ("the vent is closed. action is ", "log receipt."),
    ("servo moves to open. action is ", "ventilate."),
    ("servo moves to close. action is ", "log receipt."),
    ("the servo opens the vent. action is ", "ventilate."),
    ("the servo closes the vent. action is ", "safe."),
    ("actuator engaged. action is ", "ventilate."),
    ("actuator disengaged. action is ", "safe."),
]

# Additional natural-language context for richer language modeling
NARRATIVE_TEMPLATES = [
    "the esp32 reads the room and sends the evidence to the local model. ",
    "the sensor sees warm air and heavy humidity. the gateway decides what matters. ",
    "a tiny board watches the world. a larger model answers only when needed. ",
    "the oled shows a short answer. the receipt keeps the reason. ",
    "local first means the room can think without the cloud. ",
    "the temperature is evidence. the humidity is evidence. the action must match the evidence. ",
    "when the data is stale the model must say the data is stale. ",
    "the small model stays awake. the big model wakes only when the signal is worth it. ",
    "the gateway writes a receipt before it takes action. ",
    "do not invent readings. do not pretend the room is safe if the sensor is missing. ",
    "the physical node is cheap, quiet, and always watching. ",
    "the sentinel logs every reading with a hash chain. ",
    "co2 is high so the sentinel says ventilate now. ",
    "gas detected means the sentinel says ventilate. ",
    "the door is open so the sentinel logs the event. ",
    "vibration is high so the sentinel says inspect. ",
    "soil is dry so the sentinel says water. ",
    "voltage is low so the sentinel says check power. ",
    "the sentinel never decides safety. the policy decides. the model only phrases it. ",
]


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.num_layers = layers
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, vocab_size)
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.shape[0]
                p.data[n // 4:n // 2].fill_(1.0)  # forget gate bias

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def filter_text(text: str) -> str:
    text = text.lower()
    out = []
    for ch in text:
        if ch in CHAR_TO_IDX:
            out.append(ch)
        elif ch in "\r\t":
            out.append(" ")
        elif ch == "\n":
            out.append("\n")
    return "".join(out)


def build_corpus() -> str:
    """Build multi-sensor domain corpus from templates."""
    rng = random.Random(42)
    parts = []

    # Repeat sensor templates with variations
    for _ in range(800):
        prompt, completion = rng.choice(SENSOR_TEMPLATES)
        parts.append(prompt + completion + "\n")

    # Interleave narrative templates
    for _ in range(400):
        parts.append(filter_text(rng.choice(NARRATIVE_TEMPLATES)))

    # Build longer sequences: chain multiple sensor readings
    for _ in range(300):
        chain = []
        for _ in range(rng.randint(2, 5)):
            prompt, completion = rng.choice(SENSOR_TEMPLATES)
            chain.append(prompt + completion + "\n")
        parts.append("".join(chain))

    rng.shuffle(parts)
    corpus = "".join(parts)
    return corpus


def encode(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=np.int64)


def batch_iter(data: np.ndarray, seq_len: int, batch_size: int):
    n = len(data) // (seq_len * batch_size)
    data = data[:n * seq_len * batch_size].reshape(batch_size, -1)
    for i in range(0, data.shape[1] - seq_len - 1, seq_len):
        yield data[:, i:i+seq_len], data[:, i+1:i+seq_len+1]


def lr_at(step, warmup, lr, min_lr, total):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def evaluate(model, data, device, seq_len, max_chunks=256):
    model.eval()
    losses = []
    with torch.no_grad():
        stride = max(seq_len, (len(data) - seq_len - 1) // max_chunks)
        for i in range(0, len(data) - seq_len - 1, stride):
            x = torch.from_numpy(data[i:i+seq_len]).unsqueeze(0).to(device)
            y = torch.from_numpy(data[i+1:i+seq_len+1]).unsqueeze(0).to(device)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                          y.reshape(-1)).item())
            if len(losses) >= max_chunks:
                break
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def sample(model, seed, length, device, temp=0.5):
    model.eval()
    chars = [CHAR_TO_IDX[c] for c in seed.lower() if c in CHAR_TO_IDX] or [CHAR_TO_IDX['i']]
    hidden = None
    result = seed
    with torch.no_grad():
        for ch in chars[:-1]:
            _, hidden = model(torch.tensor([[ch]], device=device), hidden)
        cur = chars[-1]
        for _ in range(length):
            logits, hidden = model(torch.tensor([[cur]], device=device), hidden)
            probs = F.softmax(logits[0, -1] / temp, dim=-1)
            cur = torch.multinomial(probs, 1).item()
            result += IDX_TO_CHAR[cur]
    model.train()
    return result


def q8(arr: np.ndarray):
    m = float(np.max(np.abs(arr)))
    scale = m / 127.0 if m > 0 else 1.0
    return np.round(arr / scale).clip(-128, 127).astype(np.int8), scale


def export_rilm(model: nn.Module, out_path: str):
    """Export PyTorch CharLSTM to RILM v1 binary format."""
    st = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    hidden = model.hidden_size
    layers = model.num_layers

    buf = bytearray()
    # Header
    buf += struct.pack('<I', 0x4d4c4952)  # magic "RILM"
    buf += struct.pack('<H', 1)           # version
    buf += struct.pack('<H', 0)           # reserved
    buf += struct.pack('<I', 0)           # tensor_count (filled later)

    tensors = []

    # embed.weight (int8, dtype=1 matching firmware I8 enum)
    q, s = q8(st['embed.weight'])
    tensors.append(('embed.weight', 1, 2, q.shape, s, q.tobytes()))

    # LSTM weights per layer (int8, dtype=1)
    for layer in range(layers):
        for prefix in ['weight_ih_l', 'weight_hh_l']:
            key = f'lstm.{prefix}{layer}'
            q, s = q8(st[key])
            tensors.append((key, 1, 2, q.shape, s, q.tobytes()))

        # Biases (f32, dtype=0)
        bih = st[f'lstm.bias_ih_l{layer}']
        bhh = st[f'lstm.bias_hh_l{layer}']
        combined = (bih + bhh).astype(np.float32)
        tensors.append((f'lstm.bias_ih_l{layer}', 0, 1, combined.shape, 1.0, combined.tobytes()))
        bhh_zero = np.zeros_like(bhh, dtype=np.float32)
        tensors.append((f'lstm.bias_hh_l{layer}', 0, 1, bhh_zero.shape, 1.0, bhh_zero.tobytes()))

    # fc.weight (int8, dtype=1) and fc.bias (f32, dtype=0)
    q, s = q8(st['fc.weight'])
    tensors.append(('fc.weight', 1, 2, q.shape, s, q.tobytes()))
    fcb = st['fc.bias'].astype(np.float32)
    tensors.append(('fc.bias', 0, 1, fcb.shape, 1.0, fcb.tobytes()))

    # Fix tensor count
    struct.pack_into('<I', buf, 8, len(tensors))

    # Write tensors
    for name, dtype, ndim, dims, scale, payload in tensors:
        name_bytes = name.encode('utf-8')
        buf += struct.pack('<H', len(name_bytes))
        buf += name_bytes
        buf += struct.pack('<B', dtype)
        buf += struct.pack('<B', ndim)
        for d in dims:
            buf += struct.pack('<I', d)
        buf += struct.pack('<f', float(scale))
        buf += struct.pack('<I', len(payload))
        buf += payload

    data = bytes(buf)
    sha = hashlib.sha256(data).hexdigest()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(data)

    print(f"RILM exported: {out_path}")
    print(f"  size={len(data)} bytes")
    print(f"  sha256={sha}")
    print(f"  tensors={len(tensors)}")
    print(f"  params={model.count_params():,}")
    return sha, len(data)


def main():
    ap = argparse.ArgumentParser(description='Train multi-sensor Sentinel LSTM')
    ap.add_argument('--hidden', type=int, default=320, help='hidden size (256 or 320)')
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--out', default=None, help='output directory')
    args = ap.parse_args()

    if args.out is None:
        args.out = f'runs/multisensor_h{args.hidden}_l{args.layers}'

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    print(json.dumps({
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'args': vars(args),
    }, indent=2), flush=True)

    # Build corpus
    corpus = build_corpus()
    data = encode(corpus)
    val_n = max(10_000, int(len(data) * 0.05))
    train, val = data[:-val_n], data[-val_n:]
    print(f"corpus tokens={len(data):,} train={len(train):,} val={len(val):,}", flush=True)

    model = CharLSTM(VOCAB_SIZE, args.hidden, args.layers, dropout=0.1).to(device)
    print(f"params={model.count_params():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    best_loss = float('inf')
    best_path = out / 'best.pt'
    start = time.time()
    step = 0
    epoch = 0
    log = []

    while step < args.steps:
        epoch += 1
        offset = random.randint(0, args.seq_len - 1) if epoch > 1 else 0
        epoch_data = np.concatenate([train[offset:], train[:offset]])
        for xb, yb in batch_iter(epoch_data, args.seq_len, args.batch_size):
            if step >= args.steps:
                break
            lr = lr_at(step, args.warmup, args.lr, args.min_lr, args.steps)
            for g in opt.param_groups:
                g['lr'] = lr
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            y = torch.from_numpy(yb).to(device, non_blocking=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"step {step:05d}/{args.steps} loss={loss.item():.4f} "
                      f"ppl={math.exp(loss.item()):.2f} lr={lr:.2e} "
                      f"elapsed={elapsed/60:.1f}m", flush=True)
            if step > 0 and step % 250 == 0:
                vl = evaluate(model, val, device, args.seq_len)
                vp = math.exp(vl)
                print(f"VAL step={step} loss={vl:.4f} ppl={vp:.2f}", flush=True)
                log.append({'step': step, 'train_loss': float(loss.item()),
                           'val_loss': vl, 'val_ppl': vp,
                           'elapsed_s': time.time() - start})
                if vl < best_loss:
                    best_loss = vl
                    torch.save({
                        'model_state': model.state_dict(),
                        'args': vars(args),
                        'vocab_chars': VOCAB_CHARS,
                        'param_count': model.count_params(),
                        'best_val_loss': best_loss,
                    }, best_path)
                    print(f"  BEST saved {best_path}", flush=True)
                # Sample with sentinel prompts
                for seed in ['hot room. action is ', 'co2 is high. action is ',
                             'gas detected. action is ', 'normal room. action is ']:
                    s = sample(model, seed, 32, device, temp=0.3)
                    print(f"  SAMPLE [{seed!r}] → {s!r}", flush=True)
                (out / 'training_log.json').write_text(
                    json.dumps(log, indent=2), encoding='utf-8')
            step += 1

    # Load best model
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    final_loss = evaluate(model, val, device, args.seq_len)
    print(f"\nFINAL val_loss={final_loss:.4f} ppl={math.exp(final_loss):.2f}", flush=True)

    # Export to RILM v1 binary
    weights_dir = out / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    rilm_path = weights_dir / f'multisensor_h{args.hidden}_all_int8.bin'
    sha, size = export_rilm(model, str(rilm_path))

    # Save summary
    summary = {
        'params': model.count_params(),
        'hidden': args.hidden,
        'layers': args.layers,
        'final_val_loss': final_loss,
        'final_val_ppl': math.exp(final_loss),
        'best_val_loss': best_loss,
        'rilm_path': str(rilm_path),
        'rilm_sha256': sha,
        'rilm_size': size,
        'elapsed_s': time.time() - start,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\nSUMMARY {json.dumps(summary, indent=2)}", flush=True)


if __name__ == '__main__':
    main()