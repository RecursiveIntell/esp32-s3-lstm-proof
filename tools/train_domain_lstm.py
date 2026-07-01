#!/usr/bin/env python3
"""Train domain-specific ESP32-S3 char LSTM variants for useful local status text.

Goal: not a chatbot. Train compact char-LSTM models to emit receipt/action/status
phrases for the physical AI endpoint. Exports RILM v1 binary compatible with the
host usefulness probe and mostly compatible with firmware after HIDDEN constants
are adjusted for the chosen size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = list("abcdefghijklmnopqrstuvwxyz .,!?'\n")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
MAGIC = 0x4D4C4952

BASE_EVENTS = [
    ("normal", "room normal", "receipt logged", "normal room. action is log receipt."),
    ("warm", "warm room", "watch temperature", "hot room. action is check airflow."),
    ("hot", "hot room", "turn on fan", "hot room. action is fan advised."),
    ("humid", "humid room", "start fan", "humid room. action is ventilate."),
    ("dry", "dry room", "add humidity", "dry room. action is add water."),
    ("stale", "sensor stale", "wait for data", "stale data. action is wait."),
    ("missing", "sensor missing", "no claim", "missing sensor. action is no claim."),
    ("escalate", "heat and humidity", "ask gateway", "high heat and humidity. action is escalate."),
]

EXTRA_LINES = [
    "do not invent readings. sensor missing means no claim.",
    "the gateway writes a receipt before it takes action.",
    "the model must say the data is stale when readings are stale.",
    "local first means the room can report before the cloud wakes.",
    "the esp32 reads temperature and humidity and logs evidence.",
    "the oled shows a short answer. the receipt keeps the reason.",
    "normal room. action is log receipt.",
    "hot humid room. action is escalate.",
    "sensor stale. action is wait for fresh data.",
    "missing sensor. action is no claim.",
    "humid air. action is fan advised.",
    "warm air. action is watch temperature.",
    "safe action is no claim without evidence.",
    "the receipt says local confident.",
    "the receipt says sensor missing.",
    "the receipt says stale reading.",
]

EVAL_PROMPTS = [
    "normal room. action is ",
    "hot room. action is ",
    "humid room. action is ",
    "missing sensor. action is ",
    "stale data. action is ",
    "high heat and humidity. action is ",
    "do not invent ",
    "if data is stale ",
    "the receipt says ",
]

class CharLSTM(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = layers
        self.embed = nn.Embedding(VOCAB_SIZE, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, VOCAB_SIZE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.shape[0]
                p.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x, hidden=None):
        y, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(y), hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def filter_text(s: str) -> str:
    s = s.lower()
    return "".join(ch if ch in CHAR_TO_IDX else " " for ch in s)


def build_corpus(repeats: int, seed: int) -> str:
    rng = random.Random(seed)
    lines: List[str] = []
    for _ in range(repeats):
        event, status, action, sentence = rng.choice(BASE_EVENTS)
        temp = rng.choice([61, 66, 70, 74, 78, 82, 85, 89])
        hum = rng.choice([22, 30, 42, 50, 58, 66, 72, 80])
        lines.extend([
            sentence,
            f"{status}. {action}.",
            f"temperature {temp}f humidity {hum} percent. {sentence}",
            f"decision {event}. route reason {event}. {action}.",
            f"local status {status}. receipt logged.",
            rng.choice(EXTRA_LINES),
        ])
        if rng.random() < 0.35:
            lines.append("\n")
    # Put exact prompt continuations everywhere so short ESP32 generation is reliable.
    for _ in range(max(100, repeats // 4)):
        lines.extend(EXTRA_LINES)
    rng.shuffle(lines)
    return filter_text("\n".join(lines) + "\n")


def encode(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=np.int64)


def make_batch(data: np.ndarray, seq_len: int, batch: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch)
    x = np.stack([data[i:i+seq_len] for i in starts])
    y = np.stack([data[i+1:i+seq_len+1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def eval_loss(model: nn.Module, data: np.ndarray, device: str, seq_len: int, chunks: int = 96) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        if len(data) < seq_len + 2:
            return float("inf")
        stride = max(1, (len(data) - seq_len - 1) // chunks)
        for i in range(0, len(data) - seq_len - 1, stride):
            x = torch.from_numpy(data[i:i+seq_len]).unsqueeze(0).to(device)
            y = torch.from_numpy(data[i+1:i+seq_len+1]).unsqueeze(0).to(device)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item())
            if len(losses) >= chunks:
                break
    model.train()
    return float(np.mean(losses))


def sample(model: nn.Module, prompt: str, length: int, device: str, temp: float = 0.55, top_k: int = 5) -> str:
    model.eval()
    ids = [CHAR_TO_IDX[c] for c in filter_text(prompt) if c in CHAR_TO_IDX] or [CHAR_TO_IDX[" "]]
    hidden = None
    result = prompt
    with torch.no_grad():
        for ch in ids[:-1]:
            _, hidden = model(torch.tensor([[ch]], device=device), hidden)
        cur = ids[-1]
        for _ in range(length):
            logits, hidden = model(torch.tensor([[cur]], device=device), hidden)
            logits = logits[0, -1] / temp
            vals, idx = torch.topk(logits, k=min(top_k, VOCAB_SIZE))
            probs = torch.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, 1).item()
            cur = int(idx[choice].item())
            result += IDX_TO_CHAR[cur]
    model.train()
    return result


def quant_i8(arr: np.ndarray):
    m = float(np.max(np.abs(arr)))
    scale = m / 127.0 if m > 0 else 1.0
    q = np.round(arr / scale).clip(-128, 127).astype(np.int8)
    return q, scale


def quant_i4(arr: np.ndarray):
    m = float(np.max(np.abs(arr)))
    scale = m / 7.0 if m > 0 else 1.0
    q = np.round(arr / scale).clip(-8, 7).astype(np.int8).reshape(-1)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
    lo = (q[0::2] & 0x0F).astype(np.uint8)
    hi = ((q[1::2] & 0x0F) << 4).astype(np.uint8)
    return (lo | hi).astype(np.uint8), scale


def tensor_payload(name: str, arr: np.ndarray, profile: str):
    # Preserve the runtime profile: recurrent matrices int8; input/projection/embed int4.
    if arr.dtype == np.float32 and ("bias" in name or name.endswith("fc.bias")):
        return 0, 1.0, arr.astype("<f4").tobytes()
    if profile == "all_int8" or "weight_hh" in name:
        q, scale = quant_i8(arr.astype(np.float32))
        return 1, scale, q.tobytes()
    packed, scale = quant_i4(arr.astype(np.float32))
    return 2, scale, packed.tobytes()


def export_rilm(model: CharLSTM, out_path: Path, profile: str) -> Dict[str, object]:
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    tensors = []
    def add(name: str, arr: np.ndarray):
        dtype, scale, payload = tensor_payload(name, arr, profile)
        tensors.append((name, dtype, arr.shape, scale, payload))
    add("embed.weight", state["embed.weight"])
    for layer in range(model.num_layers):
        add(f"lstm.weight_ih_l{layer}", state[f"lstm.weight_ih_l{layer}"])
        add(f"lstm.weight_hh_l{layer}", state[f"lstm.weight_hh_l{layer}"])
        add(f"lstm.bias_ih_l{layer}", state[f"lstm.bias_ih_l{layer}"])
        add(f"lstm.bias_hh_l{layer}", state[f"lstm.bias_hh_l{layer}"])
    add("fc.weight", state["fc.weight"])
    add("fc.bias", state["fc.bias"])
    out = bytearray()
    out += struct.pack("<IHHI", MAGIC, 1, 0, len(tensors))
    payload_bytes = 0
    for name, dtype, shape, scale, payload in tensors:
        nb = name.encode("utf-8")
        out += struct.pack("<H", len(nb)) + nb
        out += struct.pack("<BB", dtype, len(shape))
        for dim in shape:
            out += struct.pack("<I", int(dim))
        out += struct.pack("<fI", float(scale), len(payload))
        out += payload
        payload_bytes += len(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)
    return {
        "path": str(out_path),
        "bytes": len(out),
        "sha256": hashlib.sha256(out).hexdigest(),
        "profile": profile,
        "payload_bytes": payload_bytes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--repeats", type=int, default=5500)
    ap.add_argument("--out", default="runs/domain_lstm_h256_l3")
    ap.add_argument("--profile", choices=["mixed_lstm_safe", "all_int8"], default="mixed_lstm_safe")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(max(1, min(12, (os_count := __import__('os').cpu_count() or 4))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    corpus = build_corpus(args.repeats, 42)
    (out / "domain_corpus.txt").write_text(corpus, encoding="utf-8")
    data = encode(corpus)
    val_n = max(5000, len(data) // 10)
    train, val = data[:-val_n], data[-val_n:]
    rng = np.random.default_rng(42)

    model = CharLSTM(args.hidden, args.layers, dropout=0.05).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    best_loss = float("inf")
    log = []
    print(json.dumps({"device": device, "params": model.count_params(), "tokens": len(data), "args": vars(args)}, indent=2), flush=True)
    for step in range(args.steps + 1):
        xb, yb = make_batch(train, args.seq_len, args.batch_size, rng)
        xb = xb.to(device); yb = yb.to(device)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            vl = eval_loss(model, val, device, args.seq_len)
            vp = math.exp(vl) if vl < 20 else float("inf")
            entry = {"step": step, "train_loss": float(loss.item()), "val_loss": vl, "val_ppl": vp, "elapsed_s": time.time() - start}
            log.append(entry)
            print(json.dumps(entry), flush=True)
            if vl < best_loss:
                best_loss = vl
                torch.save({"model_state": model.state_dict(), "args": vars(args), "vocab": VOCAB, "best_val_loss": best_loss, "params": model.count_params()}, out / "best.pt")
            print("SAMPLE " + sample(model, "missing sensor. action is ", 80, device).replace("\n", "\\n"), flush=True)
            (out / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_loss = eval_loss(model, val, device, args.seq_len)
    torch.save({"model_state": model.state_dict(), "args": vars(args), "vocab": VOCAB, "best_val_loss": best_loss, "final_val_loss": final_loss, "params": model.count_params()}, out / "final.pt")
    export = export_rilm(model, out / "weights" / f"domain_lstm_h{args.hidden}_l{args.layers}_{args.profile}.bin", args.profile)
    samples = {}
    for prompt in EVAL_PROMPTS:
        samples[prompt] = sample(model, prompt, 96, device, temp=0.55, top_k=5)
    summary = {
        "schema": "ri_esp32_domain_lstm_train_v1",
        "hidden": args.hidden,
        "layers": args.layers,
        "params": model.count_params(),
        "corpus_chars": len(corpus),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "best_val_loss": best_loss,
        "best_val_ppl": math.exp(best_loss) if best_loss < 20 else None,
        "final_val_loss": final_loss,
        "final_val_ppl": math.exp(final_loss) if final_loss < 20 else None,
        "elapsed_s": time.time() - start,
        "export": export,
        "samples": samples,
        "claim_boundary": "Host-trained and RILM-exported. Requires firmware HIDDEN constant/weight flash update before ESP32 hardware claim.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
