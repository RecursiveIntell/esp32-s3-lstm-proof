#!/usr/bin/env python3
"""Train a H768 3-layer char-LSTM on TinyStories for the ESP32-S3 3-board cluster.

Target: ~14.2M params, fits in 8MB PSRAM at int8/mixed quantization.
Exports RILM v1 binary compatible with existing firmware parser.

Usage:
  python3 tools/train_tinystories_h768.py --data /home/sikmindz/projects/dying-llm/training/tinystories.txt --epochs 3 --batch-size 64 --seq-len 128 --lr 0.001
"""
from __future__ import annotations

import argparse
import hashlib
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

VOCAB = list("abcdefghijklmnopqrstuvwxyz .,!?'\"\n")
VOCAB_SIZE = len(VOCAB)  # 34
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
MAGIC = 0x4D4C4952  # "RILM"

# All-int4 weight profile: ih, hh, embed, fc all int4. Biases float32.
# H=768 3-layer = 14.2M params, ~7.1MB all-int4. Split across 3 boards
# (one layer per board) = ~2.4MB per board, fits in 5MB partition.
PROFILE_ALL_INT4 = "tinystories_h768_all_int4"

EVAL_PROMPTS = [
    "once upon a ",
    "there was a ",
    "one day, a ",
    "the little ",
    "she said, ",
]


class CharLSTM(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = layers
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, vocab_size)
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_corpus(path: str, max_chars: int = 0) -> str:
    print(f"Loading corpus from {path}...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if max_chars > 0:
        text = text[:max_chars]
    print(f"  loaded {len(text):,} chars")
    return text


def encode(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX.get(c, 0) for c in text], dtype=np.int64)


def make_batches(data: np.ndarray, seq_len: int, batch_size: int, device: str):
    n_batches = len(data) // (seq_len + 1)
    if n_batches < batch_size:
        return
    data = data[:n_batches * (seq_len + 1)].reshape(n_batches, seq_len + 1)
    for i in range(0, n_batches, batch_size):
        chunk = data[i:i + batch_size]
        if len(chunk) < batch_size:
            break
        x = torch.tensor(chunk[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(chunk[:, 1:], dtype=torch.long, device=device)
        yield x, y


def train(model, data, args, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 100)

    step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in make_batches(data, args.seq_len, args.batch_size, device):
            optimizer.zero_grad()
            out, _ = model(x)
            loss = F.cross_entropy(out.reshape(-1, model.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            step += 1

            if step % 100 == 0:
                avg = total_loss / n_batches
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1}/{args.epochs} step {step} loss={avg:.4f} batches={n_batches} elapsed={elapsed:.1f}s")

        if n_batches > 0:
            print(f"  epoch {epoch+1} done: avg_loss={total_loss/n_batches:.4f} time={time.time()-t0:.1f}s")

    return model


@torch.no_grad()
def evaluate(model, device, n_samples=5, gen_len=64):
    model.eval()
    for prompt in EVAL_PROMPTS[:n_samples]:
        chars = list(prompt)
        x = torch.tensor([[CHAR_TO_IDX.get(c, 0) for c in chars]], dtype=torch.long, device=device)
        hidden = None
        output = prompt
        for _ in range(gen_len):
            out, hidden = model(x[:, -1:].unsqueeze(0) if x.dim() == 1 else x[:, -1:], hidden)
            # Use last position
            logits = out[0, -1]
            prob = F.softmax(logits, dim=0)
            idx = torch.multinomial(prob, 1).item()
            ch = IDX_TO_CHAR.get(idx, '?')
            output += ch
            x = torch.cat([x, torch.tensor([[idx]], dtype=torch.long, device=device)], dim=1)
        print(f"  prompt='{prompt}' -> '{output}'")


def quantize_int8(arr: np.ndarray) -> tuple:
    """Quantize to int8 with scale."""
    max_abs = max(np.abs(arr).max(), 1e-8)
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def quantize_int4(arr: np.ndarray) -> tuple:
    """Quantize to int4 with scale."""
    max_abs = max(np.abs(arr).max(), 1e-8)
    scale = max_abs / 7.0
    q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
    return q, float(scale)


def tensor_payload(name: str, arr: np.ndarray, profile: str) -> tuple:
    """Return (dtype, shape, scale, payload_bytes) for RILM export."""
    if "bias" in name:
        # Float32 biases
        return 0, arr.shape, 1.0, arr.astype(np.float32).tobytes()
    elif "all_int4" in profile:
        # All weights as int4 (dtype=1)
        q, scale = quantize_int4(arr)
        return 1, arr.shape, scale, q.tobytes()
    elif "weight_ih" in name and "mixed" in profile:
        # Int4 input weights
        q, scale = quantize_int4(arr)
        return 1, arr.shape, scale, q.tobytes()
    elif "weight_hh" in name and "mixed" in profile:
        # Int8 recurrent weights (firmware converts to int4 at boot)
        q, scale = quantize_int8(arr)
        return 2, arr.shape, scale, q.tobytes()
    else:
        # Default int8
        q, scale = quantize_int8(arr)
        return 2, arr.shape, scale, q.tobytes()


def export_rilm(model: CharLSTM, path: str, profile: str):
    """Export model to RILM v1 binary format."""
    state = model.state_dict()
    tensors = []

    def add(name, arr):
        dtype, shape, scale, payload = tensor_payload(name, arr, profile)
        tensors.append((name, dtype, len(shape), list(shape), scale, payload))

    add("embed.weight", state["embed.weight"].numpy())
    for layer in range(model.num_layers):
        add(f"lstm.weight_ih_l{layer}", state[f"lstm.weight_ih_l{layer}"].numpy())
        add(f"lstm.weight_hh_l{layer}", state[f"lstm.weight_hh_l{layer}"].numpy())
        add(f"lstm.bias_ih_l{layer}", state[f"lstm.bias_ih_l{layer}"].numpy())
        add(f"lstm.bias_hh_l{layer}", state[f"lstm.bias_hh_l{layer}"].numpy())
    add("fc.weight", state["fc.weight"].numpy())
    add("fc.bias", state["fc.bias"].numpy())

    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<H", 1))  # version
        f.write(struct.pack("<H", 0))  # reserved
        f.write(struct.pack("<I", len(tensors)))
        for name, dtype, ndim, dims, scale, payload in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("B", dtype))
            f.write(struct.pack("B", ndim))
            for d in dims:
                f.write(struct.pack("<I", d))
            f.write(struct.pack("<f", scale))
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    file_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
    file_size = os.path.getsize(path)
    print(f"Exported RILM to {path}")
    print(f"  size={file_size:,} bytes")
    print(f"  sha256={file_hash}")
    print(f"  profile={profile}")
    print(f"  tensors={len(tensors)}")
    print(f"  params={count_params(model):,}")
    return file_hash, file_size


def export_rilm_layer_shards(model: CharLSTM, output_prefix: str, profile: str):
    """Export model as per-board RILM shards for layer-sharded cluster.

    Board 0: embed + layer0 + fc (coordinator owns embedding, first layer, output head)
    Board 1: layer1
    Board 2: layer2
    """
    state = model.state_dict()

    def write_rilm(path, tensor_names):
        tensors = []
        for name in tensor_names:
            arr = state[name].numpy()
            dtype, shape, scale, payload = tensor_payload(name, arr, profile)
            tensors.append((name, dtype, len(shape), list(shape), scale, payload))
        with open(path, "wb") as f:
            f.write(struct.pack("<I", MAGIC))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", 0))
            f.write(struct.pack("<I", len(tensors)))
            for name, dtype, ndim, dims, scale, payload in tensors:
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("B", dtype))
                f.write(struct.pack("B", ndim))
                for d in dims:
                    f.write(struct.pack("<I", d))
                f.write(struct.pack("<f", scale))
                f.write(struct.pack("<I", len(payload)))
                f.write(payload)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        s = os.path.getsize(path)
        print(f"  {path}: {s:,} bytes sha256={h[:16]}... tensors={len(tensors)}")
        return h, s

    print("Exporting layer-shard RILM files:")
    # Board 0: embed + layer0 + fc
    board0_names = ["embed.weight", "lstm.weight_ih_l0", "lstm.weight_hh_l0",
                    "lstm.bias_ih_l0", "lstm.bias_hh_l0", "fc.weight", "fc.bias"]
    h0, s0 = write_rilm(f"{output_prefix}_board0.bin", board0_names)

    # Board 1: layer1
    board1_names = ["lstm.weight_ih_l1", "lstm.weight_hh_l1",
                    "lstm.bias_ih_l1", "lstm.bias_hh_l1"]
    h1, s1 = write_rilm(f"{output_prefix}_board1.bin", board1_names)

    # Board 2: layer2
    board2_names = ["lstm.weight_ih_l2", "lstm.weight_hh_l2",
                    "lstm.bias_ih_l2", "lstm.bias_hh_l2"]
    h2, s2 = write_rilm(f"{output_prefix}_board2.bin", board2_names)

    print(f"Layer-shard export complete. Total: {s0+s1+s2:,} bytes")
    return [(h0, s0), (h1, s1), (h2, s2)]


def main():
    parser = argparse.ArgumentParser(description="Train H768 TinyStories char-LSTM for ESP32-S3")
    parser.add_argument("--data", required=True, help="Path to tinystories.txt")
    parser.add_argument("--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("--layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-chars", type=int, default=20000000, help="Max chars from corpus")
    parser.add_argument("--output", default="weights_h768_tinystories.bin", help="Output RILM file")
    parser.add_argument("--profile", default=PROFILE_ALL_INT4, help="Quantization profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cpu"
    print(f"Device: {device}")

    model = CharLSTM(args.hidden, args.layers, dropout=0.1, vocab_size=VOCAB_SIZE)
    n_params = count_params(model)
    print(f"Model: H={args.hidden} L={args.layers} V={VOCAB_SIZE}")
    print(f"Params: {n_params:,}")
    print(f"  embed: {VOCAB_SIZE * args.hidden:,}")
    for l in range(args.layers):
        ih = 4 * args.hidden * args.hidden
        hh = 4 * args.hidden * args.hidden
        b = 4 * args.hidden
        print(f"  layer {l}: ih={ih:,} hh={hh:,} bias={b:,}")
    fc = args.hidden * VOCAB_SIZE
    print(f"  fc: {fc:,}")
    print(f"  total: {n_params:,}")

    text = load_corpus(args.data, args.max_chars)
    data = encode(text)
    print(f"Encoded {len(data):,} tokens")

    print("Starting training...")
    model = train(model, data, args, device)

    print("Evaluation samples:")
    evaluate(model, device, n_samples=5, gen_len=64)

    print("Exporting RILM binary...")
    file_hash, file_size = export_rilm(model, args.output, args.profile)

    print("Exporting layer-shard RILM files (one per board)...")
    shard_prefix = args.output.replace(".bin", "_layer_shard")
    shard_info = export_rilm_layer_shards(model, shard_prefix, args.profile)

    # Save model state dict too
    torch.save(model.state_dict(), args.output + ".pt")
    print(f"Saved PyTorch state to {args.output}.pt")

    print("\nDone. Full model flash to ESP32-S3 with:")
    print(f"  esptool.py --chip esp32s3 --port /dev/ttyACM0 write_flash 0x210000 {args.output}")
    print(f"\nLayer-shard flash (one per board):")
    for i, (h, s) in enumerate(shard_info):
        print(f"  board{i}: esptool.py ... write_flash 0x210000 {shard_prefix}_board{i}.bin ({s:,} bytes)")
    print(f"\nBuild flags needed in platformio.ini:")
    print(f"  -D RI_HIDDEN={args.hidden}")
    print(f"  -D RI_LAYERS={args.layers}")
    print(f"  -D RI_WEIGHTS_SHA256=\"{file_hash}\"")
    print(f"  -D RI_MODEL_PROFILE=\"{args.profile}\"")
    print(f"  -D RI_MODEL_PARAMS={n_params}")
    print(f"  -D RI_COMPRESSED_BYTES={file_size}")


if __name__ == "__main__":
    main()