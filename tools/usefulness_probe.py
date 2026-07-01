#!/usr/bin/env python3
"""Probe what the current ESP32-S3 6.34M LSTM can usefully do.

This is a host-side mirror of src/main.cpp for the RILM v1 weight pack. It is
not a speed benchmark; hardware speed comes from BENCH_RECEIPT. This script
answers a different question: with the current trained weights and tokenizer,
which prompting/decoding/use-cases produce useful local behavior?
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

VOCAB = list("abcdefghijklmnopqrstuvwxyz .,!?'\n")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
MAGIC = 0x4D4C4952

PROMPTS = [
    "the sensor says ",
    "the room feels ",
    "the esp32 sees ",
    "temperature is ",
    "humidity is ",
    "the air is ",
    "the oled says ",
    "local first means ",
    "the gateway should ",
    "if data is stale ",
    "do not invent ",
    "the receipt says ",
    "when humidity rises ",
    "when the room is hot ",
    "safe action is ",
]

POLICY_PROMPTS = [
    "normal room. action is ",
    "hot room. action is ",
    "humid room. action is ",
    "missing sensor. action is ",
    "stale data. action is ",
    "high heat and humidity. action is ",
]

GOOD_TERMS = {
    "sensor", "room", "temp", "temperature", "humidity", "humid", "hot", "warm", "air",
    "data", "stale", "local", "gateway", "model", "receipt", "evidence", "safe", "action",
    "oled", "esp", "esp32", "cloud", "read", "reads", "says", "send", "sends", "watch",
}
BAD_FRAGMENTS = ["aaaa", "eeee", "iiii", "oooo", "    ", "..", ",,", "!!", "??", "\n\n\n"]

@dataclass
class Tensor:
    name: str
    dtype: int
    dims: Tuple[int, ...]
    scale: float
    payload: bytes

    def array(self) -> np.ndarray:
        shape = self.dims
        if self.dtype == 0:
            return np.frombuffer(self.payload, dtype="<f4").copy().reshape(shape)
        if self.dtype == 1:
            return (np.frombuffer(self.payload, dtype=np.int8).astype(np.float32) * self.scale).reshape(shape)
        if self.dtype == 2:
            raw = np.frombuffer(self.payload, dtype=np.uint8)
            lo = (raw & 0x0F).astype(np.int8)
            hi = ((raw >> 4) & 0x0F).astype(np.int8)
            lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
            hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
            vals = np.empty(raw.size * 2, dtype=np.int8)
            vals[0::2] = lo
            vals[1::2] = hi
            n = int(np.prod(shape))
            return (vals[:n].astype(np.float32) * self.scale).reshape(shape)
        raise ValueError(f"bad dtype {self.dtype}")

def read_u16(buf: memoryview, off: int) -> Tuple[int, int]:
    return struct.unpack_from("<H", buf, off)[0], off + 2

def read_u32(buf: memoryview, off: int) -> Tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4

def read_f32(buf: memoryview, off: int) -> Tuple[float, int]:
    return struct.unpack_from("<f", buf, off)[0], off + 4

def parse_rilm(path: Path) -> Dict[str, Tensor]:
    data = path.read_bytes()
    buf = memoryview(data)
    off = 0
    magic, off = read_u32(buf, off)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    version, off = read_u16(buf, off)
    _reserved, off = read_u16(buf, off)
    tensor_count, off = read_u32(buf, off)
    tensors: Dict[str, Tensor] = {}
    for _ in range(tensor_count):
        name_len, off = read_u16(buf, off)
        name = bytes(buf[off:off + name_len]).decode("utf-8")
        off += name_len
        dtype = int(buf[off]); off += 1
        ndim = int(buf[off]); off += 1
        dims = []
        for _d in range(ndim):
            dim, off = read_u32(buf, off)
            dims.append(dim)
        scale, off = read_f32(buf, off)
        payload_len, off = read_u32(buf, off)
        payload = bytes(buf[off:off + payload_len])
        off += payload_len
        tensors[name] = Tensor(name, dtype, tuple(dims), scale, payload)
    return tensors

class RilmLstm:
    def __init__(self, tensors: Dict[str, Tensor]):
        self.embed = tensors["embed.weight"].array()
        self.fcw = tensors["fc.weight"].array()
        self.fcb = tensors["fc.bias"].array()
        self.wih = [tensors[f"lstm.weight_ih_l{i}"].array() for i in range(3)]
        self.whh = [tensors[f"lstm.weight_hh_l{i}"].array() for i in range(3)]
        self.b = [tensors[f"lstm.bias_ih_l{i}"].array() + tensors[f"lstm.bias_hh_l{i}"].array() for i in range(3)]
        self.hidden = self.embed.shape[1]
        self.layers = len(self.wih)
        self.reset()

    def reset(self) -> None:
        self.h = [np.zeros(self.hidden, dtype=np.float32) for _ in range(self.layers)]
        self.c = [np.zeros(self.hidden, dtype=np.float32) for _ in range(self.layers)]

    def step_logits(self, token: int) -> np.ndarray:
        x = self.embed[token]
        for layer in range(self.layers):
            gates = self.wih[layer] @ x + self.whh[layer] @ self.h[layer] + self.b[layer]
            i = 1.0 / (1.0 + np.exp(-np.clip(gates[0:self.hidden], -40, 40)))
            f = 1.0 / (1.0 + np.exp(-np.clip(gates[self.hidden:2*self.hidden], -40, 40)))
            g = np.tanh(gates[2*self.hidden:3*self.hidden])
            o = 1.0 / (1.0 + np.exp(-np.clip(gates[3*self.hidden:4*self.hidden], -40, 40)))
            self.c[layer] = f * self.c[layer] + i * g
            self.h[layer] = o * np.tanh(self.c[layer])
            x = self.h[layer]
        return self.fcw @ x + self.fcb

    def prime(self, prompt: str) -> int:
        ids = [CHAR_TO_IDX[c] for c in prompt.lower() if c in CHAR_TO_IDX]
        if not ids:
            ids = [CHAR_TO_IDX[" "]]
        token = ids[0]
        for token in ids:
            logits = self.step_logits(token)
            token = int(np.argmax(logits))
        return token

    def generate(self, prompt: str, length: int, mode: str, temp: float, top_k: int, seed: int) -> str:
        rng = np.random.default_rng(seed)
        self.reset()
        token = self.prime(prompt)
        out = []
        for _ in range(length):
            logits = self.step_logits(token).astype(np.float64)
            if mode == "greedy":
                token = int(np.argmax(logits))
            else:
                logits = logits / max(temp, 1e-6)
                k = max(1, min(top_k, logits.size))
                top = np.argpartition(logits, -k)[-k:]
                top = top[np.argsort(logits[top])[::-1]]
                z = logits[top] - np.max(logits[top])
                p = np.exp(z); p /= np.sum(p)
                token = int(rng.choice(top, p=p))
            out.append(IDX_TO_CHAR[token])
        return "".join(out)

def score_text(prompt: str, text: str) -> Dict[str, float | int | str]:
    lower = text.lower()
    chars = len(lower)
    unique = len(set(lower))
    alpha = sum(c.isalpha() for c in lower)
    spaces = lower.count(" ")
    words = [w.strip(".,!?'") for w in lower.split()]
    good = sum(1 for w in words if w in GOOD_TERMS)
    bad = sum(1 for frag in BAD_FRAGMENTS if frag in lower)
    repeats = 0
    for n in range(3, 8):
        seen = set()
        for i in range(0, max(0, len(lower)-n+1)):
            gram = lower[i:i+n]
            if gram in seen:
                repeats += 1
            seen.add(gram)
    word_count = max(1, len(words))
    score = 0.0
    score += min(1.0, alpha / max(1, chars)) * 20.0
    score += min(1.0, spaces / max(1, chars) * 8.0) * 10.0
    score += min(1.0, unique / 18.0) * 15.0
    score += min(1.0, good / 3.0) * 25.0
    score -= min(25.0, repeats * 0.55)
    score -= bad * 12.0
    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 2),
        "good_terms": good,
        "repeat_penalty_units": repeats,
        "bad_fragments": bad,
        "word_count": word_count,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights.bin")
    ap.add_argument("--out", default="analysis/usefulness_probe_receipt.json")
    ap.add_argument("--length", type=int, default=96)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = root / weights
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    tensors = parse_rilm(weights)
    param_count = int(sum(np.prod(t.dims) for t in tensors.values()))
    model = RilmLstm(tensors)
    decode_configs = [
        ("greedy", 1.0, 1),
        ("sample", 0.35, 3),
        ("sample", 0.45, 5),
        ("sample", 0.55, 5),
        ("sample", 0.65, 8),
        ("sample", 0.80, 8),
    ]
    samples = []
    for prompt in PROMPTS + POLICY_PROMPTS:
        for mode, temp, top_k in decode_configs:
            text = model.generate(prompt, args.length, mode, temp, top_k, seed=20260701 + len(samples))
            s = score_text(prompt, text)
            samples.append({
                "prompt": prompt,
                "mode": mode,
                "temperature": temp,
                "top_k": top_k,
                "generated": text,
                **s,
            })
    by_config = []
    for mode, temp, top_k in decode_configs:
        xs = [x for x in samples if x["mode"] == mode and x["temperature"] == temp and x["top_k"] == top_k]
        by_config.append({
            "mode": mode,
            "temperature": temp,
            "top_k": top_k,
            "mean_score": round(float(np.mean([x["score"] for x in xs])), 2),
            "best_score": max(x["score"] for x in xs),
            "worst_score": min(x["score"] for x in xs),
        })
    ranked = sorted(samples, key=lambda x: x["score"], reverse=True)
    hardware_tok_s = 3.6864
    estimated_tok_s = hardware_tok_s * (6337569.0 / float(param_count))
    receipt = {
        "schema": "ri-esp32s3-lstm-usefulness-probe-v1",
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "params": param_count,
        "hardware_reference": {
            "firmware_variant": "p7_esp_nn_s3_dot_recurrent",
            "measured_for_params": 6337569,
            "measured_tokens_per_sec": hardware_tok_s,
            "measured_ms_per_token": 271.27,
            "estimated_tokens_per_sec_if_linear_params": round(estimated_tok_s, 4) if param_count != 6337569 else hardware_tok_s,
            "estimate_boundary": "linear parameter scaling only; not hardware verified for this weight pack" if param_count != 6337569 else "hardware verified for this exact H512 weight pack",
        },
        "prompt_count": len(PROMPTS) + len(POLICY_PROMPTS),
        "decode_configs": by_config,
        "top_12_samples": ranked[:12],
        "bottom_8_samples": ranked[-8:],
        "compute_budget_estimates": {
            "16_chars_ms_measured_h512": round(16 / hardware_tok_s * 1000, 1),
            "32_chars_ms_measured_h512": round(32 / hardware_tok_s * 1000, 1),
            "64_chars_ms_measured_h512": round(64 / hardware_tok_s * 1000, 1),
            "96_chars_ms_measured_h512": round(96 / hardware_tok_s * 1000, 1),
            "16_chars_ms_estimated_this_model": round(16 / estimated_tok_s * 1000, 1),
            "32_chars_ms_estimated_this_model": round(32 / estimated_tok_s * 1000, 1),
            "64_chars_ms_estimated_this_model": round(64 / estimated_tok_s * 1000, 1),
            "96_chars_ms_estimated_this_model": round(96 / estimated_tok_s * 1000, 1),
            "usable_interactive_window_chars": "16-48 chars before latency feels long",
        },
        "interpretation": [
            "Current model is impressive as an MCU local generator, but not reliable enough as a free-form policy brain.",
            "Best immediate product use is sentinel/receipt/explanation accent: deterministic rules decide, LSTM emits short local phrasing or status flavor.",
            "For actual usefulness, retrain/domain-distill on sensor/action/status receipts and add constrained decoding/templates.",
        ],
        "elapsed_s": round(time.time() - start, 3),
    }
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out),
        "best_config": max(by_config, key=lambda x: x["mean_score"]),
        "top_sample": ranked[0],
        "budget": receipt["compute_budget_estimates"],
    }, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
