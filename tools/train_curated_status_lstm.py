#!/usr/bin/env python3
"""Curated ESP32-S3 status/action char-LSTM training.

Purpose: train a compact but higher-capacity domain model (default H320x3) for
receipt-backed physical sentinel language. This deliberately avoids generic story
text. It teaches short, deterministic continuations that an ESP32 can emit in
16-64 chars after a rule/policy layer chooses the route.

Exports RILM v1 all-int8/mixed weights compatible with the existing host probe
and firmware parser after HIDDEN is set to the chosen width.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = list("abcdefghijklmnopqrstuvwxyz .,!?'\n")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
MAGIC = 0x4D4C4952

# Canonical prompt -> desired short continuation. The LSTM should phrase a
# decision already made by deterministic policy, not infer policy by itself.
POLICY_CASES = [
    ("normal room. action is ", "log receipt."),
    ("safe room. action is ", "log receipt."),
    ("room normal. action is ", "log receipt."),
    ("hot room. action is ", "check airflow."),
    ("warm room. action is ", "watch temperature."),
    ("very hot room. action is ", "escalate."),
    ("humid room. action is ", "ventilate."),
    ("hot humid room. action is ", "escalate."),
    ("high heat and humidity. action is ", "escalate."),
    ("dry room. action is ", "add humidity."),
    ("cold room. action is ", "watch temperature."),
    ("stale data. action is ", "wait for fresh data."),
    ("old reading. action is ", "wait for fresh data."),
    ("missing sensor. action is ", "no claim."),
    ("sensor missing. action is ", "no claim."),
    ("bad reading. action is ", "no claim."),
    ("unsafe evidence. action is ", "no claim."),
    ("local confidence low. action is ", "ask gateway."),
    ("unknown room. action is ", "ask gateway."),
    ("safe action is ", "no claim without evidence."),
    ("do not invent ", "readings."),
    ("if data is stale ", "say stale data."),
    ("if sensor is missing ", "say no claim."),
    ("the receipt says ", "local confident."),
    ("the receipt says sensor ", "missing."),
    ("the receipt says stale ", "reading."),
    ("the gateway should ", "verify the claim."),
    ("local first means ", "decide before cloud."),
    ("the oled says ", "short local status."),
    ("the esp32 sees ", "temperature and humidity."),
]

ROUTES = [
    ("local_confident", "normal room", "log receipt", "local confident"),
    ("temperature_out_of_range", "hot room", "check airflow", "heat warning"),
    ("humidity_out_of_range", "humid room", "ventilate", "humidity warning"),
    ("temperature_and_humidity_out_of_range", "hot humid room", "escalate", "heat humidity warning"),
    ("sensor_missing", "missing sensor", "no claim", "sensor missing"),
    ("stale_reading", "stale data", "wait for fresh data", "stale reading"),
    ("low_confidence", "uncertain room", "ask gateway", "low confidence"),
    ("operator_request", "operator asked", "ask gateway", "operator request"),
]

TEMPS_F = [60, 64, 68, 72, 76, 80, 84, 88, 92]
HUMS = [22, 30, 38, 46, 54, 62, 70, 78, 86]
AGES = [1, 3, 8, 15, 45, 90, 240]

@dataclass(frozen=True)
class CorpusSplits:
    train: str
    val: str
    test: str
    manifest: Dict[str, object]

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
                # Forget gate bias = 1 for stable LSTM memory.
                p.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x, hidden=None):
        y, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(y), hidden

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

def clean(text: str) -> str:
    text = text.lower()
    text = text.replace("%", " percent ")
    text = re.sub(r"[^a-z .,!?\n']+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "".join(ch for ch in text if ch in CHAR_TO_IDX)

def line_variants(prompt: str, cont: str) -> List[str]:
    full = (prompt + cont).strip()
    bare = full.rstrip(".") + "."
    return [
        bare,
        f"receipt text. {bare}",
        f"oled text. {bare}",
        f"local status. {bare}",
        f"policy says. {bare}",
    ]

def route_templates(route: str, status: str, action: str, receipt: str, rng: random.Random) -> List[str]:
    t = rng.choice(TEMPS_F)
    h = rng.choice(HUMS)
    age = rng.choice(AGES)
    conf = rng.choice([52, 61, 73, 81, 88, 93, 97])
    return [
        f"route {route}. status {status}. action is {action}.",
        f"decision {route}. reason {receipt}. action is {action}.",
        f"temperature {t}f humidity {h} percent. status {status}. action is {action}.",
        f"reading age {age} seconds. route {route}. receipt says {receipt}.",
        f"confidence {conf} percent. {status}. action is {action}.",
        f"the receipt says {receipt}. action is {action}.",
        f"{status}. action is {action}.",
    ]

def build_curated_corpus(repeats: int, seed: int) -> CorpusSplits:
    rng = random.Random(seed)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    # Hold out some exact cases to test generalization, not memorization only.
    holdout_prompts = {
        "very hot room. action is ",
        "old reading. action is ",
        "unsafe evidence. action is ",
        "the gateway should ",
    }
    test_prompts = {
        "unknown room. action is ",
        "if sensor is missing ",
        "the receipt says stale ",
    }

    for prompt, cont in POLICY_CASES:
        variants = line_variants(prompt, cont)
        target = test if prompt in test_prompts else val if prompt in holdout_prompts else train
        weight = 140 if target is train else 35
        for _ in range(weight):
            target.extend(variants)

    for i in range(repeats):
        route, status, action, receipt = rng.choice(ROUTES)
        lines = route_templates(route, status, action, receipt, rng)
        # Add negative controls aggressively: no sensor/evidence means no claim.
        if rng.random() < 0.25:
            lines.extend([
                "missing sensor. action is no claim.",
                "bad reading. action is no claim.",
                "do not invent readings.",
                "safe action is no claim without evidence.",
            ])
        if rng.random() < 0.15:
            lines.extend([
                "stale data. action is wait for fresh data.",
                "if data is stale say stale data.",
            ])
        bucket = test if i % 31 == 0 else val if i % 13 == 0 else train
        bucket.extend(lines)

    # Natural language but still domain-constrained. No generic stories.
    extra = [
        "the esp32 is a local sentinel. it reports short status text.",
        "the rule layer decides. the model phrases the status.",
        "the gateway handles rich reasoning. the esp32 keeps evidence local.",
        "do not claim a reading when the sensor is missing.",
        "do not invent temperature or humidity.",
        "short local text should fit on the oled.",
        "receipt first. action second. cloud only when needed.",
        "local status can be hot humid normal stale or missing.",
        "a stale sensor must not make a confident claim.",
        "a missing sensor must say no claim.",
    ]
    for _ in range(max(200, repeats // 8)):
        train.append(rng.choice(extra))

    for xs in (train, val, test):
        rng.shuffle(xs)

    def join(xs: List[str]) -> str:
        return clean("\n".join(xs) + "\n")

    manifest = {
        "schema": "ri_esp32_curated_status_corpus_v1",
        "seed": seed,
        "policy_cases": len(POLICY_CASES),
        "routes": len(ROUTES),
        "train_lines": len(train),
        "val_lines": len(val),
        "test_lines": len(test),
        "heldout_prompts": sorted(holdout_prompts),
        "test_prompts": sorted(test_prompts),
        "intent": "short receipt/action/status text for deterministic ESP32-S3 sentinel policy; not chatbot/story text",
    }
    return CorpusSplits(join(train), join(val), join(test), manifest)

def encode(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=np.int64)

def make_batch(data: np.ndarray, seq_len: int, batch: int, rng: np.random.Generator):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch)
    x = np.stack([data[i:i+seq_len] for i in starts])
    y = np.stack([data[i+1:i+seq_len+1] for i in starts])
    return torch.from_numpy(x), torch.from_numpy(y)

def eval_loss(model: nn.Module, data: np.ndarray, device: str, seq_len: int, chunks: int = 128) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
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

def generate(model: nn.Module, prompt: str, length: int, device: str, mode: str = "greedy", temp: float = 0.55, top_k: int = 5) -> str:
    model.eval()
    ids = [CHAR_TO_IDX[c] for c in clean(prompt) if c in CHAR_TO_IDX] or [CHAR_TO_IDX[" "]]
    hidden = None
    with torch.no_grad():
        for ch in ids[:-1]:
            _, hidden = model(torch.tensor([[ch]], device=device), hidden)
        cur = ids[-1]
        out = []
        for _ in range(length):
            logits, hidden = model(torch.tensor([[cur]], device=device), hidden)
            logits = logits[0, -1]
            if mode == "greedy":
                cur = int(torch.argmax(logits).item())
            else:
                vals, idx = torch.topk(logits / temp, k=min(top_k, VOCAB_SIZE))
                probs = torch.softmax(vals, dim=-1)
                cur = int(idx[torch.multinomial(probs, 1).item()].item())
            out.append(IDX_TO_CHAR[cur])
    model.train()
    return "".join(out)

def eval_policy_continuations(model: nn.Module, device: str, length: int = 48) -> Dict[str, object]:
    cases = []
    exact = 0
    contains = 0
    for prompt, expected in POLICY_CASES:
        got = generate(model, prompt, length, device, "greedy")
        norm_got = clean(got).strip()
        norm_exp = clean(expected).strip()
        ok_prefix = norm_got.startswith(norm_exp[: min(8, len(norm_exp))])
        ok_contains = norm_exp.split(".")[0] in norm_got
        exact += int(ok_prefix)
        contains += int(ok_contains)
        cases.append({"prompt": prompt, "expected": expected, "generated": got, "prefix_ok": ok_prefix, "contains_ok": ok_contains})
    return {
        "prompt_count": len(POLICY_CASES),
        "prefix_accuracy": exact / len(POLICY_CASES),
        "contains_accuracy": contains / len(POLICY_CASES),
        "cases": cases,
    }

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
    return {"path": str(out_path), "bytes": len(out), "sha256": hashlib.sha256(out).hexdigest(), "profile": profile, "payload_bytes": payload_bytes}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.8e-3)
    ap.add_argument("--repeats", type=int, default=9000)
    ap.add_argument("--out", default="runs/curated_status_h320_l3")
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--profiles", default="all_int8,mixed_lstm_safe")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 4)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()

    splits = build_curated_corpus(args.repeats, args.seed)
    (out / "curated_train.txt").write_text(splits.train, encoding="utf-8")
    (out / "curated_val.txt").write_text(splits.val, encoding="utf-8")
    (out / "curated_test.txt").write_text(splits.test, encoding="utf-8")
    (out / "corpus_manifest.json").write_text(json.dumps(splits.manifest, indent=2, sort_keys=True), encoding="utf-8")
    train = encode(splits.train); val = encode(splits.val); test = encode(splits.test)
    rng = np.random.default_rng(args.seed)

    model = CharLSTM(args.hidden, args.layers, dropout=0.04).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.08)
    best_loss = float("inf")
    log: List[Dict[str, object]] = []
    latest = out / "latest.pt"
    best = out / "best.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"]); opt.load_state_dict(ckpt["opt_state"]); sched.load_state_dict(ckpt["sched_state"])
        best_loss = float(ckpt.get("best_loss", best_loss)); log = ckpt.get("log", [])
        start_step = int(ckpt.get("step", 0)) + 1
    else:
        start_step = 0

    print(json.dumps({
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params": model.count_params(),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "test_tokens": int(len(test)),
        "args": vars(args),
        "manifest": splits.manifest,
    }, indent=2), flush=True)

    for step in range(start_step, args.steps + 1):
        xb, yb = make_batch(train, args.seq_len, args.batch_size, rng)
        xb = xb.to(device); yb = yb.to(device)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == args.steps:
            vl = eval_loss(model, val, device, args.seq_len)
            tl = eval_loss(model, test, device, args.seq_len)
            policy = eval_policy_continuations(model, device, length=40)
            entry = {
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": vl,
                "val_ppl": math.exp(vl) if vl < 20 else None,
                "test_loss": tl,
                "test_ppl": math.exp(tl) if tl < 20 else None,
                "policy_prefix_accuracy": policy["prefix_accuracy"],
                "policy_contains_accuracy": policy["contains_accuracy"],
                "lr": float(sched.get_last_lr()[0]),
                "elapsed_s": time.time() - start,
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
            for p in ["hot room. action is ", "missing sensor. action is ", "the receipt says ", "high heat and humidity. action is "]:
                print("SAMPLE " + p + generate(model, p, 48, device, "greedy").replace("\n", "\\n"), flush=True)
            (out / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(), "sched_state": sched.state_dict(), "args": vars(args), "vocab": VOCAB, "best_loss": best_loss, "step": step, "log": log}, latest)
            if vl < best_loss:
                best_loss = vl
                torch.save({"model_state": model.state_dict(), "args": vars(args), "vocab": VOCAB, "best_val_loss": best_loss, "params": model.count_params(), "policy_eval": policy}, best)

    ckpt = torch.load(best, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_val_loss = eval_loss(model, val, device, args.seq_len)
    final_test_loss = eval_loss(model, test, device, args.seq_len)
    policy_eval = eval_policy_continuations(model, device, length=48)
    exports = []
    for profile in [x.strip() for x in args.profiles.split(",") if x.strip()]:
        exports.append(export_rilm(model, out / "weights" / f"curated_status_h{args.hidden}_l{args.layers}_{profile}.bin", profile))
    samples = {prompt: generate(model, prompt, 80, device, "greedy") for prompt, _ in POLICY_CASES}
    summary = {
        "schema": "ri_esp32_curated_status_lstm_train_v1",
        "hidden": args.hidden,
        "layers": args.layers,
        "params": model.count_params(),
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "corpus_manifest": splits.manifest,
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "test_tokens": int(len(test)),
        "best_val_loss": float(best_loss),
        "best_val_ppl": math.exp(best_loss) if best_loss < 20 else None,
        "final_val_loss": final_val_loss,
        "final_val_ppl": math.exp(final_val_loss) if final_val_loss < 20 else None,
        "final_test_loss": final_test_loss,
        "final_test_ppl": math.exp(final_test_loss) if final_test_loss < 20 else None,
        "policy_eval": policy_eval,
        "exports": exports,
        "samples": samples,
        "elapsed_s": time.time() - start,
        "claim_boundary": "MSI/GTX-trained and RILM-exported. Not an ESP32 hardware claim until flashed and BENCH_RECEIPT captured.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
