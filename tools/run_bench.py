#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, cwd=None, timeout=None):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True, capture_output=True)


def flash(port: str, cwd: Path):
    p = run(["pio", "run", "-t", "upload", "--upload-port", port], cwd=cwd, timeout=120)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        raise SystemExit(p.returncode)
    return p.stdout + p.stderr


def monitor_once(port: str, timeout_s: int) -> tuple[str, dict]:
    code = (
        "import json, re, serial, sys, time\n"
        f"ser=serial.Serial({port!r},115200,timeout=0.25)\n"
        "ser.setDTR(False); ser.setRTS(False); time.sleep(0.1); ser.setRTS(True); time.sleep(0.1); ser.setRTS(False)\n"
        "deadline=time.time()+" + str(timeout_s) + "\n"
        "buf=[]\n"
        "receipt=None\n"
        "while time.time()<deadline:\n"
        "    data=ser.read(4096)\n"
        "    if data:\n"
        "        txt=data.decode('utf-8','replace')\n"
        "        sys.stdout.write(txt); sys.stdout.flush()\n"
        "        buf.append(txt)\n"
        "        joined=''.join(buf)\n"
        "        m=re.search(r'BENCH_RECEIPT (\\{.*\\})', joined)\n"
        "        if m:\n"
        "            receipt=json.loads(m.group(1)); print('\\n__PARSED_RECEIPT__'+json.dumps(receipt)); break\n"
        "ser.close()\n"
        "if receipt is None: sys.exit(2)\n"
    )
    p = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=timeout_s + 10)
    raw = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"monitor failed exit={p.returncode}\n{raw}")
    marker = "__PARSED_RECEIPT__"
    idx = raw.rfind(marker)
    if idx < 0:
        raise RuntimeError("receipt marker missing\n" + raw)
    receipt_line = raw[idx + len(marker):].strip().splitlines()[0]
    receipt = json.loads(receipt_line)
    return raw, receipt


def aggregate(receipts):
    vals = [float(r["tokens_per_sec"]) for r in receipts]
    ms = [float(r["ms_per_token_mean"]) for r in receipts]
    return {
        "runs": len(receipts),
        "tokens_per_sec_mean": sum(vals) / len(vals),
        "tokens_per_sec_min": min(vals),
        "tokens_per_sec_max": max(vals),
        "ms_per_token_mean": sum(ms) / len(ms),
        "ms_per_token_min": min(ms),
        "ms_per_token_max": max(ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--variant", default="p0_p4_cached_internal_dtype_fastpath")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out-dir", default="benchmarks/current")
    ap.add_argument("--flash", action="store_true")
    args = ap.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    out_dir = cwd / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.flash:
        log = flash(args.port, cwd)
        (out_dir / "flash.log").write_text(log)

    receipts = []
    for i in range(args.repeat):
        print(f"=== run {i+1}/{args.repeat} ===", flush=True)
        raw, receipt = monitor_once(args.port, args.timeout)
        receipt["host_observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        receipt["run_index"] = i + 1
        receipts.append(receipt)
        (out_dir / f"{args.variant}-run-{i+1:03d}.raw.log").write_text(raw)
        (out_dir / f"{args.variant}-run-{i+1:03d}.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    summary = {
        "variant": args.variant,
        "aggregate": aggregate(receipts),
        "receipts": receipts,
    }
    (out_dir / f"{args.variant}-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
