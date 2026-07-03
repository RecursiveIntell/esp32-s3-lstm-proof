#!/usr/bin/env python3
import argparse
import json
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


def _extract_balanced_json(text: str, start: int) -> tuple[str | None, int | None]:
    """Extract a JSON object from text[start:] with brace-aware scanning.

    Returns (json_text, end_index) or (None, None) if not yet complete.
    """
    if start < 0:
        return None, None

    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None, None

    depth = 0
    in_string = False
    escaped = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1], j + 1
        if depth < 0:
            return None, None

    return None, None


def _extract_bench_receipt(joined: str) -> tuple[str | None, int | None]:
    marker = "BENCH_RECEIPT "
    marker_idx = joined.find(marker)
    if marker_idx < 0:
        return None, None

    json_text, end = _extract_balanced_json(joined, marker_idx + len(marker))
    return json_text, end


def _test_receipt_parser() -> None:
    examples = [
        (
            'noise\nBENCH_RECEIPT {"tokens_per_sec": 1.2}\nPROOF_DONE\n',
            '{"tokens_per_sec": 1.2}',
        ),
        (
            'xBENCH_RECEIPT {"a": {"b": [1, {"c": "x"}]}, "d": "we \\"{ braces }\\""} trailing',
            '{"a": {"b": [1, {"c": "x"}]}, "d": "we \\"{ braces }\\""}',
        ),
    ]
    for raw, expected_json in examples:
        json_text, _ = _extract_bench_receipt(raw)
        if json_text != expected_json:
            raise AssertionError(f"unexpected parse result: {json_text!r} != {expected_json!r}")
        parsed = json.loads(json_text)
        assert isinstance(parsed, dict)


def monitor_once(port: str, timeout_s: int) -> tuple[str, dict]:
    import serial

    ser = serial.Serial(port, 115200, timeout=0.25)
    ser.setDTR(False)
    ser.setRTS(False)
    time.sleep(0.1)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    deadline = time.time() + timeout_s
    buf = []
    receipt = None
    receipt_seen_at = None
    proof_done_seen = False
    raw = ""

    try:
        while time.time() < deadline:
            data = ser.read(4096)
            if not data:
                continue
            txt = data.decode("utf-8", "replace")
            sys.stdout.write(txt)
            sys.stdout.flush()
            buf.append(txt)
            raw = "".join(buf)

            if receipt_seen_at is None:
                marker_idx = raw.find("BENCH_RECEIPT ")
                if marker_idx >= 0:
                    receipt_seen_at = marker_idx + len("BENCH_RECEIPT ")

            if receipt_seen_at is not None:
                if not proof_done_seen and "PROOF_DONE" in raw[receipt_seen_at:]:
                    proof_done_seen = True
                json_text, _ = _extract_balanced_json(raw, receipt_seen_at)
                if json_text is not None:
                    receipt = json.loads(json_text)
                    break
                if proof_done_seen:
                    break

        if receipt is None:
            raise RuntimeError("balanced receipt JSON not found before timeout or PROOF_DONE")
    finally:
        ser.close()

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
