#!/usr/bin/env python3

import argparse
import re
import sys
import time

FC_GATHER_RE = re.compile(
    r"CLUSTER_FC_GATHER\s+"
    r"seq=(?P<seq>\d+)\s+prompt_id=(?P<prompt_id>\d+)\s+prompt=\"(?P<prompt>[^\"]*)\"\s+"
    r"worker1_token=(?P<w1_token>\d+)\s+worker1_logit=(?P<w1_logit>-?\d+(?:\.\d+)?)\s+"
    r"worker2_token=(?P<w2_token>\d+)\s+worker2_logit=(?P<w2_logit>-?\d+(?:\.\d+)?)\s+"
    r"global_token=(?P<global_token>\d+)\s+global_char=(?P<global_char>.)\s+global_logit=(?P<global_logit>-?\d+(?:\.\d+)?)\s+"
    r"local_token=(?P<local_token>\d+)\s+local_char=(?P<local_char>.)\s+local_logit=(?P<local_logit>-?\d+(?:\.\d+)?)\s+ok=(?P<ok>true|false)"
)

MODEL_READY_RE = re.compile(r"CLUSTER_MODEL_READY\s+board_id=(?P<board>\d+)\s+ok=(?P<ok>\d+)\s+role=(?P<role>\w+)")


def parse_line(line: str, gathers: list[dict], ready: dict[int, bool]) -> None:
    m = MODEL_READY_RE.search(line)
    if m:
        ready[int(m.group("board"))] = m.group("ok") == "1"
        return
    m = FC_GATHER_RE.search(line)
    if m:
        d = m.groupdict()
        for k in ("seq", "prompt_id", "w1_token", "w2_token", "global_token", "local_token"):
            d[k] = int(d[k])
        for k in ("w1_logit", "w2_logit", "global_logit", "local_logit"):
            d[k] = float(d[k])
        gathers.append(d)


def complete(gathers: list[dict], min_gathers: int) -> bool:
    return len([g for g in gathers if g["ok"] == "true"]) >= min_gathers


def read_from_serial(port: str, baud: int, timeout_s: float, min_gathers: int) -> tuple[list[dict], dict[int, bool]]:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port: python3 -m pip install pyserial") from exc
    deadline = time.monotonic() + timeout_s
    gathers: list[dict] = []
    ready: dict[int, bool] = {}
    buf = ""
    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        while time.monotonic() < deadline:
            chunk = ser.read(512)
            if not chunk:
                continue
            text = chunk.decode("utf-8", errors="ignore")
            sys.stdout.write(text)
            sys.stdout.flush()
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                parse_line(line.strip(), gathers, ready)
                if complete(gathers, min_gathers):
                    return gathers, ready
    if buf:
        parse_line(buf.strip(), gathers, ready)
    return gathers, ready


def read_from_stdin(timeout_s: float, min_gathers: int) -> tuple[list[dict], dict[int, bool]]:
    deadline = time.monotonic() + timeout_s
    gathers: list[dict] = []
    ready: dict[int, bool] = {}
    for line in sys.stdin:
        parse_line(line.strip(), gathers, ready)
        if complete(gathers, min_gathers) or time.monotonic() >= deadline:
            break
    return gathers, ready


def validate(gathers: list[dict], ready: dict[int, bool], min_gathers: int) -> None:
    bad_ready = sorted(board for board, ok in ready.items() if not ok)
    if bad_ready:
        raise AssertionError(f"model not ready on boards: {bad_ready}")
    good = [g for g in gathers if g["ok"] == "true" and g["global_token"] == g["local_token"]]
    if len(good) < min_gathers:
        raise AssertionError(f"need {min_gathers} good FC gathers; saw {len(good)} good of {len(gathers)} total: {gathers!r}")
    parts = []
    for g in good[:min_gathers]:
        parts.append(
            f"seq={g['seq']} prompt_id={g['prompt_id']} prompt=\"{g['prompt']}\" "
            f"global_token={g['global_token']} global_char={g['global_char']} "
            f"local_token={g['local_token']} local_char={g['local_char']}"
        )
    print("PASS cluster sharded_fc_inference " + "; ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify coordinator receipts for sharded FC inference.")
    ap.add_argument("--port", help="Coordinator serial port. If omitted, read stdin.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--min-gathers", type=int, default=2)
    args = ap.parse_args()
    if args.port:
        gathers, ready = read_from_serial(args.port, args.baud, args.timeout, args.min_gathers)
    else:
        gathers, ready = read_from_stdin(args.timeout, args.min_gathers)
    validate(gathers, ready, args.min_gathers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
