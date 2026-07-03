#!/usr/bin/env python3

import argparse
import re
import sys
import time


RESULT_RE = re.compile(
    r"CLUSTER_MATMUL_RESULT\s+"
    r"src_board=(?P<src>\d+)\s+seq=(?P<seq>\d+)\s+fixture=(?P<fixture>\d+)\s+"
    r"dot=(?P<dot>-?\d+)\s+expected=(?P<expected>-?\d+)\s+ok=(?P<ok>true|false)"
)
GATHER_RE = re.compile(
    r"CLUSTER_MATMUL_GATHER\s+"
    r"seq=(?P<seq>\d+)\s+worker1=(?P<worker1>-?\d+)\s+worker2=(?P<worker2>-?\d+)\s+"
    r"total=(?P<total>-?\d+)\s+expected=(?P<expected>-?\d+)\s+ok=(?P<ok>true|false)"
)

FIXTURE_ID = 1
VECTOR_LEN = 16


def fixture_vector() -> list[int]:
    return [i - 8 for i in range(VECTOR_LEN)]


def fixture_weight(worker: int, i: int) -> int:
    if worker == 1:
        return i + 1
    if worker == 2:
        return VECTOR_LEN - i
    raise ValueError(f"unsupported worker {worker}")


def expected_dot(worker: int) -> int:
    return sum(value * fixture_weight(worker, i) for i, value in enumerate(fixture_vector()))


def parse_line(line: str, results: dict[int, dict[str, int]], gathers: list[dict[str, int]]) -> None:
    result_match = RESULT_RE.search(line)
    if result_match:
        data = {key: int(value) if key != "ok" else value for key, value in result_match.groupdict().items()}
        src = data["src"]
        if src in (1, 2):
            results[src] = data
        return

    gather_match = GATHER_RE.search(line)
    if gather_match:
        data = {key: int(value) if key != "ok" else value for key, value in gather_match.groupdict().items()}
        gathers.append(data)


def validate(results: dict[int, dict[str, int]], gathers: list[dict[str, int]]) -> None:
    expected = {1: expected_dot(1), 2: expected_dot(2)}
    expected_total = expected[1] + expected[2]

    missing = sorted(set(expected) - set(results))
    if missing:
        raise AssertionError(f"missing worker result(s): {missing}")

    for worker, dot in expected.items():
        result = results[worker]
        if result["fixture"] != FIXTURE_ID:
            raise AssertionError(f"worker{worker} fixture mismatch: {result['fixture']} != {FIXTURE_ID}")
        if result["dot"] != dot or result["expected"] != dot or result["ok"] != "true":
            raise AssertionError(f"worker{worker} bad result: {result}")

    good_gather = None
    for gather in gathers:
        if (
            gather["worker1"] == expected[1]
            and gather["worker2"] == expected[2]
            and gather["total"] == expected_total
            and gather["expected"] == expected_total
            and gather["ok"] == "true"
        ):
            good_gather = gather
            break

    if good_gather is None:
        raise AssertionError(f"missing valid gather receipt; saw {gathers!r}")

    print(
        "PASS cluster matmul "
        f"seq={good_gather['seq']} worker1={expected[1]} worker2={expected[2]} total={expected_total}"
    )


def read_from_stdin(timeout_s: float) -> tuple[dict[int, dict[str, int]], list[dict[str, int]]]:
    deadline = time.monotonic() + timeout_s
    results: dict[int, dict[str, int]] = {}
    gathers: list[dict[str, int]] = []
    for line in sys.stdin:
        parse_line(line, results, gathers)
        if set(results) >= {1, 2} and gathers:
            break
        if time.monotonic() >= deadline:
            break
    return results, gathers


def read_from_serial(port: str, baud: int, timeout_s: float) -> tuple[dict[int, dict[str, int]], list[dict[str, int]]]:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port: python3 -m pip install pyserial") from exc

    deadline = time.monotonic() + timeout_s
    results: dict[int, dict[str, int]] = {}
    gathers: list[dict[str, int]] = []
    buffer = ""

    with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
        while time.monotonic() < deadline:
            chunk = ser.read(256)
            if not chunk:
                continue
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                parse_line(line.strip(), results, gathers)
                if set(results) >= {1, 2} and gathers:
                    return results, gathers

    if buffer:
        parse_line(buffer.strip(), results, gathers)
    return results, gathers


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify coordinator serial receipts for the WiFi matmul proof.")
    parser.add_argument("--port", help="Coordinator serial port, such as /dev/ttyACM0. If omitted, read stdin.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate. Default: 115200.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for receipts. Default: 60.")
    args = parser.parse_args()

    if args.port:
        results, gathers = read_from_serial(args.port, args.baud, args.timeout)
    else:
        results, gathers = read_from_stdin(args.timeout)

    validate(results, gathers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
