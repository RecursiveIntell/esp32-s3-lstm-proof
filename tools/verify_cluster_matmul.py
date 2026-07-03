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
    r"seq=(?P<seq>\d+)\s+(?:fixture=(?P<fixture>\d+)\s+)?"
    r"worker1=(?P<worker1>-?\d+)\s+worker2=(?P<worker2>-?\d+)\s+"
    r"total=(?P<total>-?\d+)\s+expected=(?P<expected>-?\d+)\s+ok=(?P<ok>true|false)"
)

FIXTURE_INT8_ID = 1
FIXTURE_INT4_ID = 2
VECTOR_LEN = 16


def fixture_vector() -> list[int]:
    return [i - 8 for i in range(VECTOR_LEN)]


def fixture_int8_weight(worker: int, i: int) -> int:
    if worker == 1:
        return i + 1
    if worker == 2:
        return VECTOR_LEN - i
    raise ValueError(f"unsupported worker {worker}")


def clamp_i4(value: int) -> int:
    return min(7, max(-8, value))


def pack_i4_pair(low: int, high: int) -> int:
    return (low & 0x0F) | ((high & 0x0F) << 4)


def unpack_i4_low(packed: int) -> int:
    value = packed & 0x0F
    return value - 16 if value >= 8 else value


def unpack_i4_high(packed: int) -> int:
    value = (packed >> 4) & 0x0F
    return value - 16 if value >= 8 else value


def fixture_int4_weight_unpacked(worker: int, i: int) -> int:
    if worker == 1:
        return clamp_i4((i % 8) - 4)
    if worker == 2:
        return clamp_i4(3 - (i % 8))
    raise ValueError(f"unsupported worker {worker}")


def fixture_int4_weight_packed_byte(worker: int, byte_index: int) -> int:
    low_i = byte_index * 2
    return pack_i4_pair(
        fixture_int4_weight_unpacked(worker, low_i),
        fixture_int4_weight_unpacked(worker, low_i + 1),
    )


def fixture_int4_weight(worker: int, i: int) -> int:
    packed = fixture_int4_weight_packed_byte(worker, i >> 1)
    return unpack_i4_high(packed) if i & 1 else unpack_i4_low(packed)


def fixture_weight(fixture: int, worker: int, i: int) -> int:
    if fixture == FIXTURE_INT8_ID:
        return fixture_int8_weight(worker, i)
    if fixture == FIXTURE_INT4_ID:
        return fixture_int4_weight(worker, i)
    raise ValueError(f"unsupported fixture {fixture}")


def expected_dot(fixture: int, worker: int) -> int:
    return sum(value * fixture_weight(fixture, worker, i) for i, value in enumerate(fixture_vector()))


def parse_line(
    line: str, results: dict[int, dict[int, dict[str, int | str]]], gathers: list[dict[str, int | str | None]]
) -> None:
    result_match = RESULT_RE.search(line)
    if result_match:
        data = {key: int(value) if key != "ok" else value for key, value in result_match.groupdict().items()}
        src = data["src"]
        if src in (1, 2):
            results.setdefault(data["fixture"], {})[src] = data
        return

    gather_match = GATHER_RE.search(line)
    if gather_match:
        data = {}
        for key, value in gather_match.groupdict().items():
            data[key] = value if key == "ok" or value is None else int(value)
        gathers.append(data)


def receipts_complete(
    fixture_ids: list[int], results: dict[int, dict[int, dict[str, int | str]]], gathers: list[dict[str, int | str | None]]
) -> bool:
    for fixture in fixture_ids:
        if set(results.get(fixture, {})) < {1, 2}:
            return False
        if not any(gather_matches_fixture(gather, fixture) for gather in gathers):
            return False
    return True


def gather_matches_fixture(gather: dict[str, int | str | None], fixture: int) -> bool:
    return gather["fixture"] == fixture or (gather["fixture"] is None and fixture == FIXTURE_INT8_ID)


def validate_fixture(
    fixture: int, results: dict[int, dict[int, dict[str, int | str]]], gathers: list[dict[str, int | str | None]]
) -> dict[str, int | str | None]:
    expected = {1: expected_dot(fixture, 1), 2: expected_dot(fixture, 2)}
    expected_total = expected[1] + expected[2]
    fixture_results = results.get(fixture, {})

    missing = sorted(set(expected) - set(fixture_results))
    if missing:
        raise AssertionError(f"fixture {fixture} missing worker result(s): {missing}")

    for worker, dot in expected.items():
        result = fixture_results[worker]
        if result["fixture"] != fixture:
            raise AssertionError(f"worker{worker} fixture mismatch: {result['fixture']} != {fixture}")
        if result["dot"] != dot or result["expected"] != dot or result["ok"] != "true":
            raise AssertionError(f"fixture {fixture} worker{worker} bad result: {result}")

    good_gather = None
    for gather in gathers:
        if (
            gather_matches_fixture(gather, fixture)
            and
            gather["worker1"] == expected[1]
            and gather["worker2"] == expected[2]
            and gather["total"] == expected_total
            and gather["expected"] == expected_total
            and gather["ok"] == "true"
        ):
            good_gather = gather
            break

    if good_gather is None:
        raise AssertionError(f"fixture {fixture} missing valid gather receipt; saw {gathers!r}")

    return good_gather


def validate(
    fixture_ids: list[int], results: dict[int, dict[int, dict[str, int | str]]], gathers: list[dict[str, int | str | None]]
) -> None:
    passes = []
    for fixture in fixture_ids:
        expected = {1: expected_dot(fixture, 1), 2: expected_dot(fixture, 2)}
        expected_total = expected[1] + expected[2]
        good_gather = validate_fixture(fixture, results, gathers)
        passes.append(
            f"fixture={fixture} seq={good_gather['seq']} "
            f"worker1={expected[1]} worker2={expected[2]} total={expected_total}"
        )

    print("PASS cluster matmul " + "; ".join(passes))


def read_from_stdin(
    timeout_s: float, fixture_ids: list[int]
) -> tuple[dict[int, dict[int, dict[str, int | str]]], list[dict[str, int | str | None]]]:
    deadline = time.monotonic() + timeout_s
    results: dict[int, dict[int, dict[str, int | str]]] = {}
    gathers: list[dict[str, int | str | None]] = []
    for line in sys.stdin:
        parse_line(line, results, gathers)
        if receipts_complete(fixture_ids, results, gathers):
            break
        if time.monotonic() >= deadline:
            break
    return results, gathers


def read_from_serial(
    port: str, baud: int, timeout_s: float, fixture_ids: list[int]
) -> tuple[dict[int, dict[int, dict[str, int | str]]], list[dict[str, int | str | None]]]:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required for --port: python3 -m pip install pyserial") from exc

    deadline = time.monotonic() + timeout_s
    results: dict[int, dict[int, dict[str, int | str]]] = {}
    gathers: list[dict[str, int | str | None]] = []
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
                if receipts_complete(fixture_ids, results, gathers):
                    return results, gathers

    if buffer:
        parse_line(buffer.strip(), results, gathers)
    return results, gathers


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify coordinator serial receipts for the WiFi matmul proof.")
    parser.add_argument("--port", help="Coordinator serial port, such as /dev/ttyACM0. If omitted, read stdin.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate. Default: 115200.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for receipts. Default: 60.")
    parser.add_argument(
        "--fixture",
        choices=("1", "2", "both"),
        default="both",
        help="Fixture to validate: 1=int8, 2=packed int4, both=both fixtures. Default: both.",
    )
    args = parser.parse_args()
    fixture_ids = [FIXTURE_INT8_ID, FIXTURE_INT4_ID] if args.fixture == "both" else [int(args.fixture)]

    if args.port:
        results, gathers = read_from_serial(args.port, args.baud, args.timeout, fixture_ids)
    else:
        results, gathers = read_from_stdin(args.timeout, fixture_ids)

    validate(fixture_ids, results, gathers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
