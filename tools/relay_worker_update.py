#!/usr/bin/env python3

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

import serial

ROLE_ENVS = {
    "ping": {
        "worker1": "cluster_worker1_ap_ping",
        "worker2": "cluster_worker2_ap_ping",
    },
    "matmul": {
        "worker1": "cluster_worker1_ap_matmul",
        "worker2": "cluster_worker2_ap_matmul",
    },
    "infer": {
        "worker1": "cluster_worker1_ap_infer",
        "worker2": "cluster_worker2_ap_infer",
    },
}


def build_command(env: str) -> list[str]:
    return ["pio", "run", "-e", env]


def firmware_path(env: str) -> Path:
    return Path(".pio") / "build" / env / "firmware.bin"


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def wait_for_worker_ip(ser: serial.Serial, board: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    needles = (
        f"CLUSTER_MATMUL_RESULT src_board={board} ",
        f"CLUSTER_WIFI_PONG src_board={board} ",
        f"CLUSTER_FC_RESULT src_board={board} ",
    )
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").rstrip()
        print(line)
        if any(needle in line for needle in needles):
            return
    raise TimeoutError(f"timed out waiting for coordinator to observe worker{board} IP")


def send_relay_update(ser: serial.Serial, board: int, firmware: Path, timeout: float) -> bool:
    size = firmware.stat().st_size
    command = f"CLUSTER_RELAY_UPDATE board={board} size={size}\n".encode()
    print(f"SERIAL: {command.decode().rstrip()}")
    ser.write(command)
    ser.flush()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").rstrip()
        print(line)
        if "CLUSTER_RELAY_UPDATE_READY_FOR_BYTES" in line:
            break
        if "CLUSTER_RELAY_UPDATE_ERROR" in line:
            return False
    else:
        raise TimeoutError("timed out waiting for CLUSTER_RELAY_UPDATE_READY_FOR_BYTES")

    with firmware.open("rb") as f:
        sent = 0
        while True:
            # Pace at the coordinator's relay buffer size. Bursting larger chunks can
            # overrun CDC while the coordinator is also forwarding to WiFi, causing a
            # late serial_read timeout after most of the image has been received.
            chunk = f.read(1024)
            if not chunk:
                break
            ser.write(chunk)
            ser.flush()
            sent += len(chunk)
            if sent % 65536 < len(chunk) or sent == size:
                print(f"HOST_RELAY_PROGRESS sent={sent} total={size}")
            time.sleep(0.01)
        ser.flush()

    ok = False
    saw_end = False
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").rstrip()
        print(line)
        if "CLUSTER_RELAY_UPDATE_END" in line:
            saw_end = True
            ok = " ok=1 " in f" {line} "
            break
        if "CLUSTER_RELAY_UPDATE_ERROR" in line:
            saw_end = True
            ok = False
            break
    if not saw_end:
        raise TimeoutError("timed out waiting for CLUSTER_RELAY_UPDATE_END")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a worker image and stream it through the USB-attached coordinator to a worker HTTP /update endpoint."
    )
    parser.add_argument("--role", choices=sorted(ROLE_ENVS["matmul"]), required=True)
    parser.add_argument("--mode", choices=sorted(ROLE_ENVS), default="matmul")
    parser.add_argument("--port", required=True, help="Coordinator serial port, such as /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--wait-worker", action="store_true", help="Wait until coordinator has observed the target worker IP.")
    parser.add_argument("--wait-timeout", type=float, default=90.0)
    parser.add_argument("--relay-timeout", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true", help="Build and relay. Default is dry-run.")
    args = parser.parse_args()

    board = 1 if args.role == "worker1" else 2
    env = ROLE_ENVS[args.mode][args.role]
    build_cmd = build_command(env)
    firmware = firmware_path(env)

    mode = "EXECUTE" if args.execute else "DRY_RUN"
    print(f"{mode} role={args.role} board={board} mode={args.mode} env={env} port={args.port}")
    print(f"BUILD: {format_command(build_cmd)}")
    print(f"FIRMWARE: {firmware}")
    print(f"SERIAL_RELAY: CLUSTER_RELAY_UPDATE board={board} size=<firmware-bytes>")

    if not args.execute:
        return 0

    build_result = subprocess.run(build_cmd)
    if build_result.returncode != 0:
        return build_result.returncode
    if not firmware.is_file():
        print(f"ERROR: firmware artifact not found: {firmware}", file=sys.stderr)
        return 2

    with serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=10) as ser:
        ser.reset_input_buffer()
        if args.wait_worker:
            wait_for_worker_ip(ser, board, args.wait_timeout)
        ok = send_relay_update(ser, board, firmware, args.relay_timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
