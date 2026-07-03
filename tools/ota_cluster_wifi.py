#!/usr/bin/env python3

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROLE_ENVS = {
    "ping": {
        "coord": "cluster_coord_ap_ping",
        "worker1": "cluster_worker1_ap_ping",
        "worker2": "cluster_worker2_ap_ping",
    },
    "matmul": {
        "coord": "cluster_coord_ap_matmul",
        "worker1": "cluster_worker1_ap_matmul",
        "worker2": "cluster_worker2_ap_matmul",
    },
}

DEFAULT_ESPOTA = Path.home() / ".platformio/packages/framework-arduinoespressif32/tools/espota.py"
PLATFORMIO_PACKAGES = Path.home() / ".platformio/packages"


def build_command(env: str) -> list[str]:
    return ["pio", "run", "-e", env]


def firmware_path(env: str) -> Path:
    return Path(".pio") / "build" / env / "firmware.bin"


def find_espota() -> Path:
    if DEFAULT_ESPOTA.is_file():
        return DEFAULT_ESPOTA
    if PLATFORMIO_PACKAGES.is_dir():
        matches = sorted(PLATFORMIO_PACKAGES.rglob("espota.py"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "espota.py not found under ~/.platformio/packages/framework-arduinoespressif32/tools "
        "or by searching ~/.platformio/packages"
    )


def espota_command(espota: Path, ip: str, password: str, firmware: Path) -> list[str]:
    return [
        sys.executable,
        str(espota),
        "-i",
        ip,
        "-p",
        "3232",
        "-a",
        password,
        "-f",
        str(firmware),
    ]


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and optionally OTA-upload one ESP32-S3 cluster WiFi firmware image."
    )
    parser.add_argument("--role", choices=sorted(ROLE_ENVS["ping"]), required=True, help="Board role to update.")
    parser.add_argument(
        "--mode",
        choices=sorted(ROLE_ENVS),
        default="ping",
        help="Cluster WiFi firmware mode to build. Default: ping.",
    )
    parser.add_argument("--ip", required=True, help="Target board IP address, such as 192.168.4.1.")
    parser.add_argument(
        "--password",
        default="localfirstai",
        help="ArduinoOTA password. Default: localfirstai.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Build and upload. The default is a dry-run that only prints commands.",
    )
    args = parser.parse_args()

    env = ROLE_ENVS[args.mode][args.role]
    build_cmd = build_command(env)
    firmware = firmware_path(env)

    try:
        espota = find_espota()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    upload_cmd = espota_command(espota, args.ip, args.password, firmware)
    mode = "EXECUTE" if args.execute else "DRY_RUN"
    print(f"{mode} role={args.role} mode={args.mode} env={env} ip={args.ip}")
    print(f"BUILD: {format_command(build_cmd)}")
    print(f"ESPOTA: {format_command(upload_cmd)}")

    if not args.execute:
        return 0

    build_result = subprocess.run(build_cmd)
    if build_result.returncode != 0:
        return build_result.returncode

    if not firmware.is_file():
        print(f"ERROR: firmware artifact not found: {firmware}", file=sys.stderr)
        return 2

    upload_result = subprocess.run(upload_cmd)
    return upload_result.returncode


if __name__ == "__main__":
    sys.exit(main())
