#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys


ROLE_ENVS = {
    "coord": "cluster_coord_ap_ping",
    "worker1": "cluster_worker1_ap_ping",
    "worker2": "cluster_worker2_ap_ping",
}


def build_command(role: str, port: str) -> list[str]:
    return ["pio", "run", "-e", ROLE_ENVS[role], "-t", "upload", "--upload-port", port]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or run the PlatformIO upload command for one cluster WiFi ping board."
    )
    parser.add_argument("--role", choices=sorted(ROLE_ENVS), required=True, help="Board role to flash.")
    parser.add_argument("--port", required=True, help="Serial port for the board, such as /dev/ttyACM0.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the upload command. The default is a dry-run that only prints it.",
    )
    args = parser.parse_args()

    cmd = build_command(args.role, args.port)
    printable = " ".join(shlex.quote(part) for part in cmd)
    mode = "EXECUTE" if args.execute else "DRY_RUN"
    print(f"{mode} role={args.role} env={ROLE_ENVS[args.role]}")
    print(printable)

    if not args.execute:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
