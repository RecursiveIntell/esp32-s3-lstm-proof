#!/usr/bin/env python3

import argparse
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
    "infer": {
        "coord": "cluster_coord_ap_infer",
        "worker1": "cluster_worker1_ap_infer",
        "worker2": "cluster_worker2_ap_infer",
    },
}


def build_command(role: str, port: str, mode: str) -> list[str]:
    return ["pio", "run", "-e", ROLE_ENVS[mode][role], "-t", "upload", "--upload-port", port]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or run the PlatformIO upload command for one cluster WiFi board."
    )
    parser.add_argument("--role", choices=sorted(ROLE_ENVS["ping"]), required=True, help="Board role to flash.")
    parser.add_argument(
        "--mode",
        choices=sorted(ROLE_ENVS),
        default="ping",
        help="Cluster WiFi firmware mode to flash. Default: ping.",
    )
    parser.add_argument("--port", required=True, help="Serial port for the board, such as /dev/ttyACM0.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the upload command. The default is a dry-run that only prints it.",
    )
    args = parser.parse_args()

    env = ROLE_ENVS[args.mode][args.role]
    cmd = build_command(args.role, args.port, args.mode)
    printable = " ".join(shlex.quote(part) for part in cmd)
    mode = "EXECUTE" if args.execute else "DRY_RUN"
    print(f"{mode} role={args.role} mode={args.mode} env={env}")
    print(printable)

    if not args.execute:
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
