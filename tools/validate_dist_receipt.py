#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(
    r"CLUSTER_DIST_GEN_TOKEN\s+.*?"
    r"local_p22_token=(?P<local>\d+)\s+.*?"
    r"dist_token=(?P<dist>\d+)\s+.*?"
    r"status=(?P<status>PASS|FAIL)"
)


def validate_text(text: str) -> list[str]:
    errors: list[str] = []
    token_lines = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        match = TOKEN_RE.search(line)
        if not match:
            continue
        token_lines += 1
        local = int(match.group("local"))
        dist = int(match.group("dist"))
        status = match.group("status")
        expected = "PASS" if local == dist else "FAIL"
        if status != expected:
            errors.append(
                f"line {line_no}: status={status} but local_p22_token={local} dist_token={dist}; expected {expected}"
            )
    if token_lines == 0:
        errors.append("no CLUSTER_DIST_GEN_TOKEN line found")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate distributed ESP32-S3 generation receipts")
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    errors = validate_text(args.receipt.read_text(errors="replace"))
    if errors:
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print(f"PASS {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
