#!/usr/bin/env python3

import re

TOKEN_RE = re.compile(
    r"CLUSTER_DIST_GEN_TOKEN\s+.*?"
    r"local_p22_token=(?P<local>\d+)\s+.*?"
    r"dist_token=(?P<dist>\d+)\s+.*?"
    r"status=(?P<status>PASS|FAIL)"
)


def parse_token_line(line: str) -> tuple[int, int, str]:
    match = TOKEN_RE.search(line)
    if not match:
        raise ValueError("not a distributed token receipt line")
    return int(match.group("local")), int(match.group("dist")), match.group("status")


def validate_token_status(line: str) -> None:
    local, dist, status = parse_token_line(line)
    expected = "PASS" if local == dist else "FAIL"
    if status != expected:
        raise AssertionError(
            f"token receipt status={status} but local_p22_token={local} dist_token={dist}; expected {expected}"
        )


def test_matching_token_passes() -> None:
    validate_token_status(
        'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
        'dist_token=19 dist_char=t logit=10.186430 elapsed_ms=3111 status=PASS '
        'note=worker_int8_recurrent_vs_local_int4_reference'
    )


def test_mismatching_token_must_fail() -> None:
    validate_token_status(
        'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
        'dist_token=26 dist_char=  logit=2.808062 elapsed_ms=6675 status=FAIL '
        'note=worker_int8_recurrent_vs_local_int4_reference'
    )


def test_known_bad_mismatch_pass_receipt_is_rejected() -> None:
    try:
        validate_token_status(
            'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
            'dist_token=26 dist_char=  logit=2.808062 elapsed_ms=6675 status=PASS '
            'note=worker_int8_recurrent_vs_local_int4_reference'
        )
    except AssertionError:
        return
    raise AssertionError("mismatching token with status=PASS was accepted")


def main() -> None:
    test_matching_token_passes()
    test_mismatching_token_must_fail()
    test_known_bad_mismatch_pass_receipt_is_rejected()
    print("PASS cluster log semantics")


if __name__ == "__main__":
    main()
