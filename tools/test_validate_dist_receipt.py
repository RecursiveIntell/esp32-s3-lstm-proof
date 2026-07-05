#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "validate_dist_receipt.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_dist_receipt", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matching_pass_line_validates() -> None:
    validator = load_validator()
    errors = validator.validate_text(
        'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
        'dist_token=19 dist_char=t logit=10.186430 elapsed_ms=3111 status=PASS '
        'note=worker_int8_recurrent_vs_local_int4_reference\n'
    )
    assert errors == []


def test_mismatching_fail_line_validates() -> None:
    validator = load_validator()
    errors = validator.validate_text(
        'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
        'dist_token=26 dist_char=  logit=2.808062 elapsed_ms=6675 status=FAIL '
        'note=worker_int8_recurrent_vs_local_int4_reference\n'
    )
    assert errors == []


def test_mismatching_pass_line_is_rejected() -> None:
    validator = load_validator()
    errors = validator.validate_text(
        'CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t '
        'dist_token=26 dist_char=  logit=2.808062 elapsed_ms=6675 status=PASS '
        'note=worker_int8_recurrent_vs_local_int4_reference\n'
    )
    assert errors
    assert "expected FAIL" in errors[0]


def test_missing_token_line_is_rejected() -> None:
    validator = load_validator()
    errors = validator.validate_text("CLUSTER_WIFI_PONG src_board=1 model_ready=1\n")
    assert errors == ["no CLUSTER_DIST_GEN_TOKEN line found"]


def main() -> None:
    test_matching_pass_line_validates()
    test_mismatching_fail_line_validates()
    test_mismatching_pass_line_is_rejected()
    test_missing_token_line_is_rejected()
    print("PASS dist receipt validator tests")


if __name__ == "__main__":
    main()
