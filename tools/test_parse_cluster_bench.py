#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "parse_cluster_bench.py"


def load_module():
    spec = importlib.util.spec_from_file_location("parse_cluster_bench", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_and_aggregate_two_worker_results() -> None:
    mod = load_module()
    lines = [
        'noise',
        'CLUSTER_BENCH_RESULT board=1 prompt_id=3 profile=h256_p22 generated_chars=128 elapsed_ms=3200 chars_per_sec=40.0000 checksum=0xA5A55A5A',
        'CLUSTER_BENCH_RESULT board=2 prompt_id=4 profile=h256_p22 generated_chars=96 elapsed_ms=2400 chars_per_sec=40.0000 checksum=17',
    ]
    results = mod.parse_bench_lines(lines)
    assert len(results) == 2
    report = mod.aggregate_results(results)
    assert report.schema == "ri-esp32s3-cluster-aggregate-bench-v1"
    assert report.boards == [1, 2]
    assert report.generated_chars_total == 224
    assert report.elapsed_ms_wall == 3200
    assert abs(report.aggregate_chars_per_sec - 70.0) < 1e-6
    assert abs(report.per_board_chars_per_sec["1"] - 40.0) < 1e-6
    assert abs(report.per_board_chars_per_sec["2"] - 40.0) < 1e-6
    assert "not a single-stream" in report.claim_boundary


def test_aggregate_uses_latest_result_per_board() -> None:
    mod = load_module()
    lines = [
        'CLUSTER_BENCH_RESULT board=1 prompt_id=1 profile=h256_p22 generated_chars=64 elapsed_ms=3200 chars_per_sec=20.0000 checksum=0x1',
        'CLUSTER_BENCH_RESULT board=2 prompt_id=2 profile=h256_p22 generated_chars=64 elapsed_ms=3200 chars_per_sec=20.0000 checksum=0x2',
        'CLUSTER_BENCH_RESULT board=1 prompt_id=3 profile=h256_p22 generated_chars=128 elapsed_ms=3200 chars_per_sec=40.0000 checksum=0x3',
    ]
    report = mod.aggregate_results(mod.parse_bench_lines(lines))
    assert report.result_count == 2
    assert report.generated_chars_total == 192
    assert abs(report.aggregate_chars_per_sec - 60.0) < 1e-6
    assert abs(report.per_board_chars_per_sec["1"] - 40.0) < 1e-6
    assert abs(report.per_board_chars_per_sec["2"] - 20.0) < 1e-6


def test_rejects_empty_logs() -> None:
    mod = load_module()
    try:
        mod.aggregate_results([])
    except ValueError as exc:
        assert "no CLUSTER_BENCH_RESULT" in str(exc)
    else:
        raise AssertionError("expected empty aggregate rejection")


def main() -> None:
    test_parse_and_aggregate_two_worker_results()
    test_aggregate_uses_latest_result_per_board()
    test_rejects_empty_logs()
    print("PASS cluster bench parser")


if __name__ == "__main__":
    main()
