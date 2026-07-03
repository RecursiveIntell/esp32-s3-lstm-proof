#!/usr/bin/env python3

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

BENCH_RE = re.compile(
    r"CLUSTER_BENCH_RESULT\s+"
    r"board=(?P<board>\d+)\s+"
    r"prompt_id=(?P<prompt_id>\d+)\s+"
    r"profile=(?P<profile>\S+)\s+"
    r"generated_chars=(?P<generated_chars>\d+)\s+"
    r"elapsed_ms=(?P<elapsed_ms>\d+)\s+"
    r"chars_per_sec=(?P<chars_per_sec>[0-9]+(?:\.[0-9]+)?)\s+"
    r"checksum=(?P<checksum>0x[0-9A-Fa-f]+|\d+)"
)


@dataclass(frozen=True)
class BenchResult:
    board: int
    prompt_id: int
    profile: str
    generated_chars: int
    elapsed_ms: int
    chars_per_sec: float
    checksum: int


@dataclass(frozen=True)
class AggregateReport:
    schema: str
    result_count: int
    boards: list[int]
    generated_chars_total: int
    elapsed_ms_wall: int
    aggregate_chars_per_sec: float
    per_board_chars_per_sec: dict[str, float]
    claim_boundary: str
    results: list[dict]


def parse_bench_lines(lines: Iterable[str]) -> list[BenchResult]:
    results: list[BenchResult] = []
    for line in lines:
        match = BENCH_RE.search(line)
        if not match:
            continue
        gd = match.groupdict()
        results.append(
            BenchResult(
                board=int(gd["board"]),
                prompt_id=int(gd["prompt_id"]),
                profile=gd["profile"],
                generated_chars=int(gd["generated_chars"]),
                elapsed_ms=int(gd["elapsed_ms"]),
                chars_per_sec=float(gd["chars_per_sec"]),
                checksum=int(gd["checksum"], 0),
            )
        )
    return results


def aggregate_results(results: list[BenchResult]) -> AggregateReport:
    if not results:
        raise ValueError("no CLUSTER_BENCH_RESULT lines found")
    latest_by_board: dict[int, BenchResult] = {}
    for result in results:
        latest_by_board[result.board] = result
    effective_results = [latest_by_board[board] for board in sorted(latest_by_board)]
    wall_ms = max(r.elapsed_ms for r in effective_results)
    total_chars = sum(r.generated_chars for r in effective_results)
    aggregate_cps = (total_chars * 1000.0 / wall_ms) if wall_ms else 0.0
    per_board: dict[str, float] = {}
    for r in effective_results:
        per_board[str(r.board)] = (r.generated_chars * 1000.0 / r.elapsed_ms) if r.elapsed_ms else 0.0
    return AggregateReport(
        schema="ri-esp32s3-cluster-aggregate-bench-v1",
        result_count=len(effective_results),
        boards=[r.board for r in effective_results],
        generated_chars_total=total_chars,
        elapsed_ms_wall=wall_ms,
        aggregate_chars_per_sec=aggregate_cps,
        per_board_chars_per_sec=per_board,
        claim_boundary=(
            "Aggregate chars/s across independent ESP32-S3 local generator workers; "
            "not a single-stream latency or universal BPE-token/s claim. Latest receipt per board is used."
        ),
        results=[asdict(r) for r in effective_results],
    )


def render_markdown(report: AggregateReport) -> str:
    lines = [
        "# ESP32-S3 cluster aggregate benchmark report",
        "",
        f"Schema: `{report.schema}`",
        "",
        f"Boards: {', '.join(map(str, report.boards))}",
        f"Results: {report.result_count}",
        f"Generated chars total: {report.generated_chars_total}",
        f"Wall elapsed ms: {report.elapsed_ms_wall}",
        f"Aggregate chars/s: {report.aggregate_chars_per_sec:.4f}",
        "",
        "## Per-board throughput",
        "",
        "| Board | chars/s |",
        "|---:|---:|",
    ]
    for board, cps in report.per_board_chars_per_sec.items():
        lines.append(f"| {board} | {cps:.4f} |")
    lines += ["", "## Claim boundary", "", report.claim_boundary, ""]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse ESP32-S3 cluster aggregate benchmark serial logs")
    parser.add_argument("log", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    report = aggregate_results(parse_bench_lines(args.log.read_text().splitlines()))
    if args.json_out:
        args.json_out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.write_text(render_markdown(report) + "\n")
    if not args.json_out and not args.md_out:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
