# ESP32-S3 Cluster Maximum Performance Plan v2

> Execute task-by-task with strict TDD. Every performance claim requires a live hardware receipt.

Goal: Extract maximum practical performance from the 3-board ESP32-S3 H512 cluster.

Repo: /home/sikmindz/projects/esp32-s3-lstm-proof

## Completed (Phase 0)

- [x] TG1WDT bootloop fix (commit ea1d9f7)
- [x] Bounded RILM parser with per-tensor logging
- [x] Chunked payload copy with watchdog pump
- [x] Stable distributed H512 token receipt: PASS, 3143ms, dist_char=t
- [x] Receipt validator, log semantics, layer-shard static tests
- [x] Hidden-state forward protocol packets (commit f16edad)
- [x] Layer-shard hidden-state transport scaffold (commit 41c1e28)
- [x] Layer-shard smoke trigger (commit 7348ced)

## Current evidence baseline

- H256 1.6M aggregate local: 118.23 chars/s (3 boards)
- H512 6.34M aggregate local: 34.78 chars/s (3 boards)
- H512 row-sharded one-stream UDP: 3143ms/token, token PASS
- H512 single-board local: 11.62 chars/s (~86ms/token)

## Phase 1: H512 local aggregate as public performance lane

### Task 1.1: Extend H512 local-gen to 64/128 chars

Objective: Currently H512 local_gen produces 16-char receipts. Extend to 64+ chars so the public number is harder to dismiss.

Files: src/main.cpp (TOKENS_PER_SEED or equivalent for local_gen_h512), tools/parse_cluster_bench.py

Gate: 2+ boards produce generated_chars >= 64, parse output reports per-board and aggregate chars/s.

### Task 1.2: Stability harness

Objective: 10-run median/p95 data instead of one-off receipts.

Files: tools/run_cluster_stability.py, tools/test_run_cluster_stability.py

Gate: 10/10 valid runs, JSON+markdown receipt with median/p95.

## Phase 2: Layer-shard hidden-state transport

### Task 2.1: Smoke receipt

Objective: Prove physical boards can exchange hidden state only, not row chunks. Use existing scaffold from commit 7348ced.

Gate: Live receipt proves hidden-state payload reaches worker and returns with valid payload integrity.

### Task 2.2: Real layer-sharded token

Objective: H512 one token with layer/block ownership and hidden-state transfer only.

Kill criteria: If latency > row-sharded UDP after real layer sharding, stop. If repeated manual resets required, stop.

Stretch target: <1000ms/token.

## Phase 3: Kernel/compute path

### Task 3.1: Worker int4 recurrent compute

Gate: Per-chunk compute time drops materially, token PASS remains true.

### Task 3.2: FreeRTOS queue/core isolation

Gate: 10-run receipt has no Guru/WDT, p95 improves.

## Phase 4: Bigger model only after topology proof

### Task 4.1: 12M-16M board-shaped recurrent/SSM model

Gate: Fits memory/partitions, receipt-backed generation, output visibly better than H512.

### Task 4.2: 33M decision

Proceed only if Phase 4 passes. Allowed: coarse layer/block split, hidden-state transfer only, 2-bit/ternary/recurrent/SSM. Hard no: standard transformer KV-cache, row-chunk WiFi as primary, claims without receipts.

## Claim boundary

Safe now: H512 row-sharded proof exists (slow), H512 aggregate local is useful for demo.
Not safe yet: row-sharded one-stream is strategically useful, TCP is stable, 33M will be useful, three boards accelerate a standard transformer.