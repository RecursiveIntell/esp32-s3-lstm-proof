# Three-Board ESP32-S3 Cluster Receipt (2026-07-02)

Status: stub

## Purpose

Track the live implementation plan for a 3-board ESP32-S3 tensor-parallel cluster, including claims, hardware inventory, and phase evidence.

## Claim Boundary

- Do not claim TinyStories model behavior (or performance) until a validated TinyStories hardware run is recorded.
- Do not claim the 3-board cluster hardware path is production-ready until phase-level evidence confirms transport, sharding, and generation behavior across all boards.

## Phase 0 Evidence Completed

- Routing design doc: `docs/plans/hybrid-codex-fable-cluster-routing.md`
- Cleanup commit completed: `0200e1b`
- Parser fix commit completed: `61adcf0`
- p22 baseline commit: `2b4da43`
- p22 baseline mean: `39.51703333333333 char-token/s`, `25.30333333333333 ms/token` (3 runs)
- p22 baseline benchmark summary: `benchmarks/p22_i4_wih_whh_simd_h256/p22_i4_wih_whh_simd_h256-summary.json`

## Hardware Inventory (placeholders)

### Boards

- Board IDs:
  - `BOARD_0_ID: <TODO>`
  - `BOARD_1_ID: <TODO>`
  - `BOARD_2_ID: <TODO>`
- Serial ports:
  - `BOARD_0_PORT: <TODO>`
  - `BOARD_1_PORT: <TODO>`
  - `BOARD_2_PORT: <TODO>`
- Transport wiring:
  - `COORDINATOR_TO_WORKER_1_UART: <TODO>`
  - `COORDINATOR_TO_WORKER_2_UART: <TODO>`
  - `WORKER_1_TO_COORD_UART: <TODO>`
  - `WORKER_2_TO_COORD_UART: <TODO>`
  - `COMMON_GND: <TODO>`
- Model shard hashes:
  - `shard_0_hash: <TODO>`
  - `shard_1_hash: <TODO>`
  - `shard_2_hash: <TODO>`

## Phase 1: Multi-board transport proof

- [x] Task 1.1 board-role firmware variants
  - 2026-07-02: `pio run -e cluster_coord` — SUCCESS (`00:00:05.871`), role macros: `CLUSTER_ROLE_COORD=1`, `CLUSTER_BOARD_ID=0`
  - 2026-07-02: `pio run -e cluster_worker1` — SUCCESS (`00:00:08.123`), role macros: `CLUSTER_ROLE_WORKER=1`, `CLUSTER_BOARD_ID=1`
  - 2026-07-02: `pio run -e cluster_worker2` — SUCCESS (`00:00:08.173`), role macros: `CLUSTER_ROLE_WORKER=1`, `CLUSTER_BOARD_ID=2`
- [x] Task 1.2 packet framing and CRC format
  - 2026-07-03: `python3 tools/test_cluster_protocol.py` — SUCCESS (`PASS packet encode/decode/crc`)
  - 2026-07-03: `python3 -m py_compile tools/*.py` — SUCCESS
  - 2026-07-03: `pio run -e cluster_coord` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker1` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2` — SUCCESS
- [ ] Task 1.3 worker echo firmware proof
- [ ] Task 1.4 two-worker barrier sync proof

## Phase 2: Sharded matmul proof with synthetic data

- [ ] Task 2.1 deterministic matmul fixture
- [ ] Task 2.2 worker matmul command
- [ ] Task 2.3 coordinator gather for 3 shards
- [ ] Task 2.4 int4 sharded matmul fixture

## Phase 3: Shard the existing H256 LSTM model

- [ ] Task 3.1 model shard exporter
- [ ] Task 3.2 per-board shard flash flow
- [ ] Task 3.3 one-token H256 cluster inference check
- [ ] Task 3.4 full H256 utility suite on cluster

## Phase 4: Transformer micro-model before TinyStories

- [ ] Task 4.x pending

## Phase 5: TinyStories reduced model

- [ ] Task 5.x pending

## Phase 6: TinyStories-33M attempt

- [ ] Task 6.x pending

## Phase 7: Sensor-grounded demo integration

- [ ] Task 7.x pending
