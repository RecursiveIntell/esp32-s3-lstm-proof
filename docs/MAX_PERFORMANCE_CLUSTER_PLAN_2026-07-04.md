# ESP32-S3 Cluster Maximum Performance Plan

> **For Hermes:** Execute task-by-task with strict TDD. Use hardware receipts for every performance claim.

**Goal:** Get the ESP32-S3 cluster to the highest useful TinyStories-class generation performance possible, then decide whether a 33M-class model is worth pursuing.

**Architecture:** Stop spending effort on fine-grained WiFi row-chunk sharding as the primary speed path. Keep it as a correctness/research lane, but move performance work toward coarse partitions: local replicated generation, layer/block sharding with hidden-state transfer, and board-shaped quantized recurrent/SSM models.

**Current evidence baseline:**
- H256 1.6M local aggregate: 118.23 chars/s across 3 boards.
- H512 6.34M local aggregate: 34.78 chars/s across 3 boards.
- H512 row-sharded distributed one-token stable UDP: ~2.8–3.1 s/token when board shards are healthy.
- H512 TCP had one fast receipt but unstable/crash-prone; do not treat it as production proof.
- ESP-NOW in current Arduino/IDF stack is v1 250-byte payload, not usable for 1KB+ gate/result messages without segmentation.

**Hard strategy decision:**
- Performance lane: coarse model partitioning or replicated local generation.
- Research lane: row-sharded one-stream proof, kept correct but not treated as the fastest path.
- 33M target: only after a 12M–16M board-shaped intermediate proves useful.

---

## Phase 0: Make correctness impossible to fake

### Task 0.1: Fix distributed token status semantics

**Objective:** `CLUSTER_DIST_GEN_TOKEN` must print `status=PASS` only when `dist_token == local_p22_token`; otherwise `status=FAIL`.

**Files:**
- Modify: `src/main.cpp`
- Test: `tools/test_cluster_log_semantics.py`

**Steps:**
1. Add RED test with one matching token line and one mismatching token line.
2. Run `python3 tools/test_cluster_log_semantics.py`; expect FAIL because parser/spec detects current mismatch-pass line as invalid.
3. Patch firmware status print logic.
4. Run unit tests and firmware build.

**Gate:**
```bash
python3 tools/test_cluster_log_semantics.py
python3 tools/test_cluster_protocol.py
pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
```

### Task 0.2: Add receipt validator

**Objective:** Any receipt with token mismatch and `status=PASS` fails CI/local validation.

**Files:**
- Create: `tools/validate_dist_receipt.py`
- Test: `tools/test_validate_dist_receipt.py`

**Gate:**
```bash
python3 tools/test_validate_dist_receipt.py
python3 tools/validate_dist_receipt.py receipts/dist_gen_fixed_direct_http_board2_20260704.log
```

---

## Phase 1: Stabilize and benchmark the current best lanes

### Task 1.1: Multi-token H512 local aggregate receipt

**Objective:** Convert the H512 local-generator demo from repeated 16-char receipts into a clean 64/128-char receipt across available boards.

**Files:**
- Modify: `src/main.cpp`
- Modify: `tools/parse_cluster_bench.py`
- Test: `tools/test_parse_cluster_bench.py`

**Gate:**
```bash
pio run -e cluster_coord_ap_local_gen_h512 -e cluster_worker1_ap_local_gen_h512 -e cluster_worker2_ap_local_gen_h512
# live hardware receipt must show generated_chars >= 64 on at least two boards
```

### Task 1.2: 10-run stability harness

**Objective:** Run the selected lane 10 times, parse median/p95 latency, and fail if any token/result is invalid.

**Files:**
- Create: `tools/run_cluster_stability.py`
- Create: `tools/test_run_cluster_stability.py`

**Gate:**
```bash
python3 tools/test_run_cluster_stability.py
python3 tools/run_cluster_stability.py --port /dev/ttyACM0 --runs 10 --mode h512-local
```

---

## Phase 2: Coarse hidden-state transport prototype

### Task 2.1: Add hidden-state packet protocol tests

**Objective:** Define packets for sending a quantized hidden/cell state between boards without sending row chunks.

**Files:**
- Modify: `src/cluster_protocol.h`
- Modify: `tools/test_cluster_protocol.py`

**Protocol:**
- `LSTM_STATE_FORWARD_REQUEST`
- `LSTM_STATE_FORWARD_RESULT`
- payload: `token_id`, `layer_start`, `layer_count`, `hidden_scale`, `cell_scale`, `qx[512]`, `qc[512]`
- result: `layer_end`, `hidden_scale`, `cell_scale`, `qx[512]`, `qc[512]`

**Gate:**
```bash
python3 tools/test_cluster_protocol.py
```

### Task 2.2: Implement single-layer worker state-forward path

**Objective:** Worker receives hidden/cell state, runs its assigned LSTM layer(s), sends updated state back.

**Files:**
- Modify: `src/main.cpp`
- Add envs in `platformio.ini`: `cluster_coord_ap_layer_shard_h512`, `cluster_worker1_ap_layer_shard_h512`, `cluster_worker2_ap_layer_shard_h512`

**Gate:**
```bash
pio run -e cluster_coord_ap_layer_shard_h512 -e cluster_worker1_ap_layer_shard_h512 -e cluster_worker2_ap_layer_shard_h512
```

### Task 2.3: Live one-token layer-shard receipt

**Objective:** Prove one-token H512 generation using layer/block sharding and hidden-state transfer only.

**Gate:**
- Receipt line: `CLUSTER_LAYER_SHARD_TOKEN ... status=PASS`.
- Latency target: under current row-shard UDP median.
- Stretch target: <1000 ms/token.

---

## Phase 3: Kernel speed path

### Task 3.1: Worker int4 recurrent kernels

**Objective:** Convert worker recurrent shards/layers to the same int4 path used by the coordinator/local p22 path.

**Gate:**
- Worker result drift should drop relative to local reference.
- Hardware receipt must show speedup versus worker int8 path.

### Task 3.2: Core pinning and queue isolation

**Objective:** Keep WiFi I/O off the compute hot path with FreeRTOS queues where possible.

**Gate:**
- No Guru Meditation across 10 runs.
- Median latency improves or remains stable with lower p95.

---

## Phase 4: 12M–16M intermediate model

### Task 4.1: Train/export board-shaped recurrent/SSM model

**Objective:** Build an intermediate TinyStories-class model larger than H512 but still plausible.

**Gate:**
- Fits in available partitions/PSRAM with selected quantization.
- Single-board or layer-sharded receipt.
- Output sample visibly better than 6.34M H512.

---

## Phase 5: 33M decision gate

Only proceed to 33M if Phase 4 proves useful.

**33M allowed architecture:**
- 2-bit/ternary recurrent or SSM.
- Coarse layer/block split.
- Hidden-state transfer only.
- No row-chunk WiFi loop as primary path.

**Kill criteria:**
- <1 char/s single stream after coarse split.
- Requires standard transformer KV-cache without a custom memory plan.
- Needs repeated manual board resets to complete a receipt.

---

## Claim boundary

Safe after Phase 0/1:
- Stable receipt-backed H512 local aggregate performance.

Safe after Phase 2:
- H512 coarse layer-sharded generation over physical ESP32-S3 boards.

Not safe until proven:
- 33M is useful.
- 3 boards accelerate a standard transformer.
- TCP path is stable.
