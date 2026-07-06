# ESP32-S3 Cluster Maximum Performance Execution Plan

> **For Hermes:** Execute task-by-task with strict TDD where source changes are involved. Every performance claim requires a live hardware receipt. Use Codex for non-trivial firmware changes; verify with local tests, PlatformIO builds, and serial receipts.

**Goal:** Extract the maximum practical performance from the 3-board ESP32-S3 TinyStories/H512 cluster without losing receipt-backed correctness.

**Architecture:** Keep three lanes separate: (A) restore/guard the stable row-sharded H512 lane; (B) optimize H512 local replicated/aggregate generation for public speed/utility receipts; (C) build a coarse layer/block-sharded hidden-state lane to eliminate row-chunk WiFi overhead. Do not spend more time on Arduino `WiFiClient` TCP; if large-payload TCP returns, use raw lwIP sockets only.

**Repo:** `/home/sikmindz/projects/esp32-s3-lstm-proof`

**Current evidence baseline:**
- H256 1.6M aggregate local generation: 118.23 chars/s across 3 boards.
- H512 6.34M aggregate local generation: 34.78 chars/s across 3 boards.
- H512 row-sharded one-stream stable UDP: ~2.8–3.1s/token when shards are healthy.
- Row-sharded UDP 256-row chunks are correct but strategically too slow.
- Arduino `WiFiClient` TCP is unstable; prior fast-ish receipts are not release-safe.
- ESP-NOW v1 in the installed stack is payload-hostile at 250 bytes.

**Immediate blocker as of 2026-07-05:**
- Current HEAD includes layer-shard scaffold, but coordinator repeatedly watchdog-resets during H512 model initialization (`TG1WDT_SYS_RST`) before `MODEL_READY`.
- This must be fixed before further performance work.

---

## Phase 0 — Recovery and hard gates

### Task 0.1: Add model-init watchdog/yield regression gate

**Objective:** Prevent heavy H512 model init loops from starving the task watchdog.

**Files:**
- Create/modify: `tools/test_model_init_watchdog_static.py`
- Modify: `src/main.cpp`

**Steps:**
1. Write RED static test requiring long model-init loops (`clone_payloads_to_psram`, `convert_wih_to_int4`) to feed/yield periodically.
2. Run `python3 tools/test_model_init_watchdog_static.py`; expect FAIL on current source.
3. Patch firmware with a small `model_init_pump_watchdog()` helper using `yield()`, `delay(0)`, and/or `esp_task_wdt_reset()` where safe.
4. Run focused tests.
5. Build coordinator/worker shard envs.
6. Flash coordinator and capture live boot receipt with `MODEL_READY` and no WDT reset.

**Gate:**
```bash
python3 tools/test_model_init_watchdog_static.py
python3 tools/test_cluster_protocol.py
pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
```
Live gate:
- `CLUSTER_MODEL_READY` or equivalent model-ready line appears.
- No `TG1WDT_SYS_RST` loop for 90s after reset.

### Task 0.2: Restore stable distributed H512 receipt

**Objective:** Re-establish the known-good row-sharded lane before optimizing anything.

**Steps:**
1. Verify board identity by serial/boot output, not assumptions.
2. Verify board1 and board2 PONG with distinct board IDs and `model_ready=1`.
3. Run one distributed H512 token.
4. Validate receipt with `tools/validate_dist_receipt.py`.

**Gate:**
- Receipt contains `CLUSTER_DIST_GEN_TOKEN ... status=PASS`.
- `dist_token == local_p22_token`.
- elapsed_ms recorded.

---

## Phase 1 — Make local H512 aggregate the public performance lane

### Task 1.1: 64/128-char H512 local receipt mode

**Objective:** Extend local H512 benchmark receipt beyond 16 chars so the public number is harder to dismiss.

**Files:**
- Modify: `src/main.cpp`
- Modify/test: `tools/parse_cluster_bench.py`, `tools/test_parse_cluster_bench.py`

**Gate:**
- 2+ boards produce `generated_chars >= 64`.
- Parse output reports per-board and aggregate chars/s.
- Claim boundary states this is aggregate independent generation, not one-stream acceleration.

### Task 1.2: Stability harness

**Objective:** Replace one-off receipts with repeatable median/p95 data.

**Files:**
- Create: `tools/run_cluster_stability.py`
- Create: `tools/test_run_cluster_stability.py`

**Gate:**
```bash
python3 tools/test_run_cluster_stability.py
python3 tools/run_cluster_stability.py --port /dev/ttyACM0 --mode h512-local --runs 10
```
- 10/10 valid runs.
- JSON + markdown receipt with median/p95.

---

## Phase 2 — Stop row-chunk WiFi from being the speed path

### Task 2.1: Coarse hidden-state/layer-shard smoke receipt

**Objective:** Prove physical boards can exchange hidden state only, not row chunks.

**Existing state:** Hidden-state protocol/scaffold exists in current HEAD.

**Next work:**
1. Fix boot stability from Phase 0 first.
2. Flash layer-shard smoke env.
3. Capture `CLUSTER_LAYER_SHARD_SMOKE` style receipt.
4. Validate transport latency and payload integrity.

**Gate:**
- Live receipt proves hidden-state payload reaches worker and returns.
- No model-token claim yet unless token equality is implemented.

### Task 2.2: Real layer/block-sharded token

**Objective:** Run H512 one token with layer/block ownership and hidden-state transfer only.

**Kill criteria:**
- If one-token latency remains > row-sharded UDP after real layer/block sharding, stop.
- If repeated manual board resets are required, stop and fix reliability first.

**Stretch target:** <1000 ms/token.

---

## Phase 3 — Kernel/compute path

### Task 3.1: Worker int4 recurrent compute

**Objective:** Workers must not stay on slower int8 row-shard math if row-shard lane remains useful.

**Gate:**
- Per-chunk compute time drops materially.
- Token PASS remains true.

### Task 3.2: FreeRTOS queue/core isolation

**Objective:** Reduce p95 stalls and WDT risk by isolating WiFi and compute.

**Gate:**
- 10-run receipt has no Guru/WDT.
- p95 improves or remains stable with reduced reset risk.

---

## Phase 4 — Bigger model only after topology proof

### Task 4.1: 12M–16M board-shaped recurrent/SSM model

**Objective:** Intermediate model larger than H512 but designed for board topology.

**Gate:**
- Fits memory/partitions.
- Receipt-backed generation.
- Output utility visibly better than H512.

### Task 4.2: 33M decision

Proceed only if Phase 4 passes.

Allowed architecture:
- coarse layer/block split;
- hidden-state transfer only;
- 2-bit/ternary/recurrent/SSM-friendly design.

Hard no:
- standard transformer KV-cache on ESP32-S3 as primary path;
- row-chunk WiFi as primary 33M speed path;
- claims without receipts.

---

## Verification gauntlet

Always run before final claims:
```bash
python3 -m py_compile tools/*.py
python3 tools/test_cluster_protocol.py
python3 tools/test_parse_cluster_bench.py
python3 tools/test_cluster_mode_tools.py
python3 tools/test_validate_dist_receipt.py
python3 tools/test_cluster_log_semantics.py
python3 tools/test_layer_shard_static.py
pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
```

For each live lane, produce:
- raw `.log` receipt;
- parsed `.json` when applicable;
- short `.md` summary;
- explicit claim boundary.

## Claim boundary

Safe now:
- H512 row-sharded proof exists but is slow.
- H512 aggregate local generation is useful and fast enough for public demo.

Not safe yet:
- row-sharded one-stream is strategically useful;
- TCP is stable;
- 33M will be useful;
- three boards accelerate a standard transformer.
