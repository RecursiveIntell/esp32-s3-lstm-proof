# 3-Board ESP32-S3 TinyStories Cluster Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after the user explicitly says to execute it. Keep proof receipts after every hardware phase.

**Goal:** Build a 3x ESP32-S3 tensor-parallel language-model cluster that runs a quantized TinyStories-class BPE model locally, with sensor-grounded prompts and hardware receipts.

**Architecture:** One coordinator board owns the serial/host interface, prompt/token loop, tokenization, sampling, and receipt emission. Three ESP32-S3 boards each own one shard of the model weights and execute tensor-parallel slices of the forward pass. Start with a small cluster-proof model and wire protocol, then graduate to a TinyStories-33M int4/int8 hybrid model only after transport, synchronization, sharding, and correctness are proven.

**Tech Stack:** PlatformIO Arduino ESP32-S3 firmware, ESP-NN / custom Xtensa SIMD kernels, packed int4 weights, Python model-export tooling, Python host harness, UART/SPI transport first, optional WiFi later, receipt-based verification.

---

## Current state checked

Repo path:
- `/home/sikmindz/projects/esp32-s3-lstm-proof`

Checked on:
- `2026-07-02`

Evidence commands already run in this session:
- `git status --short && git log --oneline -3`
- `find . -maxdepth 3 -type f | sort | head -120`
- `search_files("*.py", target="files")`
- `search_files("*.cpp", target="files")`
- `search_files("*.md", target="files")`
- `read_file(P22_INT4_SIMD_KERNEL_RECEIPT_2026-07-02.md)`
- `read_file(src/main.cpp)`
- `read_file(tools/run_bench.py)`
- `read_file(platformio.ini)`

Verified substrate:
- Current single-board firmware: `p22_i4_wih_whh_simd_h256`
- Current best measured throughput: `39.5062 char-token/s`, `25.31 ms/token`
- BPE-equivalent at 4 chars/BPE-token: `9.876 BPE tok/s`
- Hardware board path: `/dev/ttyACM0`
- Custom int4 SIMD recurrent kernel exists and is committed: `54a670d Add ESP32-S3 int4 SIMD recurrent kernel`
- Kernel files:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/lib/esp-nn/src/common/dot_i4_i8_esp32s3.S`
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/lib/esp-nn/src/common/dot_i4_i8_wrapper.c`
- Main firmware:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Benchmark harness:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_bench.py`
- Current receipt:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/P22_INT4_SIMD_KERNEL_RECEIPT_2026-07-02.md`

Important current receipt facts:
- Two hardware runs produced identical speed: `39.5062 char-token/s`
- `lstm_wih`: about `10.63 ms/token`
- `lstm_whh`: about `10.40 ms/token`
- `core1_wait`: about `24.21 ms/token`
- Domain outputs remain valid; one output changed from older p17 exact output but stayed semantically valid.

Working tree notes:
- `SRAM_COPY_ELIMINATION_RECEIPT_2026-07-02.md`, `benchmarks/current/`, `src/main.cpp.p16_backup`, and `src/main.cpp.p18_backup` are untracked at the time this plan was written.
- Do not start cluster implementation until deciding whether to commit, archive, or delete those untracked artifacts.

---

## Strategy verdict

Highest-ROI cluster path:

1. Do not jump straight to TinyStories-33M.
2. First build a 3-board cluster transport and tensor-shard proof using the existing H256 LSTM substrate.
3. Then build a synthetic transformer micro-model with BPE tokenizer and sharded output projection.
4. Then export and run a reduced TinyStories model.
5. Only after those pass, attempt TinyStories-33M.

Reason:
- The failure risks are transport synchronization, shard correctness, model export, and output projection bandwidth.
- The existing p22 single-board firmware is good enough to serve as a deterministic correctness substrate.
- A cluster that cannot reproduce a known small model receipt will not successfully run TinyStories-33M.

---

## Non-negotiable claim boundaries

Safe claims after Phase 1:
- “3 ESP32-S3 boards can coordinate tensor-sharded inference over a local transport.”
- “Transport latency and synchronization were measured on hardware.”

Safe claims after Phase 2:
- “A sharded model path reproduces the single-board reference within the configured tolerance.”

Safe claims after Phase 3:
- “A transformer-like BPE model path runs across 3 boards.”

Safe claims after Phase 4:
- “A TinyStories-class quantized model runs across 3 ESP32-S3 boards.”

Unsafe until proven:
- “TinyStories-33M runs at 10+ BPE tok/s.”
- “$15 local language AI for everyone.”
- “Scales linearly.”
- “Better than Raspberry Pi / phone / cloud.”
- “General chatbot.”

---

## Target hardware topology

Minimum 3-board cluster:

- Board 0: coordinator + shard 0
- Board 1: worker + shard 1
- Board 2: worker + shard 2

Recommended physical wiring for v1:

- Common ground between all boards
- Dedicated UART coordinator -> worker 1
- Dedicated UART coordinator -> worker 2
- Optional worker -> coordinator return UARTs if full duplex pins are available
- USB serial from coordinator to host PC

Preferred v1 transport:
- UART first, not WiFi.

Why UART first:
- Deterministic enough for phase-1 receipts
- Easier to debug with logic analyzer / serial logs
- No WiFi jitter hidden inside benchmark numbers
- No pairing/provisioning complexity

Future transport after correctness:
- SPI coordinator-master, worker-slave if UART bandwidth becomes limiting.
- WiFi only after wired transport has receipts.

---

## Model-parallel design

### For the H256 LSTM cluster proof

Shard by hidden/output rows:

- Each board owns a contiguous slice of gate rows.
- For H256, each LSTM layer has 4 gates x 256 rows = 1024 gate rows.
- 3 boards split rows approximately:
  - Board 0: rows 0..341
  - Board 1: rows 342..682
  - Board 2: rows 683..1023

Per token, per layer:

1. Coordinator broadcasts current quantized input vector `qx` and hidden vector `qh`.
2. Each board computes its assigned gate rows:
   - local `wih` shard dot `qx`
   - local `whh` shard dot `qh`
   - local bias add
3. Each worker returns partial gate values.
4. Coordinator assembles full gates, applies activation, updates hidden/cell state.
5. Repeat for next layer.
6. Coordinator runs final output projection or later shards it too.

This is not the final TinyStories architecture, but it proves:
- transport framing
- per-token synchronization
- row-sharded matmul
- partial result assembly
- deterministic output matching

### For TinyStories transformer cluster

Shard by rows for MLP/output projection and by heads for attention:

- Embedding table:
  - Initially replicated if reduced vocab fits.
  - For 32k vocab TinyStories, shard vocab rows across boards.

- Attention:
  - Split heads across boards.
  - Each board computes Q/K/V and attention output for its local heads.
  - Coordinator concatenates or sums projected head outputs depending on weight layout.

- MLP:
  - Row-shard up-projection.
  - Worker returns partial activations or local down-projection partials.
  - Prefer layout that minimizes transferred vector size.

- Output projection:
  - Shard vocabulary logits across boards.
  - Each board computes logits for its vocab slice.
  - Workers return top-k candidates, not all logits, for v1.
  - Coordinator merges top-k and samples/argmaxes.

Important optimization:
- Do not gather 32k logits every token unless needed.
- Return per-shard top-k, e.g. k=8 or k=16.
- Coordinator merges 3*k candidates.

---

## On-paper performance target

Current single-board p22:
- `39.506 char-token/s`
- `9.876 BPE tok/s equivalent` for the small H256 char model

Cluster target for TinyStories-class model:
- Minimum success: any coherent BPE model across 3 boards with receipts
- Good v1 target: `>= 10 BPE tok/s` on a reduced TinyStories-class model
- Stretch target: TinyStories-33M int4 hybrid at `>= 10 BPE tok/s`

Expected scaling reality:
- Not 3x. Autoregressive generation is serial by token.
- Useful gain comes from 3 independent PSRAM buses and 3 CPUs working on tensor shards.
- Communication overhead and coordinator assembly eat part of the gain.

Performance gate format:
- Always report both:
  - raw BPE tokens/s
  - ms/token
- Do not convert char-token/s to BPE-token/s once the model uses real BPE.
- For TinyStories, report real tokenizer token count.

---

## Plan phases

## Phase 0: Clean baseline and freeze p22

### Task 0.1: Clean or commit untracked artifacts

**Objective:** Start cluster work from a clean, auditable baseline.

**Files:**
- Inspect: `/home/sikmindz/projects/esp32-s3-lstm-proof/SRAM_COPY_ELIMINATION_RECEIPT_2026-07-02.md`
- Inspect: `/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/current/`
- Inspect: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp.p16_backup`
- Inspect: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp.p18_backup`

**Steps:**
1. Run:
   - `cd /home/sikmindz/projects/esp32-s3-lstm-proof && git status --short`
2. Decide for each untracked artifact:
   - commit receipt if useful
   - move backup files to `archive/firmware-variants/`
   - delete if redundant
3. Run:
   - `git status --short`
4. Expected:
   - only intentional files remain untracked or working tree is clean.
5. Commit:
   - `git add ... && git commit -m "chore: archive p16-p18 firmware receipts"`

### Task 0.2: Capture p22 3-run baseline with current harness

**Objective:** Create the baseline performance artifact that all cluster work must beat or explain.

**Files:**
- Use: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_bench.py`
- Output: `/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p22_i4_wih_whh_simd_h256/`

**Steps:**
1. Run:
   - `cd /home/sikmindz/projects/esp32-s3-lstm-proof`
   - `python3 tools/run_bench.py --port /dev/ttyACM0 --variant p22_i4_wih_whh_simd_h256 --repeat 3 --timeout 180 --out-dir benchmarks/p22_i4_wih_whh_simd_h256`
2. Expected:
   - 3 parsed receipt JSON files
   - 1 summary JSON file
   - no serial parse failures
3. Verify:
   - `python3 -m json.tool benchmarks/p22_i4_wih_whh_simd_h256/p22_i4_wih_whh_simd_h256-summary.json >/dev/null`
4. Commit:
   - `git add benchmarks/p22_i4_wih_whh_simd_h256 && git commit -m "bench: capture p22 int4 simd baseline"`

### Task 0.3: Add a cluster design receipt stub

**Objective:** Create a living receipt file that records phase gates and hardware results.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`

**Content required:**
- Hardware list
- Board IDs and serial ports
- Transport selected
- Firmware variants flashed on each board
- Per-phase result table
- Claim boundary section

**Verification:**
- File exists and has empty sections for Phases 1-5.

**Commit:**
- `git add THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md && git commit -m "docs: add three-board cluster receipt"`

---

## Phase 1: Multi-board transport proof

### Task 1.1: Add board role configuration

**Objective:** Let the same firmware image run as coordinator or worker based on compile flag.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/platformio.ini`
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

**Implementation:**
Add build environments:

```ini
[env:cluster_coord]
extends = env:esp32s3_lstm
build_flags =
  ${env:esp32s3_lstm.build_flags}
  -D CLUSTER_ROLE_COORD=1
  -D CLUSTER_BOARD_ID=0

[env:cluster_worker1]
extends = env:esp32s3_lstm
build_flags =
  ${env:esp32s3_lstm.build_flags}
  -D CLUSTER_ROLE_WORKER=1
  -D CLUSTER_BOARD_ID=1

[env:cluster_worker2]
extends = env:esp32s3_lstm
build_flags =
  ${env:esp32s3_lstm.build_flags}
  -D CLUSTER_ROLE_WORKER=1
  -D CLUSTER_BOARD_ID=2
```

In `src/main.cpp`, define defaults:

```cpp
#ifndef CLUSTER_BOARD_ID
#define CLUSTER_BOARD_ID 0
#endif
#ifndef CLUSTER_ROLE_COORD
#define CLUSTER_ROLE_COORD 0
#endif
#ifndef CLUSTER_ROLE_WORKER
#define CLUSTER_ROLE_WORKER 0
#endif
```

**Verification:**
- `pio run -e cluster_coord`
- `pio run -e cluster_worker1`
- `pio run -e cluster_worker2`
- Expected: all build successfully.

**Commit:**
- `git add platformio.ini src/main.cpp && git commit -m "cluster: add coordinator and worker build roles"`

### Task 1.2: Add binary packet framing

**Objective:** Define a deterministic, checksum-protected packet protocol.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/cluster_protocol.h`
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

**Packet format v1:**

```cpp
struct ClusterPacketHeader {
  uint32_t magic;       // 'RIEC' = 0x43454952
  uint8_t version;      // 1
  uint8_t msg_type;
  uint8_t src_board;
  uint8_t dst_board;
  uint32_t seq;
  uint16_t payload_len;
  uint16_t crc16;
};
```

Message types:
- `PING`
- `PONG`
- `BROADCAST_VECTOR`
- `MATMUL_REQUEST`
- `MATMUL_RESULT`
- `ERROR`

**Verification:**
- Add host-side Python roundtrip test for encode/decode.
- Run:
  - `python3 tools/test_cluster_protocol.py`
- Expected:
  - `PASS packet encode/decode/crc`

**Commit:**
- `git add src/cluster_protocol.h tools/test_cluster_protocol.py && git commit -m "cluster: add packet framing protocol"`

### Task 1.3: Add worker echo firmware mode

**Objective:** Prove coordinator can send packet to worker and receive response.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/cluster_ping.py`

**Steps:**
1. Worker boot prints:
   - `CLUSTER_WORKER_READY board_id=N`
2. Coordinator sends `PING seq=N`.
3. Worker replies `PONG seq=N`.
4. Host script logs latency.

**Verification:**
- Flash worker 1 and coordinator.
- Run:
  - `python3 tools/cluster_ping.py --coord /dev/ttyACM0 --count 100`
- Expected:
  - 100/100 PONG
  - p50 latency and p99 latency printed
  - no CRC failures

**Commit:**
- `git add src/main.cpp tools/cluster_ping.py && git commit -m "cluster: prove coordinator-worker ping"`

### Task 1.4: Extend ping to two workers

**Objective:** Prove the coordinator can synchronize both workers in one token-step barrier.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/cluster_ping.py`
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

**Verification:**
- Run 100 barrier rounds.
- Expected:
  - worker 1 PONG count = 100
  - worker 2 PONG count = 100
  - coordinator reports barrier p50/p99 latency

**Commit:**
- `git add src/main.cpp tools/cluster_ping.py && git commit -m "cluster: prove two-worker synchronization"`

---

## Phase 2: Sharded matmul proof with synthetic data

### Task 2.1: Add a host-generated synthetic matrix fixture

**Objective:** Create deterministic data for row-sharded dot product correctness.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/make_cluster_fixture.py`
- Output: `/home/sikmindz/projects/esp32-s3-lstm-proof/fixtures/cluster_matmul_fixture.json`

**Fixture:**
- input vector length: 256 int8
- matrix rows: 1024
- matrix cols: 256
- dtype: int8 first
- shard rows: 0..341, 342..682, 683..1023
- expected int32 dot outputs for all rows

**Verification:**
- `python3 tools/make_cluster_fixture.py`
- `python3 -m json.tool fixtures/cluster_matmul_fixture.json >/dev/null`

**Commit:**
- `git add tools/make_cluster_fixture.py fixtures/cluster_matmul_fixture.json && git commit -m "cluster: add deterministic matmul fixture"`

### Task 2.2: Add worker matmul command

**Objective:** Worker receives a vector and computes dot products for its shard.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/cluster_protocol.h`

**Message:**
- `MATMUL_REQUEST`: seq, rows_start, rows_count, cols, input vector
- `MATMUL_RESULT`: seq, rows_start, rows_count, int32 outputs

**Verification:**
- Single worker test against fixture rows.
- Expected:
  - exact int32 match for every returned row.

**Commit:**
- `git add src/main.cpp src/cluster_protocol.h && git commit -m "cluster: add worker matmul command"`

### Task 2.3: Add coordinator gather for 3 shards

**Objective:** Coordinator gathers all row shards and reconstructs full output.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_matmul.py`

**Verification:**
- Run 100 fixture matmul rounds.
- Expected:
  - 100/100 full-output exact matches
  - p50/p99 matmul round time printed

**Commit:**
- `git add src/main.cpp tools/run_cluster_matmul.py && git commit -m "cluster: gather three matmul shards"`

### Task 2.4: Repeat fixture with packed int4 matrix

**Objective:** Prove the p22 custom int4 SIMD kernel works in cluster worker mode.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/make_cluster_fixture.py`
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

**Verification:**
- Run int4 fixture with dequantization tolerance.
- Expected:
  - max absolute accumulator mismatch = 0 if comparing integer unpack path
  - or model-level tolerance recorded if comparing dequantized float outputs

**Commit:**
- `git add tools/make_cluster_fixture.py src/main.cpp fixtures/cluster_matmul_fixture_i4.json && git commit -m "cluster: prove int4 sharded matmul"`

---

## Phase 3: Shard the existing H256 LSTM model

### Task 3.1: Add model shard exporter

**Objective:** Export three board-specific RILM model files from the current H256 model.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_lstm_cluster_shards.py`
- Output:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/models/cluster_h256_board0.rilm`
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/models/cluster_h256_board1.rilm`
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/models/cluster_h256_board2.rilm`

**Rules:**
- Shard `lstm.weight_ih_l*` rows.
- Shard `lstm.weight_hh_l*` rows.
- Shard matching bias rows.
- Keep tokenizer/vocab metadata replicated.
- Keep final projection on coordinator for the first proof.

**Verification:**
- Export script writes a JSON manifest:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/models/cluster_h256_manifest.json`
- Manifest includes shard row ranges and SHA256 per file.

**Commit:**
- `git add tools/export_lstm_cluster_shards.py models/cluster_h256_* && git commit -m "cluster: export h256 model shards"`

### Task 3.2: Add per-board model partition upload/flash flow

**Objective:** Flash different model shard partitions to different boards.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/extra_script.py`
- Modify or create partition docs as needed.
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/flash_cluster.py`

**Verification:**
- `python3 tools/flash_cluster.py --board0 /dev/ttyACM0 --board1 /dev/ttyACM1 --board2 /dev/ttyACM2 --dry-run`
- Expected:
  - prints exact PlatformIO env and model shard for each board.

**Commit:**
- `git add extra_script.py tools/flash_cluster.py && git commit -m "cluster: add shard flashing flow"`

### Task 3.3: Coordinator full-token H256 cluster inference

**Objective:** Run one H256 token step across 3 boards and match single-board p22 output.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_h256.py`

**Verification:**
- Prompt: `hot room. action is `
- Run one token at a time with both:
  - single-board reference path
  - cluster path
- Expected first gate/layer output tolerance:
  - exact if using same int4 quantized path and same operation order
  - otherwise record max abs/relative error
- Expected stopped output:
  - `check airflow.`

**Commit:**
- `git add src/main.cpp tools/run_cluster_h256.py && git commit -m "cluster: run h256 inference across three boards"`

### Task 3.4: Full utility suite on cluster H256

**Objective:** Match the p22 utility behavior with cluster execution.

**Files:**
- Use: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_h256.py`
- Update: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`

**Verification prompts:**
- `hot room. action is `
- `missing sensor. action is `
- `stale data. action is `
- `high heat and humidity. action is `
- `humid room. action is `
- `normal room. action is `
- `safe action is `
- `local first means `

**Expected:**
- Outputs are either byte-identical to p22 or explicitly explained as tolerance-caused changes.
- Throughput and transport latency recorded.

**Commit:**
- `git add THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md benchmarks/cluster_h256 && git commit -m "bench: receipt h256 three-board cluster"`

---

## Phase 4: Transformer micro-model before TinyStories

### Task 4.1: Define minimal transformer inference format

**Objective:** Add a simple exported format for a transformer block without overbuilding.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/TRANSFORMER_RILM_V1.md`

**Minimum tensors:**
- token embedding
- positional embedding or RoPE constants
- q_proj, k_proj, v_proj, o_proj
- mlp_up, mlp_down
- layernorm scales/biases
- output projection

**YAGNI exclusions for v1:**
- no batching
- no streaming KV compression
- no tokenizer training
- no WiFi
- no dynamic model loading

**Verification:**
- Format doc includes tensor shapes and shard rules.

**Commit:**
- `git add docs/TRANSFORMER_RILM_V1.md && git commit -m "docs: specify transformer rilm format"`

### Task 4.2: Export a tiny transformer fixture

**Objective:** Build a tiny BPE transformer fixture that can run on host and boards.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_transformer_fixture.py`
- Output:
  - `models/tiny_transformer_ref.json`
  - `models/tiny_transformer_board0.rilm`
  - `models/tiny_transformer_board1.rilm`
  - `models/tiny_transformer_board2.rilm`

**Fixture size:**
- vocab: 256 or 1024
- dim: 64
- layers: 1
- heads: 3 or 6 so head sharding is clean

**Verification:**
- Host Python forward pass emits known next token.

**Commit:**
- `git add tools/export_transformer_fixture.py models/tiny_transformer_* && git commit -m "cluster: export tiny transformer fixture"`

### Task 4.3: Implement transformer primitives on device

**Objective:** Add minimal inference kernels needed for the transformer fixture.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- Optional create:
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/src/transformer_runtime.h`
  - `/home/sikmindz/projects/esp32-s3-lstm-proof/src/transformer_runtime.cpp`

**Primitives:**
- layernorm
- q/k/v projection using existing dot kernels
- causal attention for one token at a time
- MLP projection
- output top-k shard

**Verification:**
- Single board fixture path matches host reference within tolerance.

**Commit:**
- `git add src/main.cpp src/transformer_runtime.* && git commit -m "transformer: add tiny fixture runtime"`

### Task 4.4: Run transformer fixture across 3 boards

**Objective:** Prove the actual transformer execution path in cluster mode.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_transformer.py`
- Update: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`

**Verification:**
- Host reference and cluster output token match for at least 16 steps.
- Record per-token timings:
  - embedding
  - attention
  - MLP
  - output top-k
  - transport

**Commit:**
- `git add tools/run_cluster_transformer.py THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md benchmarks/cluster_transformer_fixture && git commit -m "cluster: run transformer fixture across three boards"`

---

## Phase 5: TinyStories reduced model

### Task 5.1: Select reduced TinyStories model target

**Objective:** Pick the smallest model that proves BPE language generation before 33M.

**Candidate target:**
- dim: 128 or 192
- layers: 2
- heads: 3 or 6
- vocab: real TinyStories tokenizer if possible, reduced vocab only if necessary
- quantization: int4 weights, int8 activations

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/TINYSTORIES_CLUSTER_MODEL_TARGET.md`

**Verification:**
- Document includes estimated per-board memory:
  - weights
  - KV cache
  - activations
  - packet buffers
  - safety margin

**Commit:**
- `git add docs/TINYSTORIES_CLUSTER_MODEL_TARGET.md && git commit -m "docs: choose reduced tinystories cluster target"`

### Task 5.2: Build/export reduced TinyStories model

**Objective:** Produce a real BPE model artifact and shard it for 3 boards.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_tinystories_cluster.py`
- Output:
  - `models/tinystories_reduced_board0.rilm`
  - `models/tinystories_reduced_board1.rilm`
  - `models/tinystories_reduced_board2.rilm`
  - `models/tinystories_reduced_manifest.json`

**Verification:**
- Host reference generation for 8 prompts.
- SHA256 recorded for all shard files.

**Commit:**
- `git add tools/export_tinystories_cluster.py models/tinystories_reduced_* && git commit -m "cluster: export reduced tinystories shards"`

### Task 5.3: Run reduced TinyStories on 3 boards

**Objective:** First real BPE model hardware receipt.

**Files:**
- Create or modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_tinystories.py`
- Update: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`

**Verification prompts:**
- `Once upon a time`
- `The little robot`
- `A sensor said`
- `In a warm room`

**Metrics:**
- real BPE tok/s
- ms/token
- transport ms/token
- per-board compute ms/token
- top-k merge ms/token
- output text

**Gate:**
- Must generate coherent short text locally.
- Must not use host/cloud inference.
- Must emit per-token receipts.

**Commit:**
- `git add tools/run_cluster_tinystories.py THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md benchmarks/tinystories_reduced_cluster && git commit -m "bench: run reduced tinystories on three esp32s3 boards"`

---

## Phase 6: TinyStories-33M attempt

### Task 6.1: Memory feasibility check with exact checkpoint

**Objective:** Prove the exact checkpoint can fit before writing firmware around it.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/analyze_tinystories_33m_memory.py`
- Output: `/home/sikmindz/projects/esp32-s3-lstm-proof/reports/tinystories_33m_memory_report.json`

**Required fields:**
- total params
- int4 bytes
- tied embedding status
- per-shard bytes
- KV cache bytes by context length
- activation scratch bytes
- packet buffer bytes
- per-board free PSRAM estimate

**Gate:**
- At least 1.0MB PSRAM free per board after weights and buffers.

**Commit:**
- `git add tools/analyze_tinystories_33m_memory.py reports/tinystories_33m_memory_report.json && git commit -m "analysis: check tinystories 33m cluster memory"`

### Task 6.2: Export TinyStories-33M shards

**Objective:** Create 3 board-specific model files if memory gate passes.

**Files:**
- Modify: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_tinystories_cluster.py`
- Output:
  - `models/tinystories_33m_board0.rilm`
  - `models/tinystories_33m_board1.rilm`
  - `models/tinystories_33m_board2.rilm`
  - `models/tinystories_33m_manifest.json`

**Verification:**
- Host forward pass on sharded quantized artifacts matches unsharded quantized reference.

**Commit:**
- `git add tools/export_tinystories_cluster.py models/tinystories_33m_* && git commit -m "cluster: export tinystories 33m shards"`

### Task 6.3: Hardware run and performance receipt

**Objective:** Produce the decisive receipt for 3-board TinyStories-33M.

**Files:**
- Use: `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_tinystories.py`
- Output: `/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/tinystories_33m_cluster/`
- Update: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`

**Verification:**
- 3 prompts
- 32 generated BPE tokens each
- no host/cloud generation
- record exact model SHA256s
- record all board serial ports
- record total wall time
- report actual BPE tok/s, not char estimate

**Success:**
- Good: any coherent local generation
- Strong: `>= 10 BPE tok/s`
- Excellent: `>= 10 BPE tok/s` with stable repeated outputs and per-token timing breakdown

**Commit:**
- `git add benchmarks/tinystories_33m_cluster THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md && git commit -m "bench: receipt tinystories 33m on three esp32s3 boards"`

---

## Phase 7: Sensor-grounded demo integration

### Task 7.1: Reconnect sensor-policy prompts to cluster coordinator

**Objective:** Replace single-board language endpoint with cluster-backed local generation.

**Source system:**
- `/home/sikmindz/projects/esp32-sensor-hub/`

**Files likely touched:**
- sensor hub bridge scripts in `/home/sikmindz/projects/esp32-sensor-hub/`
- cluster runner in `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_tinystories.py`

**Verification:**
- Real sensor reading -> deterministic prompt -> cluster output -> OLED/log receipt.

**Commit:**
- Commit in whichever repo owns the changed bridge code.

### Task 7.2: Public demo receipt

**Objective:** Produce the artifact that proves the category: local sensor-grounded language on $15 of boards.

**Files:**
- Create: `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_SENSOR_LANGUAGE_DEMO_2026-07-02.md`

**Required proof:**
- photo/video optional but useful
- board list and approximate cost
- no internet inference statement
- raw sensor values
- generated text
- BPE tok/s
- total latency
- model name/SHA
- exact prompts

**Commit:**
- `git add THREE_BOARD_SENSOR_LANGUAGE_DEMO_2026-07-02.md && git commit -m "docs: receipt three-board sensor language demo"`

---

## Engineering risks and kill criteria

### Risk 1: UART transport too slow

Symptom:
- transport is >30% of token time.

Mitigation:
- top-k logits instead of full logits
- binary packets only
- move to SPI coordinator-master after Phase 2

Kill criterion:
- If UART prevents even H256 cluster proof from matching p22 within reasonable latency, do not attempt TinyStories over UART.

### Risk 2: Numeric drift changes outputs

Symptom:
- cluster H256 output diverges from p22.

Mitigation:
- compare layer/gate intermediate values
- first require exact integer accumulators for matmul fixture
- only then allow float tolerance after activation

Kill criterion:
- If fixture matmul cannot match exactly, cluster model inference is not allowed.

### Risk 3: Coordinator becomes bottleneck

Symptom:
- workers idle while coordinator assembles, activates, samples, or emits serial logs.

Mitigation:
- reduce logging during timed region
- move activation to workers where possible
- shard output projection and return top-k only

Kill criterion:
- If coordinator time exceeds worker compute time, redesign before TinyStories.

### Risk 4: TinyStories-33M does not fit

Symptom:
- less than 1MB free PSRAM per board after weights and buffers.

Mitigation:
- smaller context length
- tied embeddings
- smaller TinyStories checkpoint
- reduced vocab
- 4-board variant

Kill criterion:
- Do not squeeze below 1MB PSRAM safety margin; silent heap failure is worse than a smaller model.

### Risk 5: Public claim overreach

Symptom:
- README says “AI for everyone” before TinyStories hardware receipt exists.

Mitigation:
- keep claim ladder in receipt
- separate prototype claims from product claims

Kill criterion:
- If only H256/domain model works, call it a sensor-language endpoint, not general local AI.

---

## File inventory for implementation

Existing files to reuse:
- `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/platformio.ini`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/extra_script.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_bench.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/lib/esp-nn/src/common/dot_i4_i8_esp32s3.S`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/lib/esp-nn/src/common/dot_i4_i8_wrapper.c`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/P22_INT4_SIMD_KERNEL_RECEIPT_2026-07-02.md`

New files expected:
- `/home/sikmindz/projects/esp32-s3-lstm-proof/src/cluster_protocol.h`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/test_cluster_protocol.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/cluster_ping.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/make_cluster_fixture.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_matmul.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_lstm_cluster_shards.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/flash_cluster.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_h256.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/TRANSFORMER_RILM_V1.md`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_transformer_fixture.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/src/transformer_runtime.h`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/src/transformer_runtime.cpp`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_transformer.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/TINYSTORIES_CLUSTER_MODEL_TARGET.md`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/export_tinystories_cluster.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/analyze_tinystories_33m_memory.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_cluster_tinystories.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/THREE_BOARD_SENSOR_LANGUAGE_DEMO_2026-07-02.md`

---

## Execution order summary

1. Clean current repo state.
2. Capture p22 baseline receipt.
3. Add role-based firmware builds.
4. Add packet protocol and ping test.
5. Prove two-worker barrier synchronization.
6. Prove synthetic sharded matmul exactness.
7. Export and run sharded H256 LSTM.
8. Add transformer fixture format and runtime.
9. Run transformer fixture across 3 boards.
10. Export and run reduced TinyStories.
11. Attempt TinyStories-33M only after memory report passes.
12. Reconnect sensor hub and produce public demo receipt.

---

## First implementation command when ready

Start with:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
git status --short
python3 tools/run_bench.py --port /dev/ttyACM0 --variant p22_i4_wih_whh_simd_h256 --repeat 3 --timeout 180 --out-dir benchmarks/p22_i4_wih_whh_simd_h256
```

Then do Phase 0 before touching transport code.
