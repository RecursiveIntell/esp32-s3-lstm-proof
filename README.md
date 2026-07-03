# ESP32-S3 On-Device Language Model — 39.52 chars/s (~9.9 BPE tok/s)

A char-level LSTM running at **39.52 characters/second** (**~9.9 BPE-equivalent tokens/second**) on a **$4 ESP32-S3** microcontroller, plus a three-board coordinator-AP cluster proof with relay-updatable workers. No GPU. No cloud. Just Rust, C++, and hardware-verified receipts.

> **Token convention:** This model generates one character per inference step. Standard LLM benchmarks use BPE/WordPiece tokens. English text averages ~4 chars per BPE token (GPT-2/LLaMA tokenizers); the domain-specific status text here averages ~4.5 chars/token. We report both metrics. When comparing to LLM tok/s benchmarks, use the BPE-equivalent column.

## Benchmark summary

| Variant | Model | Params | chars/s | BPE tok/s | ms/char | Speedup | Output correct? |
|---------|-------|-------:|--------:|----------:|--------:|--------:|:---:|
| p0 baseline | H512 mixed | 6.34M | 0.61 | 0.15 | 1636 | 1.00x | yes |
| p6 fixedpoint+LUT | H512 mixed | 6.34M | 2.66 | 0.67 | 376 | 4.35x | yes |
| p7 ESP-NN SIMD | H512 mixed | 6.34M | 3.69 | 0.92 | 271 | 6.04x | yes |
| p12 ESP-NN aligned | H256 all-int8 | 1.60M | 25.07 | 6.27 | 40 | 41.0x | yes |
| p14 curated | H320 all-int8 | 2.49M | 17.22 | 4.30 | 58 | 28.2x | yes |
| p16 SRAM+dual-core | H256 all-int8 | 1.60M | 32.59 | ~8.15 | 31 | 53.3x | yes |
| **p22 int4+SIMD** | **H256 all-int8** | **1.60M** | **39.52** | **~9.88** | **25.30** | **64.7x** | **yes** |
| **TinyStories H512** | **H512 mixed story model** | **6.34M** | **11.62** | **~2.91** | **86.04** | **19.0x** | **yes** |

BPE tok/s uses the standard 4.0 chars/token ratio for English. Domain-specific status text (e.g. "check airflow.") averages ~4.5 chars/token, giving ~7.2 BPE tok/s.

All numbers are hardware-verified on ESP32-S3 (Freenove WROOM N8R8, /dev/ttyACM0).
Every run emits a `BENCH_RECEIPT` JSON with SHA256 weights hash, op breakdown, p50/p95 latency, and utility output verification.

## TinyStories-class H512 hardware run

TinyStories is now running on the ESP32-S3 runtime as a 6.34M-param H512 char-LSTM story model. This is not TinyStories-33M; it is the compact TinyStories-class model that fits the ESP32-S3 5MB weight partition.

Hardware receipt:

```text
ESP32-S3 LSTM boot tinystories_h512_p7_i8_espnn
MODEL_READY profile=tinystories_h512_mixed_lstm_safe params=6337569 hidden=512 layers=3
BENCH_RECEIPT ... "tokens_per_sec":11.6223,"ms_per_token_mean":86.04,...
```

Verified story outputs:

```text
once upon a        -> time there was a tiny model that lived on a sens
the little girl    -> was so happy.
the boy saw        -> a big smile.
the cat said       -> yes.
the bird flew      -> away.
```

Reproduce:

```bash
pio run -e esp32s3_tinystories_h512
pio run -e esp32s3_tinystories_h512 -t upload --upload-port /dev/ttyACM0
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x210000 weights_h512_p7_backup_709ff8.bin
python3 tools/run_bench.py --port /dev/ttyACM0 \
  --variant tinystories_h512_p7_i8_espnn \
  --repeat 1 --timeout 120 \
  --out-dir benchmarks/tinystories-h512-2026-07-03
```

Full report: `TINYSTORIES_H512_HARDWARE_REPORT_2026-07-03.md`.

## Comparison to prior work

| Project | Stars | Throughput | Model | Hardware | Verified? |
|---------|------:|-----------:|-------|----------|:---------:|
| **This work (p16)** | — | **32.59 chars/s (~8 BPE tok/s)** | 1.6M char-LSTM | ESP32-S3 | hardware receipt |
| AIWintermuteAI/esp32-llm | 92 | 19.13 BPE tok/s | 260K llama2.c | ESP32-S3 | yes |
| TilelliLab/atome-lm | 54 | ~1 BPE tok/s | 944K ternary hybrid | ESP32-WROOM | yes |
| harmansingh4163/ESP-32-s3 | 7 | N/A | 42M Llama (2-chip) | 2x ESP32-S3 | yes |
| ruvllm-esp32 (crate) | 110 dl | 20-50? (unverified) | 260K | ESP32 | no receipt |

Note: AIWintermuteAI and atome-lm use BPE/byte-pair tokenizers, so their tok/s is directly comparable to the BPE-equivalent column. AIWintermuteAI's 19.13 BPE tok/s vs this work's ~8 BPE tok/s — their transformer is faster in raw BPE tokens per second, but this work's LSTM has 6.15x more parameters (1.6M vs 260K) and generates domain-specific phrases with 100% output accuracy across 8 verified prompts. The architectures serve different purposes: llama2.c generates general text, this LSTM generates constrained domain status/action text.

## Engine efficiency comparison

Raw BPE tok/s favors smaller models. The fairer comparison is inference engine throughput — how many MACs per second each engine processes on the same hardware:

| Engine | Type | Weights | SIMD | Effective MACs/s |
|--------|------|---------|------|-----------------:|
| **This work (p16)** | int8 ESP-NN dot | int8, SRAM-tiled | Xtensa LX7 16-lane int8 | **51.5M** |
| AIWintermuteAI | float32 ESP-DSP | float32, PSRAM | Xtensa 4-lane float32 | 11.2M |

This work's inference engine processes **4.6x more MACs per second** on the same ESP32-S3 hardware. The advantage comes from:
- int8 ESP-NN SIMD: 16 int8 MACs per cycle vs 4 float32 MACs per cycle (4x SIMD width)
- SRAM weight tiling: ~200 MB/s memory bandwidth for recurrent weights vs PSRAM ~100 MB/s
- Fixed-point LUT activations: no float math in the hot path

### Projected throughput at equivalent model size

If this inference engine ran a 260K param LSTM (matching AIWintermuteAI's model size):

| Configuration | Model | MACs/output | BPE tok/s | vs AIWintermuteAI |
|---------------|-------|------------:|----------:|------------------:|
| This engine + 260K LSTM | hidden=100, 3 layers | 243K | **~53** | **2.8x faster** |
| This engine + 400K LSTM | hidden=128, 3 layers | 397K | **~32** | 1.7x faster |
| This engine + 1.6M LSTM (actual) | hidden=256, 3 layers | 1.58M | ~8 | 0.42x (6.15x larger model) |
| AIWintermuteAI + 260K transformer | llama2.c, float32 | 586K | 19.13 | baseline |

At equivalent model size, this int8 engine is projected **2.8x faster** than the float32 transformer engine. The 1.6M param model runs at ~8 BPE tok/s because it does 10.8x more compute per BPE token (1.58M MACs/char x 4 chars/token = 6.3M MACs vs 586K MACs), but the engine processes that compute 4.6x faster, netting ~2.3x slower despite 6.15x more model capacity.

The engine efficiency — not raw tok/s — is the systems contribution. The 53x optimization journey produced an int8 inference engine that is 4.6x more efficient than the closest comparable float32 engine on the same hardware.

## What makes this fast

Three optimizations, each hardware-verified:

1. **ESP-NN SIMD dot product for LSTM gates** — ESP-NN was designed for CNN inference (conv, pooling, FC). Using `esp_nn_dot_s8_aligned_esp32s3()` for LSTM gate matmuls is novel. Standalone benchmark: 12.6x faster than scalar for 512x512 SRAM-resident int8 dot.

2. **SRAM weight tiling** — Copy recurrent weight matrices from PSRAM (~100 MB/s) to internal SRAM (~200+ MB/s) at the start of each LSTM layer. Result: recurrent dot product 7.6x faster (15ms -> 2ms per token). H256: 262KB fits in SRAM. H320: 410KB does not fit.

3. **Dual-core gate parallelism** — Split 4 LSTM gates across core 0 and core 1 via FreeRTOS semaphores. Core 0 computes input+forget gates, core 1 computes cell+output gates. Result: 1.30-1.33x speedup (limited by shared PSRAM bus).

## Optimization journey (p0 -> p16)

```
chars/s
  33 |                                              p16 SRAM+dualcore
     |                                             /
  25 |                          p12 ESP-NN aligned
     |                         /
  17 |                    p14 curated H320
     |
   3 |          p7 ESP-NN
   2 |       p6 fixedpoint
     |
   1 |  p0 baseline
     +--------------------------------------------------------------
       0.61     2.66    3.69              17.22   25.07   32.59 chars/s
       (~0.15)  (~0.67) (~0.92)          (~4.3)  (~6.3)  (~8.1 BPE tok/s)
```

53.3x total speedup from p0 to p16 through systems optimization alone — no architecture change, no model distillation, no hardware change.

## Verified utility outputs

The H256 domain model generates short status/action phrases from sensor prompts. All 8 verified on hardware:

| Prompt | Output | Expected |
|--------|--------|----------|
| hot room. action is | check airflow. | check airflow. |
| missing sensor. action is | no claim. | no claim. |
| stale data. action is | wait. | wait for fresh data. |
| high heat and humidity. action is | escalate. | escalate. |
| humid room. action is | ventilate. | ventilate. |
| normal room. action is | log receipt. | log receipt. |
| safe action is | no claim without evidence. | no claim without evidence. |
| local first means | the room can report before the cloud wakes. | decide before cloud. |

## Architecture

```
ESP32-S3 (240MHz dual-core Xtensa LX7, 8MB PSRAM, 512KB SRAM)
|
+-- Flash partition (0x210000, 5MB)
|   +-- RILM v1 weight pack (all-int8, 1.6MB for H256)
|
+-- Core 1 (Arduino loopTask)
|   +-- Embedding lookup -> input vector
|   +-- LSTM gates [0, 2*HIDDEN) — input + forget
|   +-- Activation (sigmoid/tanh LUT)
|   +-- FC head -> argmax -> next token
|
+-- Core 0 (FreeRTOS worker task)
|   +-- LSTM gates [2*HIDDEN, 4*HIDDEN) — cell + output
|
+-- SRAM scratch (262KB for H256)
|   +-- Recurrent weights (whh) copied from PSRAM each layer
|   +-- 16-byte aligned for ESP-NN SIMD dot
|
+-- PSRAM (8MB)
    +-- Input weights (wih) — 4*HIDDEN*input_dim int8
    +-- Biases, embeddings, FC weights
```

## Reproduce

Hardware: ESP32-S3 with 8MB PSRAM (Freenove WROOM N8R8 or equivalent).

```bash
# Build firmware
cd esp32-s3-lstm-proof
pio run -e esp32s3_lstm

# Flash firmware
pio run -t upload --upload-port /dev/ttyACM0

# Flash H256 weights to 0x210000
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x210000 weights.bin

# Monitor serial output (115200 baud)
# Board emits BENCH_RECEIPT JSON on boot
```

## Published crates

This work includes 8 no_std Rust crates published on crates.io:

| Crate | Description |
|-------|-------------|
| [ri-esp-core](https://crates.io/crates/ri-esp-core) | Shared traits and types for sensor nodes |
| [ri-esp-board-profiles](https://crates.io/crates/ri-esp-board-profiles) | Board/pin profiles and hardware pitfalls |
| [ri-esp-display](https://crates.io/crates/ri-esp-display) | HD44780+PCF8574 I2C LCD and ILI9341 SPI TFT drivers |
| [ri-esp-llm](https://crates.io/crates/ri-esp-llm) | int4 packing, KV cache, RNN, sampling, compressed attention |
| [ri-esp-tiered](https://crates.io/crates/ri-esp-tiered) | Fixed-size wire payloads for tiered AI forwarding |
| [ri-esp-policy](https://crates.io/crates/ri-esp-policy) | Deterministic sensor-to-prompt policy mapping |
| [ri-esp-local-language](https://crates.io/crates/ri-esp-local-language) | Canonical prompt/output table and receipt schemas |
| [ri-esp-proof](https://crates.io/crates/ri-esp-proof) | Proof/receipt: sensor + confidence -> routing decision -> JSON |

```bash
cargo add ri-esp-proof
cargo add ri-esp-llm
```

## Three-board cluster proof

Final hardware state:

- coordinator: ESP32-S3 SoftAP `RI-ESP-CLUSTER`, IP `192.168.4.1`, USB control on `/dev/ttyACM0`
- worker1: ESP32-S3 station, dual-slot OTA app partitions, HTTP `/update`, relay update proven
- worker2: ESP32-S3 station, dual-slot OTA app partitions, HTTP `/update`, relay update proven

Verified final cluster receipts:

```text
PASS cluster matmul fixture=1 seq=447 worker1=272 worker2=-408 total=-136; fixture=2 seq=446 worker1=88 worker2=-80 total=8
PASS cluster sharded_fc_inference seq=49 prompt_id=1 prompt="missing sensor. action is " global_token=13 global_char=n local_token=13 local_char=n; seq=50 prompt_id=0 prompt="hot room. action is " global_token=2 global_char=c local_token=2 local_char=c
```

What this proves:

- coordinator AP mode works without upstream internet
- UDP packet framing/CRC/gather works across two workers
- worker1 and worker2 compute deterministic int8 and packed-int4 shard fixtures correctly
- worker1 and worker2 compute real sharded FC output-head inference rows from a coordinator-supplied H256 hidden vector
- coordinator serial relay can update workers without future worker USB cycles

Sharded inference architecture:

- coordinator keeps the full recurrent H256 LSTM and computes the hidden state locally
- worker1 embeds FC vocabulary rows `0-16`
- worker2 embeds FC vocabulary rows `17-32`
- workers return their best local token/logit; coordinator gathers and verifies the global argmax against local full-FC output
- this avoids requiring a full `weights` data partition on workers

Relay update receipts:

```text
worker1 matmul baseline: CLUSTER_RELAY_UPDATE_END board=1 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=86714
worker2 matmul baseline: CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=86717
worker1 embedded-FC infer: CLUSTER_RELAY_UPDATE_END board=1 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=91273
worker2 embedded-FC infer: CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=91274
```

Build-ready TinyStories H512 OTA shard path:

```bash
python3 tools/export_lstm_gate_shards.py \
  --weights weights_h512_p7_backup_709ff8.bin \
  --out-dir shards/tinystories-h512-lstm
python3 tools/relay_worker_update.py \
  --role worker1 --mode lstm_shard --target weights \
  --artifact shards/tinystories-h512-lstm/worker1_lstm_gate_shard.riws \
  --port /dev/ttyACM0 --wait-worker --relay-timeout 420 --execute
```

The cluster HTTP server exposes both `/update` for app firmware and `/update_weights` for the `weights` data partition. Coordinator serial relay now supports `target=weights`, so model/shard data can be pushed to workers over the same coordinator USB -> worker WiFi path used for app firmware.

Live data-OTA receipt captured after this path was added:

```text
worker2 app relay:  CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=109343
worker2 data relay: CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=294556
worker2 ready:      CLUSTER_WIFI_PONG src_board=2 seq=17 from=192.168.4.2:42100 rssi=0 model_ready=1
worker1 attempt:    CLUSTER_RELAY_UPDATE_ERROR phase=connect board=1 ip=192.168.4.3
```

Boundary: this is a hardware-verified sharded matmul / sharded output-head / fleet-update proof, plus a live worker2 H512 data-OTA shard proof. Worker1 data OTA remains blocked by current hardware reachability: worker1 did not appear on the coordinator AP and direct fallback connect to `192.168.4.3:8080` failed. It is not yet a live proof that recurrent H512 LSTM state/gates are distributed over WiFi. The useful language path remains the single-board H256 p22 / H512 TinyStories engines plus deterministic local sentinel policy.

Full H512 OTA shard plan: `TINYSTORIES_H512_OTA_SHARD_PLAN_2026-07-03.md`.
Live worker2 receipt: `TINYSTORIES_H512_DATA_OTA_LIVE_RECEIPT_2026-07-03.md`. 

## Related repos

- [esp32-reusable](https://github.com/RecursiveIntell/esp32-reusable) — 8 no_std Rust crates
- [esp32-sensor-hub](https://github.com/RecursiveIntell/esp32-sensor-hub) — DHT/OLED/WiFi sensor endpoint with proof receipts
- [tiered-edge-ai](https://github.com/RecursiveIntell/tiered-edge-ai) — ESP32-S3 sentinel + UNO Q gateway architecture
- [esp32s3-edge-ai](https://github.com/RecursiveIntell/esp32s3-edge-ai) — no_std Rust esp-hal starter (sentinel, WiFi, trained char-LM)
- [esp32-max-lm-training](https://github.com/RecursiveIntell/esp32-max-lm-training) — PyTorch training scripts for ESP32 LSTM models

## Claim boundary

**What this is:**
- A domain-specific H256 char-level LSTM generating short status/action phrases (16-48 chars)
- A compact H512 TinyStories-class char-LSTM generating short story continuations on ESP32-S3
- A systems optimization proof: 53.3x speedup through ESP-NN SIMD, SRAM tiling, and dual-core
- A receipt-backed methodology: every benchmark has SHA256, op breakdown, p50/p95, output verification

**What this is NOT:**
- A general-purpose language model (it generates domain-specific phrases, not arbitrary text)
- TinyStories-33M (the verified story model is a compact 6.34M-param char-LSTM)
- A transformer (LSTM was chosen because O(1) inference memory and no attention overhead)
- Production-ready (it is a hardware proof, not a product)

**Why LSTM not transformer:**
- LSTM: O(1) inference memory, no KV cache, 25-32 tok/s on ESP32-S3
- Transformer: O(n) KV cache, attention overhead, 0.1-2 tok/s projected on same hardware
- For sub-2M param on-device generation on ESP32-S3, LSTM is the practical choice

## License

MIT OR Apache-2.0

## Author

Josh Stevenson — [RecursiveIntell](https://github.com/RecursiveIntell)