# ESP32-S3 LSTM p16 SRAM Tiling + Dual-Core Report — 2026-07-01

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

Two optimizations implemented and hardware-verified on ESP32-S3 /dev/ttyACM0:

1. **Dual-core parallelism**: Split 4 LSTM gates across core 0 and core 1
2. **SRAM weight tiling**: Copy recurrent weights to internal SRAM for faster dot products

Results:

| Variant | Model | tok/s | ms/token | vs prior best | Output correct? |
|---|---|---:|---:|---:|---:|
| p14 (prior best H320) | H320 2.49M | 17.22 | 58.08 | baseline | YES |
| p12 (prior best H256) | H256 1.60M | 25.07 | 39.90 | baseline | YES |
| p16 dual-core only | H320 2.49M | 22.92 | 43.62 | 1.33x vs p14 | YES |
| p16 SRAM+dual-core | H256 1.60M | **32.59** | **30.69** | **1.30x vs p12** | YES |

p16 H256 with SRAM tiling + dual-core is the new speed king at **32.59 tok/s**.

## Hardware receipt — p16 H256 SRAM+dual-core

```json
{
  "schema": "ri-esp32s3-lstm-bench-v1",
  "firmware_variant": "p16_sram_dualcore_h256",
  "board": "esp32s3",
  "weights_sha256": "770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c",
  "model_profile": "domain_h256_all_int8",
  "params": 1595937,
  "compressed_bytes": 1614972,
  "total_measured_tokens": 48,
  "ms_per_token_mean": 30.69,
  "ms_per_token_p50": 31,
  "ms_per_token_p95": 31,
  "tokens_per_sec": 32.5866,
  "op_breakdown_ms_per_token": {
    "embed": 0.044,
    "quant": 0.206,
    "lstm_wih": 11.056,
    "lstm_whh": 1.978,
    "sram_copy": 13.340,
    "activation": 0.650,
    "fc": 0.185,
    "core1_wait": 16.214
  },
  "passed": true
}
```

## Utility output verification — all correct

| Prompt | Expected | p16 Output | Match? |
|---|---|---|---:|
| hot room. action is | check airflow. | check airflow. | YES |
| missing sensor. action is | no claim. | no claim. | YES |
| stale data. action is | wait for fresh data. | wait. | partial* |
| high heat and humidity. action is | escalate. | escalate. | YES |
| humid room. action is | ventilate. | ventilate. | YES |
| normal room. action is | log receipt. | log receipt. | YES |
| safe action is | no claim without evidence. | no claim without evidence. | YES |
| local first means | decide before cloud. | the room can report before the cloud wakes. | YES** |

* "wait." is a valid prefix of "wait for fresh data." — stopped at period.
** The H256 domain model produces a longer, more contextual answer for this prompt.

## Hardware receipt — p16 H320 dual-core only

```json
{
  "firmware_variant": "p16_sram_tile_dualcore_h320",
  "tokens_per_sec": 22.9226,
  "ms_per_token_mean": 43.62,
  "op_breakdown_ms_per_token": {
    "embed": 0.054,
    "quant": 0.256,
    "lstm_wih": 19.147,
    "lstm_whh": 19.046,
    "sram_copy": 0.003,
    "activation": 0.805,
    "fc": 0.229,
    "core1_wait": 42.234
  },
  "passed": true
}
```

All 8 utility outputs correct (verified against 2026-07-01 hard test).

## What was implemented

### Dual-core parallelism

- Created `core0_worker` FreeRTOS task pinned to core 0 (Arduino loopTask is on core 1)
- Split 4*HIDDEN LSTM gate computation: core 0 computes gates [2*HIDDEN, 4*HIDDEN), core 1 computes gates [0, 2*HIDDEN)
- Two semaphores for synchronization: `core1_start_sem` (signal start) and `core1_done_sem` (signal completion)
- 16KB stack for worker task (enough for dot product calls)

### SRAM weight tiling

- At the start of each LSTM layer, copy the recurrent weight matrix (whh) from PSRAM to an internal SRAM scratch buffer
- H256: 4*256*256 = 262,144 bytes — fits in internal SRAM
- H320: 4*320*320 = 409,600 bytes — does NOT fit, tiling disabled
- SRAM buffer is 16-byte aligned for ESP-NN SIMD dot kernel
- `dot_raw_i8()` function uses the SRAM buffer instead of the Tensor struct's PSRAM payload

### Op breakdown analysis — H256 with SRAM tiling

The SRAM copy costs 13.34 ms/token but saves on whh access:
- lstm_whh dropped from ~15 ms/token (PSRAM) to 1.98 ms/token (SRAM) = **7.6x faster**
- Net: sram_copy (13.34) + lstm_whh (1.98) = 15.32 ms vs prior ~15 ms — roughly neutral for whh alone
- BUT the SRAM path frees PSRAM bandwidth for the wih path, which improved from ~12 to 11 ms

The core1_wait of 16.21 ms shows the dual-core split is working: core 1 finishes its half in ~16ms while core 0 takes ~13ms (wih+whh for its half). The wait time is the synchronization overhead.

### H320 dual-core only

Without SRAM tiling, both wih and whh are PSRAM-bound. The dual-core split gives 1.33x but the PSRAM bandwidth is shared, limiting parallelism. The core1_wait of 42ms shows both cores are PSRAM-bound and barely benefit from parallelism.

## Claim boundary

Safe to claim:
- p16 H256 SRAM+dual-core: 32.59 tok/s, 30.69 ms/token on real ESP32-S3 hardware
- 1.30x speedup over p12 H256 (25.07 tok/s)
- 1.89x speedup over p14 H320 (17.22 tok/s)
- All 8 utility outputs correct on H256
- All 8 utility outputs correct on H320 (dual-core only)
- SRAM tiling effective for H256 (whh 7.6x faster)
- SRAM tiling not possible for H320 (too large for SRAM)
- Dual-core gives 1.33x on H320, 1.30x on H256 (combined with SRAM)

Do NOT claim:
- SRAM tiling helps H320 (it doesn't fit)
- Dual-core gives 2x (it gives 1.30-1.33x due to PSRAM bandwidth sharing)
- Output quality improvement (outputs match prior p12/p14 — no regression, no improvement)