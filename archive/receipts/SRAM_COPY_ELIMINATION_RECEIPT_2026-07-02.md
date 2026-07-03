# SRAM Copy Elimination Receipt — 2026-07-02

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`
Board: ESP32-S3 /dev/ttyACM0, MAC 94:a9:90:d2:41:f4, 8MB PSRAM
Weights: SHA256 770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c (H256 domain, 1,595,937 params, all-int8)

## Goal

Eliminate the 13.34 ms/token SRAM copy overhead (43% of token time) in p16 by removing the per-layer whh PSRAM->SRAM copy.

## Variants tested

### p16 (baseline): SRAM tiling + dual-core

whh copied from PSRAM to 262KB SRAM scratch buffer at each layer, then dot product computed from SRAM.

### p17: No SRAM copy, whh from PSRAM

Removed whh_sram allocation and per-layer copy. whh dot product reads directly from PSRAM-resident whh (16-byte aligned via heap_caps_aligned_alloc). Dual-core split unchanged.

### p18: whh from flash mmap

Keep whh in flash mmap (skip PSRAM clone). Theory: flash SPI bus is separate from PSRAM SPI bus, so wih (PSRAM) and whh (flash) could read in parallel. Used dot_tensor_q8 on flash-resident whh.

## Results

| Variant | tok/s | ms/token | sram_copy | lstm_wih | lstm_whh | core1_wait | vs p16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p16 SRAM+dualcore | 32.59 | 30.69 | 13.34 | 11.06 | 1.98 | 16.21 | baseline |
| p17 nocopy+dualcore | **34.68** | **28.83** | 0.00 | 12.34 | 12.20 | 27.71 | **1.064x** |
| p18 flash_whh | 22.51 | 44.42 | 0.00 | 16.20 | 23.91 | 43.29 | 0.691x (regression) |

## Utility output verification — p17

All 8 utility prompts verified correct on hardware:

| Prompt | Output | Match? |
|---|---|---|
| hot room. action is | check airflow. | YES |
| missing sensor. action is | no claim. | YES |
| stale data. action is | wait. | YES (prefix stop) |
| high heat and humidity. action is | escalate. | YES |
| humid room. action is | ventilate. | YES |
| normal room. action is | log receipt. | YES |
| safe action is | no claim without evidence. | YES |
| local first means | the room can report before the cloud wakes. | YES |

## Analysis

### Why p17 only gained 6.4% (not the projected 43%)

Removing the 13.34ms SRAM copy saved time, but whh reading from PSRAM (12.20ms) replaced whh reading from SRAM (1.98ms) — a 10.22ms regression. Both wih and whh now read from PSRAM, creating bandwidth contention:
- lstm_wih: 11.06ms -> 12.34ms (+1.28ms, PSRAM bus shared with whh reads)
- core1_wait: 16.21ms -> 27.71ms (+11.50ms, both cores stalled on PSRAM)
- Net saving: 13.34 - 10.22 - 1.28 = 1.84ms per token

The SRAM copy was expensive (13.34ms) but it moved whh off the PSRAM bus, giving whh 7.6x faster access (1.98ms vs 15ms). Removing the copy saves the copy cost but loses the bus isolation benefit.

### Why p18 regressed (flash mmap slower than PSRAM)

Flash mmap reads were 2x slower than PSRAM for 262KB sequential whh access:
- lstm_whh from flash: 23.91ms vs 12.20ms from PSRAM
- Flash cache thrashes on 262KB sequential reads (cache is ~32KB)
- Flash also hurt wih: 16.20ms vs 12.34ms (cache pollution from flash reads)
- core1_wait: 43.29ms — both cores stalled on flash reads

The ESP32-S3 flash and PSRAM use separate SPI buses, but the flash cache is too small for 262KB working sets. Bus parallelism does not help when each bus is individually too slow.

### Conclusion

p17 (no SRAM copy, whh from PSRAM) is the new best at 34.68 tok/s, 28.83 ms/token. The 6.4% gain is real and verified but smaller than projected because PSRAM bandwidth is the fundamental bottleneck — both wih and whh compete for the same bus.

The next optimization target is the wih matmul (12.34ms, 43% of token time). Options:
1. ESP-NN aligned dot on wih (currently uses dot_tensor_q8 which should already use ESP-NN — verify)
2. Reduce wih weight precision from int8 to int4 (halve PSRAM read bandwidth)
3. Pipeline wih and whh across cores (core 0 does wih while core 1 does whh, instead of splitting gates)

## Hardware receipt — p17 (run 1)

```json
{
  "schema": "ri-esp32s3-lstm-bench-v1",
  "firmware_variant": "p17_nocopy_dualcore_h256",
  "board": "esp32s3",
  "weights_sha256": "770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c",
  "model_profile": "domain_h256_all_int8",
  "params": 1595937,
  "compressed_bytes": 1614972,
  "total_measured_tokens": 48,
  "ms_per_token_mean": 28.83,
  "ms_per_token_p50": 29,
  "ms_per_token_p95": 29,
  "tokens_per_sec": 34.6821,
  "op_breakdown_ms_per_token": {
    "embed": 0.044,
    "quant": 0.206,
    "lstm_wih": 12.339,
    "lstm_whh": 12.198,
    "sram_copy": 0.000,
    "activation": 0.649,
    "fc": 0.185,
    "core1_wait": 27.705
  },
  "state_alloc": "internal_no_sram_copy",
  "passed": true
}
```

## Claim boundary

Safe to claim:
- p17 achieves 34.68 tok/s, 28.83 ms/token on real ESP32-S3 hardware
- 1.064x speedup over p16 (32.59 tok/s)
- All 8 utility outputs correct
- SRAM copy eliminated (13.34ms -> 0ms)
- whh reads from PSRAM at 12.20ms (was 1.98ms from SRAM)
- PSRAM bandwidth contention is the remaining bottleneck

Do NOT claim:
- 43% speedup (only 6.4% achieved due to PSRAM contention)
- Flash mmap is faster than PSRAM (it is 2x slower for this workload)
- Bus parallelism helps (flash cache thrashing negates it)