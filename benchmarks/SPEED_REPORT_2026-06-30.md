# ESP32-S3 LSTM Speed Benchmark Report — 2026-06-30

Project: /home/sikmindz/projects/esp32-s3-lstm-proof

## Bottom line

The ROI speed work improved local ESP32-S3 LSTM generation from about 0.61 tok/sec to 2.66 tok/sec.

Final speedup vs original proof: 4.35x
Latency reduction vs original proof: 77.03%

This is a real hardware measurement on the ESP32-S3, not a host simulation.

## Hardware and model

Board:
- ESP32-S3 Freenove WROOM N8R8
- Port: /dev/ttyACM0
- PSRAM: 8,386,279 bytes detected

Weights:
- Profile: mixed_lstm_safe
- Params: 6,337,569
- Compressed bytes: 4,785,276
- SHA256: 709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61

## Variants tested

| Variant | Mean ms/token | Tok/sec | Speedup vs original | Speedup vs p0_p4 | Status | Notes |
|---|---:|---:|---:|---:|---|---|
| original proof | 1636.00 | 0.611 | 1.00x | 0.258x | baseline | scalar generic path + debug prints |
| p0_p4_cached_internal_dtype_fastpath | 421.94 | 2.370 | 3.88x | 1.00x | keep | no-debug, cached tensors, internal state, dtype fast paths |
| p5_fixedpoint_lut | 397.06 | 2.519 | 4.12x | 1.06x | keep | q8 activation/state quant, int4/int8 x q8 dots, sigmoid/tanh LUT |
| p6_psram_weights_fixedpoint_lut | 375.80 | 2.661 | 4.35x | 1.12x | final keep | copies payloads to PSRAM/internal before inference |
| p7_o3_psram_weights_fixedpoint_lut | 375.81 | 2.661 | 4.35x | 1.12x | neutral/kill | -O3 gave no measurable win over -O2 |
| p8_iram_psram_fixedpoint_lut | 376.19 | 2.658 | 4.35x | 1.12x | kill | IRAM_ATTR hot functions slightly worse |

Final deployed firmware variant:
- p6_psram_weights_fixedpoint_lut
- build flags: -O2, CORE_DEBUG_LEVEL=0

## Raw receipts

p0_p4:
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p0_p4/p0_p4_cached_internal_dtype_fastpath-summary.json

p5:
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p5_fixedpoint_lut/p5_fixedpoint_lut-summary.json

p6:
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p6_psram_weights/p6_psram_weights_fixedpoint_lut-summary.json
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p6_final/p6_psram_weights_fixedpoint_lut_final-summary.json

p7 O3 probe:
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p7_o3_psram_weights/p7_o3_psram_weights_fixedpoint_lut-summary.json

p8 IRAM probe:
- /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p8_iram/p8_iram_psram_fixedpoint_lut-summary.json

## Verification commands

Build:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
. /home/sikmindz/projects/esp32-sensor-hub/.venv/bin/activate
pio run
```

Result:
- passed

Final flash + benchmark:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
. /home/sikmindz/projects/esp32-sensor-hub/.venv/bin/activate
python3 tools/run_bench.py --port /dev/ttyACM0 \
  --variant p6_psram_weights_fixedpoint_lut_final \
  --repeat 1 \
  --timeout 240 \
  --out-dir benchmarks/p6_final \
  --flash
```

Result:
- flash passed
- receipt parsed
- mean: 375.81 ms/token
- throughput: 2.6609 tok/sec

Best repeated-run aggregate:

```json
{
  "runs": 3,
  "ms_per_token_mean": 375.7967,
  "ms_per_token_min": 375.77,
  "ms_per_token_max": 375.81,
  "tokens_per_sec_mean": 2.6610,
  "tokens_per_sec_min": 2.6609,
  "tokens_per_sec_max": 2.6612
}
```

## Final output samples

From p6/p5 path, 16 measured tokens per seed:

| Seed | Output |
|---|---|
| `the esp32 sees ` | `arm air and said` |
| `once upon a ` | `ime, there was a` |
| `sensor says ` | `hat day.\nendofte` |

Output remained stable across p0_p4, p5, and p6 for the benchmark seed set.

## Final op breakdown

p6 mean per measured token:

| Operation | ms/token | Notes |
|---|---:|---|
| embed | ~0.044 | negligible |
| quant | ~0.522 | q8 activation/state quantization overhead |
| LSTM input matmul | ~227.05 | largest bottleneck |
| LSTM recurrent matmul | ~152.61 | second-largest bottleneck |
| activation | ~2.03 | LUT fixed the old 10.47 ms activation cost |
| FC projection | ~1.29 | negligible |

p6 still spends ~379.66 ms/token in LSTM input+recurrent matmul. That is still ~95% of measured work.

## What each ROI pass bought

p0_p4:
- Removed pathological overhead.
- Biggest win: 3.88x.
- This was mostly engineering hygiene: no per-layer serial spam, no string lookup, internal state, dtype-specialized dot kernels.

p5:
- Added q8 activation/state quantization and int4/int8 x q8 int32 accumulation.
- Added sigmoid/tanh LUT.
- Improved 421.94 -> 397.06 ms/token.
- Additional 1.06x over p0_p4.
- Output stayed stable on benchmark seeds.

p6:
- Copied RILM payloads out of flash mmap into PSRAM/internal memory at boot.
- Improved 397.06 -> 375.80 ms/token.
- Additional 1.06x over p5.
- Final cumulative: 4.35x over original.

p7:
- Tried -O3.
- No measurable gain.
- Keep -O2 to avoid Xtensa compiler edge cases.

p8:
- Tried IRAM_ATTR on hot functions.
- Slightly worse.
- Reverted.

## Interpretation

The easy/medium ROI work is done.

The remaining bottleneck is not framework overhead anymore. It is raw matrix-vector math for a 512-hidden, 3-layer LSTM:

- 4 gates
- 3 layers
- 512 hidden width
- input and recurrent matrices per layer
- millions of int4/int8 weight reads and MACs per token

The current scalar C++ path is now reasonably clean. Further large gains require one of:

1. ESP-NN / Xtensa SIMD fully-connected kernel integration.
2. Hand-written SIMD/asm int8 dot kernels.
3. A speed-oriented model layout/export format: RILM v2 row/block alignment and maybe gate-interleaved layout.
4. Smaller/faster model architecture: H384/H256, Atome-LM-like ternary/SSM, or sensor-specific sentinel instead of general char-LSTM text generation.

## Next highest ROI

1. Build a standalone ESP-NN / SIMD fully-connected probe.
   Do not wire it into the model first. Prove kernel speed on one 2048x512 matrix-vector multiply.

2. Export RILM v2 with row alignment and row metadata.
   RILM v1 is size-first. The next format should be speed-first.

3. Train/export H384x3 and H256x3 variants.
   Benchmark quality/speed directly. If H384 gives ~5 tok/sec with tolerable PPL/output loss, it is a better interactive OLED model than H512x3.

4. Keep H512x3 as the “large local proof model.”
   Use smaller models for UX/sentinel flows.

## Claim boundary

Safe to claim:
- The compressed 6.34M-param mixed int4/int8 LSTM runs on ESP32-S3 hardware.
- The ROI speed pass improved measured local generation from ~0.61 tok/sec to ~2.66 tok/sec.
- The measured cumulative speedup is ~4.35x on the same board and same compressed weights.
- p6 is stable across repeated hardware runs and keeps the benchmark outputs unchanged.
- Matmul remains the dominant bottleneck.

Do not claim yet:
- real-time chatbot UX
- ESP-NN/SIMD acceleration
- production-ready local LLM runtime
- general LLM benchmark superiority
- quality equivalence beyond the fixed benchmark seed outputs and prior host PPL receipts
