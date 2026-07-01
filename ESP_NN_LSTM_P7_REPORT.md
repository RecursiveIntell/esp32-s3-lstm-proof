# ESP-NN LSTM p7 integration report — 2026-07-01

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

ESP-NN S3 SIMD dot was wired into the real LSTM firmware and verified on ESP32-S3 hardware.

Result: **2.661 tok/s → 3.686 tok/s**.

That is a real **1.39x end-to-end speedup** over p6, but not the projected 8.5 tok/s. The direct dot integration helps, but the remaining input-matmul path is still scalar int4 and the recurrent matrices still read from PSRAM, so the standalone 12.6x 512x512 kernel result does not transfer cleanly to the full LSTM.

## What changed

- Copied ESP-NN into the LSTM PlatformIO project: `lib/esp-nn/`
- Added PlatformIO ESP-NN build flags and `lib_ldf_mode = deep+`
- Removed P4 files incompatible with Xtensa
- Included `esp_nn.h`
- Replaced the i8×i8 dot hot path with `esp_nn_dot_s8_unaligned_esp32s3()` when `n % 16 == 0`
- Firmware variant changed to `p7_esp_nn_s3_dot_recurrent`

Files touched:

- `/home/sikmindz/projects/esp32-s3-lstm-proof/platformio.ini`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/extra_script.py`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/lib/esp-nn/`
- `/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

## Hardware receipt

Command path: built with PlatformIO, flashed to `/dev/ttyACM0`, captured serial output.

```json
{"schema":"ri-esp32s3-lstm-bench-v1","firmware_variant":"p7_esp_nn_s3_dot_recurrent","board":"esp32s3","psram_size":8386151,"free_heap_start":290720,"free_psram_start":3650419,"weights_sha256":"709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61","model_profile":"mixed_lstm_safe","params":6337569,"compressed_bytes":4785276,"tokens_per_seed":16,"total_measured_tokens":48,"ms_total":22774,"ms_per_token_mean":271.27,"ms_per_token_p50":271,"ms_per_token_p95":272,"tokens_per_sec":3.6864,"op_breakdown_ms_per_token":{"embed":0.043,"quant":0.406,"lstm_wih":196.178,"lstm_whh":59.691,"activation":1.368,"fc":1.067},"output_by_seed":{"the esp32 sees ":"heasthe sathe sa","once upon a ":"herined sathe sa","sensor says ":"ammy saund thera"},"state_alloc":"internal","heap_after":290456,"psram_after":3650419,"passed":true,"blockers":[]}
```

## Comparison against p6

Earlier p6 receipt: 375.80 ms/token, 2.661 tok/s.

p7 receipt: 271.27 ms/token, 3.6864 tok/s.

Delta:

- Latency: 375.80 → 271.27 ms/token = **27.8% lower**
- Throughput: 2.661 → 3.686 tok/s = **1.39x faster**

## Why it did not hit 8.5 tok/s

The standalone ESP-NN probe measured:

- 512x512 SRAM-friendly FC: 12.57x faster
- 2048x512 PSRAM-heavy FC: 2.27x faster

The full LSTM differs:

1. `lstm_wih` remains the dominant cost: **196.18 ms/token**. In the `mixed_lstm_safe` model, the input matrices are int4, so they still use the scalar `dot_i4_i8_acc` path. ESP-NN dot only accelerates int8×int8.
2. `lstm_whh` improved but not by 12.6x: now **59.69 ms/token**. The recurrent weights are large and cloned to PSRAM because tensors over 8192 bytes go to PSRAM. The standalone 512x512 best case had better locality.
3. The direct dot function avoids FC stack overflow, but it also avoids the full FC wrapper's row batching. It is safe and simple, but not maximal.

## Claim boundary

Safe to claim:

- ESP-NN dot is now wired into the actual ESP32-S3 LSTM firmware.
- Hardware verified p7: **3.686 tok/s**, **271.27 ms/token**, **1.39x faster than p6**.
- The recurrent int8 dot path is accelerated without the ESP-NN FC dispatcher's stack-VLA crash.

Do not claim:

- 8.5 tok/s actual. That was a projection and did not materialize.
- Full ESP-NN LSTM integration. The int4 input matrices are still scalar.
- Correctness improvement. Output text changed because the quantized dot path changed arithmetic/performance details; quality still needs separate PPL/behavior validation.

## Next speed ROI

1. Convert or duplicate the input matrices into an ESP-NN-compatible int8 layout and rerun. This directly attacks the remaining 196 ms/token `lstm_wih` bottleneck.
2. Try keeping one layer's recurrent matrix in internal SRAM or tiled SRAM scratch if memory allows. The current p7 recurrent path is still PSRAM-bound.
3. Consider a smaller H384/H256 model. With ESP-NN, smaller matrices that fit SRAM will get much closer to the 12x speedup regime.
