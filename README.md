# ESP32-S3 Compressed LSTM Hardware Proof

Path: `/home/sikmindz/projects/esp32-s3-lstm-proof`

Purpose: prove the GTX-trained ESP32-S3 LSTM can be quantized, stored as a flash data partition, memory-mapped at runtime, parsed on-device, and used for real token generation on ESP32-S3 hardware.

## Hardware receipt

ESP32-S3:
- Port: `/dev/ttyACM0`
- Chip: ESP32-S3 QFN56 rev 0.2
- MAC: `94:a9:90:d2:41:f4`
- PSRAM: 8MB detected and enabled
- Flash target profile: Freenove ESP32-S3 WROOM N8R8

Regular ESP32 sensor node:
- Port: `/dev/ttyUSB0`
- Chip: ESP32-D0WDQ6 rev 1.0
- MAC: `c8:c9:a3:d6:2a:18`
- IP: `192.168.50.244`
- Sensor/OLED/WiFi working

## Model receipt

Source quantized profile:
`/home/jstevenson/projects/esp32-max-lm-training/runs/esp32s3_max_lstm_h512_l3/quantized/esp32s3_max_lstm_mixed_lstm_safe.bin`

Local copy:
`/home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin`

SHA256:
`709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61`

Model:
- 512 hidden
- 3 LSTM layers
- 6,337,569 params
- selected profile: `mixed_lstm_safe`
- compressed binary size: 4,785,276 bytes / 4.56 MiB
- measured dequantized PyTorch PPL: 2.7831
- delta vs fp32 PPL: +0.1864

## Flash layout

Custom partition table:

- app0: `0x10000`, size `0x200000`
- weights: `0x210000`, size `0x500000`
- spiffs: `0x710000`, size `0x0F0000`

Weights are flashed directly:

```bash
/home/sikmindz/projects/esp32-sensor-hub/.venv/bin/python \
  /home/sikmindz/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 460800 \
  write_flash 0x210000 /home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin
```

## Build and flash

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
. /home/sikmindz/projects/esp32-sensor-hub/.venv/bin/activate
pio run -t upload --upload-port /dev/ttyACM0
```

## Hardware run receipt

Observed serial output included:

```text
ESP32-S3 compressed LSTM proof boot
free_heap=371332 free_psram=8386035 psram_size=8386279
weights partition addr=0x210000 size=5242880
RILM version=1 tensors=15
MODEL_READY selected_profile=mixed_lstm_safe params=6337569 compressed_bytes=4785276
SEED: the esp32 sees
GEN: the esp32 sees arm air 
PROOF_DONE
```

Observed token latency:

```text
a [tok_ms=1636]
r [tok_ms=1636]
m [tok_ms=1636]
  [tok_ms=1636]
a [tok_ms=1636]
i [tok_ms=1636]
r [tok_ms=1636]
  [tok_ms=1636]
```

Approx throughput: 0.61 token/sec with scalar C++ matmul, flash-mapped mixed int4/int8 weights, and debug layer prints enabled.

## Claim boundary

This proves on-device execution of the compressed mixed-profile LSTM on ESP32-S3 hardware.

It does not yet prove:
- optimized speed
- ESP-NN/SIMD acceleration
- OLED display integration for the S3 LSTM proof
- sensor-conditioned prompt generation in the S3 local model
- Home Assistant integration

First optimization pass completed:
- added machine-readable BENCH_RECEIPT JSON harness
- removed per-layer debug prints
- cached tensor pointers once, not lookup by name per step
- moved hot LSTM state buffers to internal SRAM
- added dtype-specific int8/int4 dot-product fast paths
- benchmarked 3 repeated hardware runs

Speed receipt:
- baseline proof: ~1636 ms/token, ~0.61 tok/sec
- p0_p4_cached_internal_dtype_fastpath: 421.94 ms/token mean, 2.37 tok/sec mean, 3.88x
- p5_fixedpoint_lut: 397.06 ms/token mean, 2.52 tok/sec mean, 4.12x
- final p6_psram_weights_fixedpoint_lut: 375.80 ms/token mean, 2.66 tok/sec mean, 4.35x
- final latency reduction: 77.03%

Benchmark report:
`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/SPEED_REPORT_2026-06-30.md`

Next optimization path:
- standalone ESP-NN / Xtensa SIMD fully-connected probe before model integration
- RILM v2 row/alignment metadata for speed-friendly memory layout
- train/export smaller H384/H256 variants and benchmark speed/quality tradeoff
- connect regular ESP32 sensor readings into the S3 local model seed/context
