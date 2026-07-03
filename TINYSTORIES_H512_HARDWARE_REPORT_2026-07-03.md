# TinyStories H512 ESP32-S3 hardware report — 2026-07-03

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

TinyStories-class story behavior is now running on real ESP32-S3 hardware.

This is not TinyStories-33M. It is the previously trained 6.34M-parameter char-LSTM story/edge-mixture model, deployed as a packed RILM v1 weight file and executed on the ESP32-S3 local LSTM runtime.

## Hardware / firmware

- Board: ESP32-S3 Freenove WROOM N8R8, 8MB PSRAM
- USB port: `/dev/ttyACM0`
- Firmware env: `esp32s3_tinystories_h512`
- Firmware variant: `tinystories_h512_p7_i8_espnn`
- Hidden size: 512
- Layers: 3
- Params: 6,337,569
- Weight pack: `weights_h512_p7_backup_709ff8.bin`
- Weight bytes: 4,785,276
- Weight SHA256: `709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61`
- Flash partition: `weights` at `0x210000`, size `0x500000`

## Commands run

```bash
pio run -e esp32s3_tinystories_h512

pio run -e esp32s3_tinystories_h512 -t upload --upload-port /dev/ttyACM0

python3 ~/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x210000 weights_h512_p7_backup_709ff8.bin

python3 tools/run_bench.py \
  --port /dev/ttyACM0 \
  --variant tinystories_h512_p7_i8_espnn \
  --repeat 1 \
  --timeout 120 \
  --out-dir benchmarks/tinystories-h512-2026-07-03
```

## Boot / load receipt

```text
ESP32-S3 LSTM boot tinystories_h512_p7_i8_espnn
weights partition addr=0x210000 size=5242880
RILM version=1 tensors=15
payloads cloned bytes=4784772 free_heap=300208 free_psram=3650419
int4 recurrent lstm.weight_hh_l0 old=1048576 new=524288 scale=0.11046
int4 recurrent lstm.weight_hh_l1 old=1048576 new=524288 scale=0.141142
int4 recurrent lstm.weight_hh_l2 old=1048576 new=524288 scale=0.127179
int4 recurrent converted=3 saved=1572864 free_psram=2077507
state allocated free_heap=270900 free_psram=2077507
MODEL_READY profile=tinystories_h512_mixed_lstm_safe params=6337569 hidden=512 layers=3
```

## Benchmark receipt

```json
{
  "firmware_variant": "tinystories_h512_p7_i8_espnn",
  "weights_sha256": "709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61",
  "model_profile": "tinystories_h512_mixed_lstm_safe",
  "params": 6337569,
  "compressed_bytes": 4785276,
  "total_measured_tokens": 48,
  "ms_per_token_mean": 86.04,
  "tokens_per_sec": 11.6223,
  "output_by_seed": {
    "once upon a ": "time there was a",
    "the little girl ": "was so happy. sh",
    "the dragon was ": "so happy.\nendoft"
  },
  "stopped_output_by_seed": {
    "once upon a ": "time there was a tiny model that lived on a sens",
    "the little girl ": "was so happy.",
    "the boy saw ": "a big smile.",
    "the dog was ": "so happy.",
    "the cat said ": "yes.",
    "the dragon was ": "so happy.",
    "the toy was ": "so happy.",
    "the bird flew ": "away."
  },
  "passed": true,
  "blockers": []
}
```

Full receipt files:

- `benchmarks/tinystories-h512-2026-07-03/tinystories_h512_p7_i8_espnn-run-001.raw.log`
- `benchmarks/tinystories-h512-2026-07-03/tinystories_h512_p7_i8_espnn-run-001.json`
- `benchmarks/tinystories-h512-2026-07-03/tinystories_h512_p7_i8_espnn-summary.json`

## Interactive prompt receipts

```text
PROMPT:once upon a 
S3_LANGUAGE_RECEIPT output="time there was a tiny model that lived on a sens" generated_chars=48 elapsed_ms=4543 chars_per_sec=10.5657

PROMPT:the little girl 
S3_LANGUAGE_RECEIPT output="was so happy." generated_chars=13 elapsed_ms=2120 chars_per_sec=6.1321
```

## Fixes required to make this work

1. Firmware constants were made build-flag configurable (`RI_HIDDEN`, `RI_MODEL_PROFILE`, `RI_WEIGHTS_SHA256`, etc.) so H256 domain and H512 TinyStories-class models can share the same runtime.
2. Added PlatformIO env `esp32s3_tinystories_h512`.
3. `convert_wih_to_int4()` no longer falsely requires `LAYERS * 2` conversions. The H512 mixed profile already stores `weight_ih` as int4 and only needs the three recurrent `weight_hh` tensors converted at boot.
4. TinyStories-mode benchmark seeds now use story prompts instead of sensor/status prompts.

## Claim boundary

Safe to claim:

- A 6.34M-parameter TinyStories-class char-LSTM is running locally on ESP32-S3 hardware.
- The deployed H512 story model emits story-like continuations from TinyStories-style prompts.
- The verified single-run hardware speed for this firmware/weight pair is 11.6223 chars/s, 86.04 ms/char.

Do not claim:

- TinyStories-33M is running.
- This is a full-quality TinyStories benchmark.
- The model is a general assistant.
- The three-board cluster distributes the recurrent H512 story model.
