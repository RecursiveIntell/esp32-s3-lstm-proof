# ESP32-S3 curated H320 status model report — 2026-07-01

## Bottom line

Created, trained, exported, flashed, and hardware-verified a higher-capacity curated domain model for the ESP32-S3 local sentinel path.

Best deployed variant:

`p14_curated_h320_all_int8_emit_first`

Hardware result over 3 repeated ESP32-S3 runs:

- 17.2166 tok/s
- 58.08 ms/token
- 2,486,433 params
- 2,510,076-byte all-int8 RILM weight pack
- SHA256: `fb042c0aa011475e0a31d2c5d271dde57c504aa1bdcdb02ecd3ce0010ebf2b7a`

This is slower than the H256 p12 speed king, but materially better on host usefulness probing and still comfortably interactive.

## What was created

Training script:

`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/train_curated_status_lstm.py`

Finalize/export script:

`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/finalize_curated_status_run.py`

MSI/GTX run mirror:

`/home/jstevenson/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3`

Local run copy:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3`

Deployed local weight file:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin`

Previous p12/H256 backup:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights_p12_h256_backup_770ed901.bin`

## Training data curation

Corpus schema:

`ri_esp32_curated_status_corpus_v1`

Intent:

Short receipt/action/status text for deterministic ESP32-S3 sentinel policy; not chatbot/story text.

The corpus was curated around:

- canonical prompt -> continuation pairs
- deterministic local route reasons
- negative controls for missing/stale/bad evidence
- receipt/action/status phrasing
- OLED-sized local text
- held-out prompts for generalization checks

Corpus manifest:

- policy cases: 30
- routes: 8
- train lines: 83,734
- validation lines: 6,282
- test lines: 3,014
- held-out prompts:
  - `old reading. action is `
  - `the gateway should `
  - `unsafe evidence. action is `
  - `very hot room. action is `
- test prompts:
  - `if sensor is missing `
  - `the receipt says stale `
  - `unknown room. action is `

Important boundary: this is not generic language training. It intentionally biases toward short useful status/action text.

## MSI/GTX 1070 training receipt

MSI GPU verified before training:

- host: `msi`
- GPU: NVIDIA GeForce GTX 1070
- VRAM: 8192 MiB
- PyTorch: 2.10.0+cu126
- CUDA available: true

Training command:

```bash
ssh msi 'cd /home/jstevenson/projects/esp32-s3-lstm-proof && \
  PYTHONUNBUFFERED=1 python3 tools/train_curated_status_lstm.py \
  --hidden 320 \
  --layers 3 \
  --steps 3200 \
  --seq-len 96 \
  --batch-size 128 \
  --repeats 9000 \
  --out runs/curated_status_h320_l3 \
  --profiles all_int8,mixed_lstm_safe \
  2>&1 | tee runs/curated_status_h320_l3/train.log'
```

Thermal/power note:

Training was stopped early after the GTX 1070 reached 88C. This was intentional because the MSI has known PSU/battery reliability risk. The best checkpoint had already been saved and was finalized/exported from `best.pt`.

Finalized checkpoint:

- params: 2,486,433
- best validation PPL: 1.219793
- finalized test PPL: 1.281179
- policy prefix accuracy across 30 curated prompts: 0.80
- policy contains accuracy: 0.80

Summary:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3/summary.json`

Training log:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3/training_log.json`

## Exported weights

All-int8 selected for deployment:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3/weights/curated_status_h320_l3_all_int8.bin`

- bytes: 2,510,076
- payload bytes: 2,509,572
- SHA256: `fb042c0aa011475e0a31d2c5d271dde57c504aa1bdcdb02ecd3ce0010ebf2b7a`

Mixed profile also exported:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3/weights/curated_status_h320_l3_mixed_lstm_safe.bin`

- bytes: 1,885,116
- SHA256: `2b7357f7423672bf233961fabcc62cd0eabdc5685d8202ae13fe1fabf728168c`

All-int8 was used because the fixed ESP-NN path accelerates int8 dot products and the weight pack still fits easily in the 5 MiB weights partition.

## Host usefulness probe

Probe receipt:

`/home/sikmindz/projects/esp32-s3-lstm-proof/analysis/curated_h320_all_int8_usefulness_probe.json`

Best decode config:

- greedy
- mean score: 60.14
- best score: 65.49
- worst score: 41.67

Compared with prior H256 probe mean score 46.14:

- absolute improvement: +14.00 points
- relative improvement: +30.34%

Interpretation:

H320 is the better usefulness/product model. H256 p12 is the speed king.

## Firmware changes

`src/main.cpp` was updated for H320:

- `HIDDEN = 320`
- firmware variant: `p14_curated_h320_all_int8_emit_first`
- model profile: `curated_status_h320_all_int8`
- params: `2486433`
- compressed bytes: `2510076`
- SHA: `fb042c0aa011475e0a31d2c5d271dde57c504aa1bdcdb02ecd3ce0010ebf2b7a`

Also fixed an output bug in the benchmark generation loop:

Older loop advanced before emitting the first predicted token. This dropped the first generated character:

- `check airflow` became `heck airflow`
- `no claim` became `o claim`

p14 emits the already-predicted token first, then advances the model.

## Build/flash verification

Firmware build:

```bash
/home/sikmindz/.local/bin/pio run -e esp32s3_lstm
```

Result:

- success
- RAM: 22,592 / 327,680 bytes = 6.9%
- Flash app: 276,961 / 2,097,152 bytes = 13.2%

Weights flash:

```bash
/home/sikmindz/projects/esp32-sensor-hub/.venv/bin/python \
  /home/sikmindz/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 230400 \
  write_flash 0x210000 weights.bin
```

Result:

- wrote 2,510,076 bytes
- hash verified

Firmware upload:

```bash
/home/sikmindz/.local/bin/pio run -t upload --upload-port /dev/ttyACM0
```

Result:

- success
- image hash verified

## Hardware BENCH_RECEIPT

Receipt summary:

`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p14_curated_h320_all_int8_emit_first/p14_curated_h320_all_int8_emit_first-summary.json`

Aggregate over 3 repeated runs:

```json
{
  "runs": 3,
  "tokens_per_sec_mean": 17.2166,
  "tokens_per_sec_min": 17.2166,
  "tokens_per_sec_max": 17.2166,
  "ms_per_token_mean": 58.08,
  "ms_per_token_min": 58.08,
  "ms_per_token_max": 58.08
}
```

Representative output:

```json
{
  "hot room. action is ": "check airflow.\nr",
  "missing sensor. action is ": "no claim.\nreadin",
  "the receipt says ": "heat warning. ac"
}
```

Representative op breakdown, ms/token:

```json
{
  "embed": 0.054,
  "quant": 0.256,
  "lstm_wih": 24.503,
  "lstm_whh": 24.495,
  "activation": 0.797,
  "fc": 0.228
}
```

Memory after run:

- heap after: 318,104 bytes
- PSRAM after: 5,907,187 bytes

## Comparison table

| Variant | Params | Weight bytes | tok/s | ms/token | Usefulness |
|---|---:|---:|---:|---:|---|
| H512 p7 | 6.34M | 4.79M | 3.6864 | 271.27 | impressive proof, weaker utility |
| H256 p12 | 1.60M | 1.61M | 25.0653 | 39.90 | fastest useful sentinel |
| H320 p14 | 2.49M | 2.51M | 17.2166 | 58.08 | better curated/product text |

H320 p14 vs H512 p7:

- 4.67x faster
- 78.59% lower latency

H320 p14 vs H256 p12:

- 68.69% of H256 speed
- 45.56% higher latency
- 1.56x params
- 1.55x weight bytes
- +30.34% host usefulness mean score vs H256 probe

Interactive budget for H320 p14:

- 16 chars: 0.93s
- 32 chars: 1.86s
- 64 chars: 3.72s

## Interpretation

Keep both models:

- H256 p12: default if speed/latency matters most.
- H320 p14: default if output quality/intent matters most.

The H320 p14 result is the better demo/product artifact because it preserves sub-2-second 32-character output while producing cleaner domain behavior.

Safe public claim:

“Curated H320 domain retraining produced a 2.49M-param all-int8 ESP32-S3 local status/action model running at 17.2 tok/s with useful short receipt/action generations. It is not a chatbot; it is a local physical-sentinel language layer.”

Do not claim:

- general LLM capability
- SOTA
- unseen superiority over all ESP32 tiny-LM demos
- policy correctness without deterministic rules
- production readiness

## Next high-ROI work

1. Add stop-at-newline/period generation in firmware so it stops after the useful phrase instead of continuing into receipt-template text.
2. Add a 4-prompt hardware utility receipt, not just 3 prompts.
3. Train a second H320 pass with explicit stop-token pressure and fewer long receipt templates.
4. Compare H256 p12 vs H320 p14 in the same demo UI: speed mode vs quality mode.
