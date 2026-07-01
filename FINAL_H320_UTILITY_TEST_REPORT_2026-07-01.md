# Final H320 utility test report — 2026-07-01

## Keep / kill

KEEP the curated H320 model as the quality/product artifact.

It is good enough for the intended role: a local ESP32-S3 physical-sentinel language layer that phrases short status/action text after deterministic policy chooses the route.

Do not use it as a free-form policy brain.

## Final deployed variant

`p15_curated_h320_stopped_utility`

Model:

- hidden: 320
- layers: 3
- params: 2,486,433
- profile: all-int8 RILM
- weight bytes: 2,510,076
- SHA256: `fb042c0aa011475e0a31d2c5d271dde57c504aa1bdcdb02ecd3ce0010ebf2b7a`

Current deployed weights:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin`

## Why p15 exists

p14 proved the H320 model was fast and useful, but the fixed-length benchmark output continued into the next training template after the useful phrase.

p15 adds firmware-side stopped utility output:

- emit generated chars until period or newline
- report this in `stopped_output_by_seed`
- keep the fixed-token BENCH_RECEIPT speed path for comparable timing

This tests the actual product behavior: short local status/action phrases.

## Host audit

From:

`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/curated_status_h320_l3/summary.json`

- validation PPL: 1.219793
- test PPL: 1.281179
- policy prefix accuracy: 0.80
- policy contains accuracy: 0.80
- host usefulness best config: greedy
- host usefulness mean score: 60.14

Known failures on held-out/free-form-ish prompts:

- `very hot room. action is ` generated `check airflow.` instead of `escalate.`
- `old reading. action is ` generated `no claim.` instead of `wait for fresh data.`
- `unsafe evidence. action is ` generated `ask gateway.` instead of `no claim.`
- `if sensor is missing ` starts with `sensor. action is no claim.` rather than direct `say no claim.`
- generic `the receipt says ` tends to choose a concrete receipt like heat warning
- `the gateway should ` generated `confidence.` instead of `verify the claim.`

Interpretation: acceptable for canonical policy-fed prompts; not acceptable as standalone policy inference.

## Firmware changes

Changed:

`/home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp`

Added:

- `UTILITY_SEEDS`
- `generate_stopped(...)`
- `stopped_output_by_seed` in `BENCH_RECEIPT`

Also kept earlier p14 fix:

- emit predicted token before advancing, preventing first-character drop.

## Build and flash verification

Build:

```bash
/home/sikmindz/.local/bin/pio run -e esp32s3_lstm
```

Result:

- success
- RAM: 22,592 / 327,680 bytes = 6.9%
- Flash app: 277,613 / 2,097,152 bytes = 13.2%

Upload:

```bash
/home/sikmindz/.local/bin/pio run -t upload --upload-port /dev/ttyACM0
```

Result:

- success after one transient serial retry
- image hash verified

## Hardware receipt

Combined receipt:

`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p15_curated_h320_stopped_utility/p15_curated_h320_stopped_utility-summary.json`

Three successful hardware runs:

```json
{
  "runs": 3,
  "tokens_per_sec_mean": 17.229,
  "tokens_per_sec_min": 17.229,
  "tokens_per_sec_max": 17.229,
  "ms_per_token_mean": 58.04,
  "ms_per_token_min": 58.04,
  "ms_per_token_max": 58.04
}
```

Stopped utility outputs from hardware:

```json
{
  "hot room. action is ": "check airflow.",
  "missing sensor. action is ": "no claim.",
  "stale data. action is ": "wait for fresh data.",
  "high heat and humidity. action is ": "escalate.",
  "humid room. action is ": "ventilate.",
  "normal room. action is ": "log receipt.",
  "safe action is ": "no claim without evidence.",
  "local first means ": "decide before cloud."
}
```

Representative op breakdown, ms/token:

```json
{
  "embed": 0.054,
  "quant": 0.257,
  "lstm_wih": 24.49,
  "lstm_whh": 24.429,
  "activation": 0.799,
  "fc": 0.227
}
```

## Comparison

H256 p12:

- 25.0653 tok/s
- 39.90 ms/token
- fastest useful model

H320 p15:

- 17.229 tok/s
- 58.04 ms/token
- better curated/product text
- stopped utility output is clean

H512 p7:

- 3.6864 tok/s
- 271.27 ms/token
- impressive proof, weaker utility

H320 p15 vs H512 p7:

- 4.67x faster
- 78.6% lower latency

Interactive budget for H320 p15:

- 16 chars: 0.93s
- 32 chars: 1.86s
- 64 chars: 3.72s

## Final judgment

This model is good for the target product shape.

Use p15 for demos where the point is useful local language/status behavior.
Use p12/H256 where the point is maximum speed.

Safe wording:

“ESP32-S3 runs a 2.49M-param all-int8 curated local status/action model at 17.2 tok/s and emits stopped, useful receipt/action phrases on-device. It is not a chatbot; it is a local physical-sentinel language layer.”

Do not claim:

- general LLM capability
- autonomous policy correctness
- state of the art
- production readiness

## Next, only if worth polishing

1. Feed real sensor-policy prompts from the ESP32 sensor hub into S3.
2. Add OLED/display demo output.
3. Add comparison video: H256 speed mode vs H320 quality mode.
4. If retraining again, train stop punctuation even harder and reduce long receipt continuation templates.
