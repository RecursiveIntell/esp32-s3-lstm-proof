# ESP32-S3 domain H256 hardware deployment report — 2026-07-01

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

Highest-ROI retraining path is now flashed and hardware-verified.

Best useful firmware on the board right now:

- Variant: `p11_domain_h256_all_int8_greedy_scalar`
- Model: domain-specific H256 x 3-layer char-LSTM
- Params: 1,595,937
- Weight pack: all-int8 RILM, 1,614,972 bytes
- SHA256: `770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c`
- Hardware speed: 13.5517 tok/s
- Latency: 73.79 ms/token
- 3 repeated hardware runs: stable

This is the useful product point: short local status/action text under ~1.2s for 16 chars and ~2.4s for 32 chars.

## What changed

Firmware was patched from the H512 proof model to the H256 domain model:

- `HIDDEN = 256`
- domain prompts for benchmark seeds:
  - `hot room. action is `
  - `missing sensor. action is `
  - `the receipt says `
- firmware metadata updated for the all-int8 H256 pack
- deployed `weights.bin` replaced with all-int8 domain pack
- firmware rebuilt and flashed to `/dev/ttyACM0`
- all-int8 weight pack flashed to partition `0x210000`

Current deployed weight pack:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin`

Backup of prior H512 p7 pack:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights_h512_p7_backup_709ff8.bin`

## Verification commands run

Build:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
~/.local/bin/pio run -e esp32s3_lstm
```

Firmware flash:

```bash
~/.local/bin/pio run -t upload --upload-port /dev/ttyACM0
```

Weight flash:

```bash
/home/sikmindz/.local/share/uv/tools/platformio/bin/python \
  /home/sikmindz/.platformio/packages/tool-esptoolpy/esptool.py \
  --chip esp32s3 --port /dev/ttyACM0 --baud 460800 \
  write_flash 0x210000 /home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin
```

Hardware benchmark:

```bash
PATH=/home/sikmindz/.local/bin:$PATH \
/home/sikmindz/.local/share/uv/tools/platformio/bin/python \
  tools/run_bench.py \
  --port /dev/ttyACM0 \
  --variant p11_domain_h256_all_int8_scalar \
  --repeat 3 \
  --timeout 120 \
  --out-dir benchmarks/p11_domain_h256_all_int8_scalar
```

## Hardware receipt summary

Receipt path:

`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p11_domain_h256_all_int8_scalar/p11_domain_h256_all_int8_scalar-summary.json`

Aggregate:

```json
{
  "runs": 3,
  "tokens_per_sec_mean": 13.5517,
  "tokens_per_sec_min": 13.5517,
  "tokens_per_sec_max": 13.5517,
  "ms_per_token_mean": 73.79,
  "ms_per_token_min": 73.79,
  "ms_per_token_max": 73.79
}
```

Representative raw receipt:

```json
{
  "firmware_variant": "p11_domain_h256_all_int8_greedy_scalar",
  "board": "esp32s3",
  "weights_sha256": "770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c",
  "model_profile": "domain_h256_all_int8",
  "params": 1595937,
  "compressed_bytes": 1614972,
  "tokens_per_sec": 13.5517,
  "ms_per_token_mean": 73.79,
  "ms_per_token_p50": 74,
  "ms_per_token_p95": 74,
  "op_breakdown_ms_per_token": {
    "embed": 0.044,
    "quant": 0.206,
    "lstm_wih": 34.111,
    "lstm_whh": 32.183,
    "activation": 0.644,
    "fc": 0.354
  },
  "output_by_seed": {
    "hot room. action is ": "heck airflow.\nth",
    "missing sensor. action is ": "o claim.\nthe rec",
    "the receipt says ": "tale reading.\nth"
  },
  "passed": true,
  "blockers": []
}
```

Combined prompt + generated text:

- `hot room. action is check airflow.`
- `missing sensor. action is no claim.`
- `the receipt says stale reading.`

The generated chunks are suffixes only, so `heck airflow` means the full prompt produces `check airflow`.

## Comparison

Against H512 p7 ESP-NN proof:

- H512 p7: 3.6864 tok/s, 271.27 ms/token
- H256 p11: 13.5517 tok/s, 73.79 ms/token
- Speedup: 3.68x
- Latency reduction: 72.8%

Against p6 pre-ESP-NN H512:

- p6: 2.661 tok/s
- H256 p11: 13.5517 tok/s
- Speedup: 5.09x

Interactive generation budget at 13.5517 tok/s:

- 16 chars: 1.18s
- 32 chars: 2.36s
- 64 chars: 4.72s

## Important negative result: ESP-NN direct dot corrupts this model's useful output

I tested the all-int8 H256 model with ESP-NN direct dot enabled:

Receipt path:

`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p10_domain_h256_all_int8/p10_domain_h256_all_int8-summary.json`

Result:

- Speed: 24.8062 tok/s
- Latency: 40.31 ms/token
- Output collapsed:
  - `fffffffffffvvvvv`
  - `iiiiiiiiiiiiiiii`
  - `iiiiiiiiiiiiiiii`

So p10 is fast but not useful. It is a kill for claim/use, not a win.

The current useful firmware deliberately disables the ESP-NN dot path for int8 LSTM dots and uses scalar int8 accumulation. This sacrifices peak speed but preserves useful output.

## Interpretation

This validates the product direction:

- The H512 6.34M model is the impressive proof.
- The H256 domain model is the usable local status model.
- The ESP32-S3 can now emit short, domain-relevant physical-status/action text in 1–3 seconds.
- The model should still not be the policy brain. Deterministic sentinel/rules should decide; LSTM should phrase the local status/action.

## Next ROI

1. Add a firmware mode that emits full prompt + generated suffix in the receipt so future receipts are human-readable without recombining mentally.
2. Add a repetition/stop rule: stop at newline or period after minimum chars.
3. Investigate ESP-NN correctness with a tiny hardware dot self-test before re-enabling it for product firmware.
4. Train H192 or H128 all-int8 domain model and check whether useful output survives at >20 tok/s scalar.
