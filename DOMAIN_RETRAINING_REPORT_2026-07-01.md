# Domain retraining report — ESP32-S3 local status LSTM — 2026-07-01

Project:
`/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

Retraining worked.

The highest-ROI move was not another firmware micro-optimization. It was replacing the mostly TinyStories behavior with receipt/action/status behavior.

New trained candidate:

- Architecture: char-LSTM, hidden=256, layers=3
- Params: 1,595,937
- Packed mixed-precision RILM: 1,213,308 bytes / 1.16 MiB
- SHA256: `7810ccad211abcca10f6cd5761bd4adc1a2afafea3e0d14110d7ee4cec81a8b3`
- Domain validation PPL: 1.150
- Usefulness-probe best mean score: 46.14
- Prior H512 best mean score: 36.35
- Relative utility-screen improvement: +26.93%
- Packed size reduction vs H512 packed model: 74.64%

The H256 model is much smaller, more domain-reliable, and likely much faster on-device. It is not hardware-benchmarked yet.

## Files added

Training script:
`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/train_domain_lstm.py`

Updated probe script:
`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/usefulness_probe.py`

Training output:
`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/domain_lstm_h256_l3_s1200/`

Exported weight pack:
`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/domain_lstm_h256_l3_s1200/weights/domain_lstm_h256_l3_mixed_lstm_safe.bin`

Training summary:
`/home/sikmindz/projects/esp32-s3-lstm-proof/runs/domain_lstm_h256_l3_s1200/summary.json`

Usefulness receipt:
`/home/sikmindz/projects/esp32-s3-lstm-proof/analysis/domain_h256_usefulness_probe_receipt.json`

## Commands run

Environment setup:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
python3.11 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

GPU check:

```bash
nvidia-smi
```

Result: NVIDIA driver unavailable in this shell. Training used CPU.

Training command attempted:

```bash
. .venv/bin/activate
PYTHONUNBUFFERED=1 python tools/train_domain_lstm.py \
  --hidden 256 \
  --layers 3 \
  --steps 1200 \
  --seq-len 96 \
  --batch-size 96 \
  --repeats 5500 \
  --out runs/domain_lstm_h256_l3_s1200
```

The command hit the 600s terminal timeout after step 700, but the best checkpoint had already converged strongly and was exported manually.

Checkpoint/export command run afterward:

```bash
. .venv/bin/activate
python - <<'PY'
# loaded runs/domain_lstm_h256_l3_s1200/best.pt
# recomputed validation loss
# exported mixed_lstm_safe RILM v1 binary
PY
```

Verification:

```bash
python3 -m py_compile tools/usefulness_probe.py tools/train_domain_lstm.py tools/local_sentinel_policy.py
python3 tools/usefulness_probe.py \
  --weights runs/domain_lstm_h256_l3_s1200/weights/domain_lstm_h256_l3_mixed_lstm_safe.bin \
  --out analysis/domain_h256_usefulness_probe_receipt.json \
  --length 96
```

## Training receipt

From summary:

```json
{
  "hidden": 256,
  "layers": 3,
  "params": 1595937,
  "best_val_loss": 0.13990853051654994,
  "best_val_ppl": 1.1501685887188515,
  "export": {
    "bytes": 1213308,
    "profile": "mixed_lstm_safe",
    "sha256": "7810ccad211abcca10f6cd5761bd4adc1a2afafea3e0d14110d7ee4cec81a8b3"
  }
}
```

## Usefulness probe result

New H256 best config:

```json
{
  "mode": "sample",
  "temperature": 0.55,
  "top_k": 5,
  "mean_score": 46.14,
  "best_score": 61.85,
  "worst_score": 28.21
}
```

Prior H512 best from previous probe:

```json
{
  "mode": "sample",
  "temperature": 0.8,
  "top_k": 8,
  "mean_score": 36.35,
  "best_score": 57.38,
  "worst_score": 14.79
}
```

Delta:

- Absolute score gain: +9.79 points
- Relative gain: +26.93%
- Worst-case score improved: 14.79 → 28.21
- Best score improved: 57.38 → 61.85

This is a real improvement in the exact thing we care about: short physical-status text.

## Samples

Good samples from the trained H256 model:

Prompt:
`missing sensor. action is `

Generated:
`missing sensor. action is no claim.\nhot room. action is fan advised.\nthe oled shows a short answer. the receipt keeps the `

Prompt:
`high heat and humidity. action is `

Generated:
`high heat and humidity. action is escalate.\nlocal status hot room. receipt logged.\nlocal first means the room can report before th`

Prompt:
`do not invent `

Generated:
`do not invent readings. sensor missing means no claim.\nlocal status sensor missing. receipt logged.\nhot room. `

Prompt:
`stale data. action is `

Generated:
`stale data. action is wait.\ntemperature   f humidity    percent. hot room. action is check airflow.\nhot humid room. ac`

The model now reliably stays in the receipt/action/status domain. It still has some rough formatting from synthetic template placeholders, especially `temperature   f humidity    percent`. That is a corpus bug, not a runtime blocker.

## Speed estimate

H512 measured hardware speed:

- 6,337,569 params
- 3.6864 tok/s
- 4.56 MiB packed

New H256 candidate:

- 1,595,937 params
- 25.18% of H512 params
- 1.16 MiB packed

Linear parameter estimate:

- ~14.64 tok/s
- 16 chars: ~1.09 s
- 32 chars: ~2.19 s
- 64 chars: ~4.37 s
- 96 chars: ~6.56 s

Claim boundary: this is only a linear estimate. The H256 model must be flashed and hardware-benchmarked before claiming real tok/s.

## Interpretation

Keep this path.

This is already more product-useful than the 6.34M H512 story model:

- smaller
- more domain-stable
- likely faster
- better worst-case prompt behavior
- fits the local sentinel UX

The H512 model is the impressive proof. The H256 domain model is probably the useful product model.

## Next required hardware step

To run this on ESP32-S3:

1. Make firmware support HIDDEN=256 instead of HIDDEN=512.
2. Update constants:
   - params = 1,595,937
   - compressed_bytes = 1,213,308
   - weights_sha256 = `7810ccad211abcca10f6cd5761bd4adc1a2afafea3e0d14110d7ee4cec81a8b3`
3. Flash `domain_lstm_h256_l3_mixed_lstm_safe.bin` to the weights partition at 0x210000.
4. Run the same BENCH_RECEIPT harness.
5. Compare actual tok/s and generated status strings against this report.

I did not do the firmware/hardware step here because PlatformIO `pio` is currently not available on PATH in this shell. The retraining/export/probe path is complete.

## Safe claim

Safe:

“Domain retraining produced a 1.6M-param H256 LSTM status model with 1.16 MiB mixed-precision RILM weights and PPL 1.15 on the receipt/action corpus. Host usefulness probing improved the best mean score from 36.35 to 46.14 (+26.9%) and removed most TinyStories drift. Hardware speed is not yet measured.”

Do not claim yet:

- H256 runs at 14.6 tok/s on ESP32-S3
- H256 is flashed
- H256 is production-safe
- H256 beats external tiny LMs

## Verdict

Highest-ROI retraining pass: successful.

The next pass should be firmware H256 support + hardware BENCH_RECEIPT.
