# Local sentinel policy receipt — 2026-07-01

Path:

`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/local_sentinel_policy.py`

Purpose:

Make the current ESP32-S3 compute useful by separating decision from generation.

The deterministic sentinel decides and emits a receipt. The 6.34M LSTM should only generate short local phrasing from the suggested prompt, and only within the latency budget.

Verification command run:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
python3 -m py_compile tools/usefulness_probe.py tools/local_sentinel_policy.py
python3 tools/local_sentinel_policy.py --temp-c 29.4 --humidity 72 --age-s 3
```

Observed receipt:

```json
{
  "age_s": 3.0,
  "ai_route": true,
  "blockers": [],
  "confidence": 0.9,
  "decision": "escalate_heat_humidity",
  "device_id": "esp32s3-lstm-node",
  "humidity_pct": 72.0,
  "local_status": "hot humid 85f 72%.",
  "route_reason": "temperature_and_humidity_out_of_range",
  "schema": "ri_esp32_local_sentinel_v1",
  "suggested_lstm_chars": 16,
  "suggested_lstm_prompt": "high heat and humidity. action is ",
  "temp_c": 29.4
}
```

Interpretation:

- Deterministic local receipt is immediate and reliable.
- LSTM generation budget is intentionally capped to 16 chars for escalation states.
- `ai_route=true` means the larger Gemma/Ollama tier should handle rich explanation.
- This avoids using the LSTM as an unsafe free-form policy brain.

Build note:

I did not rebuild firmware in this pass because `pio`/PlatformIO is currently not on PATH in this shell. The added artifacts are host-side Python tools and markdown receipts only; they do not change firmware behavior.
