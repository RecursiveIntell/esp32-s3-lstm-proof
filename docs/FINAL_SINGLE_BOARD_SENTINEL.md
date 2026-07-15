# ESP32-S3 Final Single-Board Sentinel Firmware

**Variant:** `final_sentinel_h256_p22`  
**Model:** H256×3 char-LSTM, 1,595,937 params, all-int8 RILM  
**Weights SHA-256:** `770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c`  
**Performance (hardware-verified):** 39.52 chars/s (≈9.88 BPE tok/s) at 240 MHz, 8 MB PSRAM  

---

## Architecture Overview

This firmware transforms an ESP32-S3 into an **offline-first physical-AI sentinel**: a self-contained device that

1. Accepts sensor readings via serial command (`SENSOR:<temp_c>,<humidity_pct>,<age_s>`)
2. Runs a **deterministic local policy** (temperature/humidity/staleness thresholds) to choose a canonical action
3. Optionally generates a **bounded local-language phrase** (≤16 chars) from the on-device H256 char-LSTM
4. Emits a **cryptographically-bound receipt** (`S3_SENTINEL_RECEIPT`) with full provenance
5. Exposes **read-only HTTP health/status** endpoints when WiFi credentials are supplied at build time

No external connectivity is required for core operation. WiFi, OTA, and HTTP are **opt-in, build-time features** — the device boots and runs inference with zero network dependencies.

---

## Token Convention (Critical)

This model generates **one character per inference step**. Standard LLM benchmarks use BPE/WordPiece tokens.

- **Character tokens/sec (chars/s):** 39.52 (hardware mean, 3 runs)  
- **Estimated BPE-equivalent tok/s:** 39.52 / 4.0 ≈ **9.88** (using 4 chars/BPE-token)  
- **Domain-specific status text:** averages ~4.5 chars/token → **~8.8 BPE tok/s**

**Always report both numbers.** The 9.88 BPE tok/s figure is an *estimate* for cross-benchmark comparison; the hardware receipt contains the ground-truth 39.52 chars/s.

---

## Model Artifact Selection & Flashing

The firmware expects the H256 all-int8 RILM artifact at the `weights` partition (0x210000, 5 MB). The exact artifact is:

```
File: weights_p12_h256_backup_770ed901.bin
Size: 1,614,972 bytes
SHA-256: 770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c
```

**Flash command:**
```bash
python3 -m esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
  write_flash 0x210000 weights_p12_h256_backup_770ed901.bin
```

The firmware validates the SHA-256 of the **parsed RILM payload** (not the raw partition) on boot and reports:
```
S3_SENTINEL_MODEL_HASH verified=1 bytes=1614972 expected=770ed901... actual=770ed901...
```

If the hash mismatches, `model_hash_verified=false` appears in receipts — the device still runs but output provenance is flagged.

---

## Build

**PlatformIO environment:** `esp32s3_final_sentinel_h256`

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
pio run -e esp32s3_final_sentinel_h256
pio run -e esp32s3_final_sentinel_h256 -t upload --upload-port /dev/ttyACM0
```

### Build-time Configuration (Optional WiFi/OTA/HTTP)

| Macro | Default | Purpose |
|-------|---------|---------|
| `RI_FINAL_WIFI_SSID` | `""` (disabled) | WiFi STA SSID |
| `RI_FINAL_WIFI_PASSPHRASE` | `""` (disabled) | WiFi password |
| `RI_FINAL_OTA_PASSWORD` | `""` (disabled) | ArduinoOTA password |

**Both** `RI_FINAL_WIFI_SSID` **and** `RI_FINAL_WIFI_PASSPHRASE` must be non-empty to enable networking. OTA is enabled only when `RI_FINAL_OTA_PASSWORD` is also set.

Example with networking:
```bash
pio run -e esp32s3_final_sentinel_h256 \
  -D RI_FINAL_WIFI_SSID=\"MyWiFi\" \
  -D RI_FINAL_WIFI_PASSPHRASE=\"secret\" \
  -D RI_FINAL_OTA_PASSWORD=\"ota-secret\"
```

---

## Serial Command Protocol

All commands and responses are newline-terminated. JSON fields use `\` escaping for `\`, `"`, and newline.

### `SENSOR:<temp_c>,<humidity_pct>,<age_s>`
Inject a sensor reading. Values must be valid floats; `age_s` is unsigned seconds since measurement.

```text
SENSOR:29.4,72.0,3
```
Response:
```json
S3_SENTINEL_SENSOR {"valid":true,"temp_c":29.40,"humidity_pct":72.00,"age_ms":3000}
```

Invalid reading example:
```text
SENSOR:nan,nan,999
```
Response:
```json
S3_SENTINEL_SENSOR {"valid":false,"temp_c":nan,"humidity_pct":nan,"age_ms":0}
```

### `STATUS`
Print authoritative device status receipt.

```json
S3_SENTINEL_STATUS {"schema":"ri_esp32s3_sentinel_status_v1","firmware_variant":"final_sentinel_h256_p22","model_profile":"domain_h256_all_int8","params":1595937,"weights_sha256":"770ed901...","model_hash_verified":true,"reading_valid":true,"reading_age_ms":3000,"temperature_c":29.40,"humidity_pct":72.00,"uptime_ms":123456,"free_heap":310152,"free_psram":6009619,"psram_size":8386055}
```

### `RUN`
Evaluate deterministic policy on current reading, optionally generate local phrase, emit full `S3_SENTINEL_RECEIPT`.

```json
S3_SENTINEL_RECEIPT {"schema":"ri_esp32s3_sentinel_receipt_v1","firmware_variant":"final_sentinel_h256_p22","weights_sha256":"770ed901...","model_profile":"domain_h256_all_int8","params":1595937,"model_hash_verified":true,"reading_valid":true,"reading_age_ms":3000,"temperature_c":29.40,"humidity_pct":72.00,"canonical_prompt":"hot room. action is ","canonical_action":"check airflow.","confidence":0.77,"ai_route":true,"local_generated":true,"local_output":"check airflow.","local_gen_elapsed_ms":400,"local_gen_chars_per_sec":37.5,"uptime_ms":125000,"free_heap":310152,"free_psram":6009619}
```

Key fields:
- `canonical_prompt` / `canonical_action`: **deterministic**, policy-selected, never overridden by LM
- `ai_route`: true when policy escalates (hot/humid/stale/missing) — signals upstream AI should handle explanation
- `local_generated`: true iff `ai_route==true` and LM ran (16-char budget)
- `local_output`: LM phrase (stopped at `.` or `\n` or 16 chars)
- `model_hash_verified`: true only when parsed RILM SHA-256 matches build-time expectation

### `PROMPT:<text>`
Free-form language generation (unchanged from base firmware). Emits `S3_LANGUAGE_RECEIPT`.

---

## Deterministic Policy (Source of Truth)

Thresholds match `esp32-sensor-hub` contract v1:

| Condition | Threshold | Prompt | Action | AI Route |
|-----------|-----------|--------|--------|----------|
| Sensor missing | N/A | `missing sensor. action is ` | `no claim.` | true |
| Stale reading | > 120 s | `stale data. action is ` | `wait.` | true |
| Hot + Humid | ≥ 27.78 °C & ≥ 65% | `high heat and humidity. action is ` | `escalate.` | true |
| Hot only | ≥ 27.78 °C | `hot room. action is ` | `check airflow.` | true |
| Humid only | ≥ 65% | `humid room. action is ` | `ventilate.` | true |
| Cold / Dry | ≤ 15.56 °C or ≤ 25% | `safe action is ` | `no claim without evidence.` | true |
| Normal | — | `normal room. action is ` | `log receipt.` | false |

**The LM never decides safety actions.** It only provides a bounded phrasing of the already-determined canonical action when `ai_route=true`.

---

## HTTP Endpoints (WiFi Build Only)

| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | `{"ok":true,"variant":"final_sentinel_h256_p22","model":"domain_h256_all_int8","uptime_ms":123456}` |
| `/status` | GET | Same fields as `S3_SENTINEL_STATUS` JSON |

No control endpoints exist. WiFi reconnects automatically with 30 s backoff.

---

## Boot Log (Expected)

```
ESP32-S3 LSTM boot final_sentinel_h256_p22
free_heap=310152 free_psram=6009619 psram_size=8386055
worker task started on core 0
weights partition addr=0x210000 size=5242880
RILM version=1 tensors=15
RILM_TENSOR index=0 name=embedding.weight dtype=3 ndim=2 dim0=33 dim1=256 scale=0.015625 payload_len=8448 payload_offset=0 next_offset=8448 model_len=1614972
...
payloads cloned bytes=1614972 free_heap=310152 free_psram=6009619
wih int4 conversion done
S3_SENTINEL_MODEL_HASH verified=1 bytes=1614972 expected=770ed901... actual=770ed901...
state allocated free_heap=310152 free_psram=6009619
MODEL_READY profile=domain_h256_all_int8 params=1595937 hidden=256 layers=3
S3_SENTINEL_READY variant=final_sentinel_h256_p22 model_hash_verified=true offline_first=true
```

---

## Claim Boundary

| ✅ Verified / Safe to Claim | ❌ Do Not Claim |
|----------------------------|----------------|
| 39.52 chars/s (25.30 ms/char) on ESP32-S3 H256 int4+SIMD | "10 BPE tok/s" without the 4× conversion disclaimer |
| Deterministic policy with 8 canonical prompt/action pairs | LM determines safety — it does not |
| Offline-first: boots and runs without WiFi | Works as a general chatbot — vocab is 33 chars, domain-specific |
| SHA-256 model verification at boot | Model hash = model quality — it only proves artifact identity |
| Read-only HTTP endpoints | Remote control / actuation |
| OTA only with explicit build-time password | OTA without credentials |

---

## Verification Procedure

1. **Build & flash firmware:**
   ```bash
   pio run -e esp32s3_final_sentinel_h256
   pio run -e esp32s3_final_sentinel_h256 -t upload --upload-port /dev/ttyACM0
   ```

2. **Flash H256 weights:**
   ```bash
   python3 -m esptool --chip esp32s3 --port /dev/ttyACM0 --baud 921600 \
     write_flash 0x210000 weights_p12_h256_backup_770ed901.bin
   ```

3. **Reset board, open serial (115200 baud).** Observe `S3_SENTINEL_READY`.

4. **Inject sensor reading & run:**
   ```
   SENSOR:29.4,72.0,3
   RUN
   ```
   Verify `S3_SENTINEL_RECEIPT` has `canonical_action="check airflow."`, `ai_route=true`, `local_generated=true`.

5. **Test all 8 policy branches** by varying `SENSOR:` values.

6. **Run static test suite:**
   ```bash
   python3 tools/test_final_sentinel_static.py
   ```

7. **Optional: Enable WiFi build** and verify `/health` and `/status` endpoints.

---

## Files Added/Modified

| File | Purpose |
|------|---------|
| `platformio.ini` | New `esp32s3_final_sentinel_h256` env with `RI_FINAL_SENTINEL=1` |
| `src/main.cpp` | Sentinel structs, policy, serial commands, receipts, optional HTTP |
| `tools/test_final_sentinel_static.py` | Stdlib-only structural tests (no hardware) |
| `docs/FINAL_SINGLE_BOARD_SENTINEL.md` | This document |

---

## Related Artifacts

| Artifact | Location |
|----------|----------|
| H256 benchmark receipts | `benchmarks/p22_i4_wih_whh_simd_h256/` |
| Deterministic policy contract | `esp32-sensor-hub/contracts/sensor_policy_s3_local_language_v1.json` |
| Model training log | `runs/domain_lstm_h256_l3_s1200/training_log.json` |
| Prior cluster/sentinel work | `tiered-edge-ai/esp32-s3/` |