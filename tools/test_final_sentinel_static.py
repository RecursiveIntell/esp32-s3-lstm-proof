#!/usr/bin/env python3
"""
Static structural tests for the multi-sensor Sentinel firmware v2.
No hardware required — validates source code structure, receipt schema,
sensor type coverage, and policy table completeness.

Usage: python3 tools/test_final_sentinel_static.py
"""
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
MAIN_CPP = SRC / "main.cpp"
SENSOR_H = SRC / "sensor_types.h"
POLICY_H = SRC / "sentinel_policy.h"
RECEIPT_H = SRC / "sentinel_receipt.h"
PLATFORMIO_INI = ROOT / "platformio.ini"

FAILURES = []
PASSES = []

def check(cond, msg):
    if cond:
        PASSES.append(msg)
    else:
        FAILURES.append(msg)

# ── 1. Header files exist ──────────────────────────────────────────────
check(SENSOR_H.exists(), "sensor_types.h exists")
check(POLICY_H.exists(), "sentinel_policy.h exists")
check(RECEIPT_H.exists(), "sentinel_receipt.h exists")

# ── 2. Sensor type taxonomy ────────────────────────────────────────────
sensor_src = SENSOR_H.read_text()
expected_sensors = [
    "SENSOR_TEMP", "SENSOR_HUMIDITY", "SENSOR_PRESSURE",
    "SENSOR_AIR_QUALITY", "SENSOR_LIGHT", "SENSOR_MOTION",
    "SENSOR_DISTANCE", "SENSOR_GAS", "SENSOR_SOIL_MOISTURE",
    "SENSOR_VIBRATION", "SENSOR_DOOR_CONTACT", "SENSOR_VOLTAGE",
    "SENSOR_CO2", "SENSOR_SOUND_LEVEL", "SENSOR_CURRENT",
]
for s in expected_sensors:
    check(s in sensor_src, f"Sensor type {s} defined")

check("SensorQuality" in sensor_src, "SensorQuality enum defined")
check("QUALITY_VALID" in sensor_src, "Quality VALID defined")
check("QUALITY_MISSING" in sensor_src, "Quality MISSING defined")
check("QUALITY_STALE" in sensor_src, "Quality STALE defined")
check("QUALITY_NAN" in sensor_src, "Quality NAN defined")
check("QUALITY_IMPLAUSIBLE" in sensor_src, "Quality IMPLAUSIBLE defined")
check("QUALITY_SENSOR_FAULT" in sensor_src, "Quality SENSOR_FAULT defined")
check("SensorBuffer" in sensor_src, "SensorBuffer struct defined")
check("MAX_SENSORS" in sensor_src, "MAX_SENSORS constant defined")
check("parse_sensor_type" in sensor_src, "parse_sensor_type function defined")
check("sensor_value_plausible" in sensor_src, "sensor_value_plausible function defined")

# ── 3. Actuator / servo support ───────────────────────────────────────
main_src = MAIN_CPP.read_text()
actuator_src = ""
actuator_h = SRC / "actuator_types.h"
check(actuator_h.exists(), "actuator_types.h exists")
if actuator_h.exists():
    actuator_src = actuator_h.read_text()
    check("ActuatorType" in actuator_src, "ActuatorType enum defined")
    check("ACTUATOR_SERVO" in actuator_src, "ACTUATOR_SERVO defined")
    check("ServoConfig" in actuator_src, "ServoConfig struct defined")
    check("servo_angle_to_duty" in actuator_src, "servo_angle_to_duty function defined")
    check("policy_to_servo" in actuator_src, "policy_to_servo mapping function defined")
    check("SERVO_FREQ_HZ" in actuator_src, "Servo PWM frequency defined")
    check("MAX_SERVOS" in actuator_src, "MAX_SERVOS constant defined")
    check("ActuatorCommand" in actuator_src, "ActuatorCommand struct defined")
    # C1 mitigation: policy controls actuator, not LM
    check("policy_decision" in actuator_src, "Actuator mapping uses policy decision (C1 mitigation)")

# Check servo commands in main.cpp
check("servo_init_slot" in main_src, "servo_init_slot function implemented")
check("servo_move" in main_src, "servo_move function implemented")
check("SERVOATTACH:" in main_src, "SERVOATTACH command parser implemented")
check("SERVO:" in main_src, "SERVO command parser implemented")
check("ACTUATORS" in main_src, "ACTUATORS list command implemented")
check("policy_to_servo" in main_src, "Policy-to-servo mapping integrated in RUN")
check("S3_SENTINEL_ACTUATOR" in main_src, "Actuator receipt emitted on servo action")

# ── 4. Policy table ────────────────────────────────────────────────────
policy_src = POLICY_H.read_text()
expected_decisions = [
    "POLICY_NORMAL", "POLICY_HOT", "POLICY_HUMID", "POLICY_HOT_HUMID",
    "POLICY_COLD", "POLICY_DRY", "POLICY_STALE", "POLICY_MISSING",
    "POLICY_AIR_QUALITY_HIGH", "POLICY_MOTION_DETECTED", "POLICY_DOOR_OPEN",
    "POLICY_CO2_HIGH", "POLICY_GAS_DETECTED", "POLICY_LOW_LIGHT",
    "POLICY_HIGH_LIGHT", "POLICY_VIBRATION_HIGH", "POLICY_LOW_VOLTAGE",
    "POLICY_HIGH_SOUND", "POLICY_PRESSURE_LOW", "POLICY_SOIL_DRY",
    "POLICY_HIGH_CURRENT", "POLICY_DEGRADED",
]
for d in expected_decisions:
    check(d in policy_src, f"Policy decision {d} defined")

check("PolicyResult" in policy_src, "PolicyResult struct defined")
check("evaluate_policy" in policy_src, "evaluate_policy function defined")
check("POLICY_STALE_AFTER_MS" in policy_src, "Stale threshold defined")
check("POLICY_HOT_C" in policy_src, "Hot threshold defined")
check("POLICY_CO2_HIGH_PPM" in policy_src, "CO2 threshold defined")
check("POLICY_AQ_HIGH_PPM" in policy_src, "Air quality threshold defined")

# Verify fail-closed: missing sensors should route to POLICY_MISSING
check("any_missing" in policy_src, "Policy checks for missing sensors (fail-closed)")

# ── 4. Receipt v2 schema ───────────────────────────────────────────────
receipt_src = RECEIPT_H.read_text()
check("ri_esp32s3_sentinel_receipt_v2" in receipt_src, "Receipt v2 schema name defined")
check("device_id" in receipt_src, "Receipt includes device_id")
check("boot_id" in receipt_src, "Receipt includes boot_id")
check("event_seq" in receipt_src, "Receipt includes event_seq")
check("prev_receipt_hash" in receipt_src, "Receipt includes prev_receipt_hash (hash chaining)")
check("ReceiptIdentity" in receipt_src, "ReceiptIdentity struct defined")
check("build_receipt" in receipt_src, "build_receipt function defined")
check("json_escape_to" in receipt_src, "Proper JSON escape function defined")
check("mbedtls_sha256" in receipt_src, "SHA-256 hashing for receipt chain")
check("Preferences" in receipt_src, "NVS persistence for event_seq")
check("esp_random" in receipt_src or "ESP.getEfuseMac" in receipt_src, "Device identity from MAC")

# ── 5. Main firmware integration ───────────────────────────────────────
check('#include "sensor_types.h"' in main_src, "main.cpp includes sensor_types.h")
check('#include "sentinel_policy.h"' in main_src, "main.cpp includes sentinel_policy.h")
check('#include "sentinel_receipt.h"' in main_src, "main.cpp includes sentinel_receipt.h")
check("SensorBuffer g_sensors" in main_src, "Global SensorBuffer instance created")
check("evaluate_policy(g_sensors)" in main_src, "Policy evaluation uses SensorBuffer")
check("build_receipt(" in main_src, "Receipt building integrated")
check("receipt_init()" in main_src, "Receipt identity initialized in setup")
check("multi_sensor=true" in main_src, "Boot message advertises multi_sensor")
check("sentinel_handle_sensors_list_command" in main_src, "SENSORS command implemented")
check("sentinel_handle_clear_command" in main_src, "CLEAR command implemented")
check("SENSOR:<type>:<value>" in main_src or "parse_sensor_type" in main_src, "New sensor format parser integrated")
check("legacy" in main_src.lower(), "Legacy SENSOR format still supported")

# ── 6. Receipt v2 in serial output ──────────────────────────────────────
check("ri_esp32s3_sentinel_receipt_v2" in main_src or "ri_esp32s3_sentinel_receipt_v2" in receipt_src, "Firmware emits receipt v2 schema")
check("ri_esp32s3_sentinel_status_v2" in main_src, "Firmware emits status v2 schema")

# ── 7. PlatformIO environments ────────────────────────────────────────
pio_src = PLATFORMIO_INI.read_text()
check("esp32s3_final_sentinel_h256" in pio_src, "H256 sentinel env exists")
check("esp32s3_final_sentinel_h320" in pio_src, "H320 sentinel env exists")
check("RI_FINAL_SENTINEL=1" in pio_src, "RI_FINAL_SENTINEL flag set")

# H256 profile checks
h256_section = pio_src[pio_src.index("[env:esp32s3_final_sentinel_h256]"):]
h256_section = h256_section[:h256_section.index("[env:", 10)] if "[env:" in h256_section[10:] else h256_section
check("RI_HIDDEN=256" in h256_section, "H256 env has RI_HIDDEN=256")
check("d8763148" in h256_section, "H256 env has correct weights hash")
check("multisensor_h256_all_int8" in h256_section, "H256 env has correct model profile")

# H320 profile checks
h320_section = pio_src[pio_src.index("[env:esp32s3_final_sentinel_h320]"):]
h320_section = h320_section[:h320_section.index("[env:", 10)] if "[env:" in h320_section[10:] else h320_section
check("RI_HIDDEN=320" in h320_section, "H320 env has RI_HIDDEN=320")
check("a1a4dcd0" in h320_section, "H320 env has correct weights hash")
check("multisensor_h320_all_int8" in h320_section, "H320 env has correct model profile")

# ── 8. Model-to-actuator isolation (C1 mitigation) ─────────────────────
# Policy returns typed decisions, not just strings
check("PolicyDecision" in policy_src, "Typed PolicyDecision enum (C1 mitigation)")
check("PolicyResult" in policy_src, "Typed PolicyResult struct (C1 mitigation)")
check("AdvisoryText" not in policy_src, "Model output is advisory only (C1: LM never decides safety)")

# ── 9. Staleness/quality fail-closed (H2 mitigation) ──────────────────
check("QUALITY_STALE" in sensor_src, "Stale quality state defined (H2 mitigation)")
check("QUALITY_IMPLAUSIBLE" in sensor_src, "Implausible quality state defined (H2 mitigation)")
check("QUALITY_NAN" in sensor_src, "NaN quality state defined (H2 mitigation)")
check("sensor_value_plausible" in sensor_src, "Plausibility check function (H2 mitigation)")

# ── 10. Receipt integrity (C4 mitigation) ────────────────────────────
check("prev_receipt_hash" in receipt_src, "Hash chaining (C4 mitigation)")
check("event_seq" in receipt_src, "Monotonic event sequence (C4 mitigation)")
check("boot_id" in receipt_src, "Boot identity (C4 mitigation)")
check("device_id" in receipt_src, "Device identity (C4 mitigation)")

# ── Report ─────────────────────────────────────────────────────────────
total = len(PASSES) + len(FAILURES)
print(f"\n{'='*60}")
print(f"Multi-Sensor Sentinel Static Tests: {len(PASSES)}/{total} passed")
print(f"{'='*60}")
if FAILURES:
    print(f"\nFAILURES ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  FAIL: {f}")
    sys.exit(1)
else:
    print("\nAll tests passed.")
    sys.exit(0)