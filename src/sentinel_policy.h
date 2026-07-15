#pragma once
#include <Arduino.h>
#include "sensor_types.h"

// ── Typed policy decision ─────────────────────────────────────────────
enum PolicyDecision : uint8_t {
  POLICY_NORMAL           = 0,
  POLICY_HOT              = 1,
  POLICY_HUMID            = 2,
  POLICY_HOT_HUMID        = 3,
  POLICY_COLD             = 4,
  POLICY_DRY              = 5,
  POLICY_STALE            = 6,
  POLICY_MISSING          = 7,
  POLICY_AIR_QUALITY_HIGH = 8,
  POLICY_MOTION_DETECTED  = 9,
  POLICY_DOOR_OPEN        = 10,
  POLICY_CO2_HIGH         = 11,
  POLICY_GAS_DETECTED     = 12,
  POLICY_LOW_LIGHT        = 13,
  POLICY_HIGH_LIGHT       = 14,
  POLICY_VIBRATION_HIGH   = 15,
  POLICY_LOW_VOLTAGE      = 16,
  POLICY_HIGH_SOUND       = 17,
  POLICY_PRESSURE_LOW     = 18,
  POLICY_SOIL_DRY         = 19,
  POLICY_HIGH_CURRENT     = 20,
  POLICY_DEGRADED         = 21,
  POLICY_DECISION_COUNT   = 22,
};

struct PolicyResult {
  PolicyDecision decision;
  const char *prompt;
  const char *action;
  bool ai_route;
  float certainty;  // 0.0–1.0, typed reason-based not pseudo-probability
  const char *reason;  // short machine-readable reason string
};

// ── Thresholds ─────────────────────────────────────────────────────────
static constexpr unsigned long POLICY_STALE_AFTER_MS = 120000UL;  // 2 min
static constexpr float POLICY_HOT_C    = 27.78f;
static constexpr float POLICY_COLD_C   = 15.56f;
static constexpr float POLICY_HUMID_PCT = 65.0f;
static constexpr float POLICY_DRY_PCT   = 25.0f;
static constexpr float POLICY_AQ_HIGH_PPM  = 1000.0f;   // air quality
static constexpr float POLICY_CO2_HIGH_PPM = 1500.0f;  // CO2
static constexpr float POLICY_GAS_HIGH_PPM  = 400.0f;    // generic gas
static constexpr float POLICY_LIGHT_LOW_LUX = 10.0f;
static constexpr float POLICY_LIGHT_HIGH_LUX = 5000.0f;
static constexpr float POLICY_VIBRATION_HIGH_MG = 2000.0f;
static constexpr float POLICY_VOLTAGE_LOW_V  = 3.0f;
static constexpr float POLICY_SOUND_HIGH_DB  = 85.0f;
static constexpr float POLICY_PRESSURE_LOW_HPA = 980.0f;
static constexpr float POLICY_SOIL_DRY_PCT  = 20.0f;
static constexpr float POLICY_HIGH_CURRENT_A = 10.0f;

// ── Policy table ───────────────────────────────────────────────────────
static const char *policy_decision_name(PolicyDecision d) {
  switch (d) {
    case POLICY_NORMAL:           return "normal";
    case POLICY_HOT:              return "hot";
    case POLICY_HUMID:            return "humid";
    case POLICY_HOT_HUMID:        return "hot_humid";
    case POLICY_COLD:             return "cold";
    case POLICY_DRY:              return "dry";
    case POLICY_STALE:            return "stale";
    case POLICY_MISSING:          return "missing";
    case POLICY_AIR_QUALITY_HIGH: return "air_quality_high";
    case POLICY_MOTION_DETECTED:  return "motion_detected";
    case POLICY_DOOR_OPEN:        return "door_open";
    case POLICY_CO2_HIGH:         return "co2_high";
    case POLICY_GAS_DETECTED:     return "gas_detected";
    case POLICY_LOW_LIGHT:        return "low_light";
    case POLICY_HIGH_LIGHT:       return "high_light";
    case POLICY_VIBRATION_HIGH:   return "vibration_high";
    case POLICY_LOW_VOLTAGE:      return "low_voltage";
    case POLICY_HIGH_SOUND:       return "high_sound";
    case POLICY_PRESSURE_LOW:    return "pressure_low";
    case POLICY_SOIL_DRY:        return "soil_dry";
    case POLICY_HIGH_CURRENT:    return "high_current";
    case POLICY_DEGRADED:        return "degraded";
    default:                     return "unknown";
  }
}

// ── Policy evaluation ──────────────────────────────────────────────────
// Evaluates all sensor readings and returns the highest-priority decision.
// Priority order: missing > stale > hot_humid > air_quality_high > co2_high >
// gas_detected > hot > humid > cold/dry > pressure_low > soil_dry > low_voltage >
// low_light > high_light > vibration_high > high_sound > motion > door_open >
// high_current > normal

static PolicyResult evaluate_policy(SensorBuffer &buf) {
  buf.update_age();

  // Check for missing sensors (if any were registered but never updated)
  bool any_missing = false;
  bool any_stale = false;
  for (int i = 0; i < buf.count; i++) {
    if (buf.readings[i].quality == QUALITY_MISSING) any_missing = true;
    if (buf.readings[i].quality == QUALITY_STALE ||
        (buf.readings[i].quality == QUALITY_VALID && buf.readings[i].age_ms > POLICY_STALE_AFTER_MS))
      any_stale = true;
  }

  // If no sensors at all, treat as missing
  if (buf.count == 0) any_missing = true;

  if (any_missing)
    return { POLICY_MISSING, "missing sensor. action is ", "no claim.", true, 0.0f, "sensor_missing" };
  if (any_stale)
    return { POLICY_STALE, "stale data. action is ", "wait.", true, 0.1f, "stale_reading" };

  // Get specific sensor readings
  SensorReading *temp = buf.find(SENSOR_TEMP);
  SensorReading *hum  = buf.find(SENSOR_HUMIDITY);
  SensorReading *aq   = buf.find(SENSOR_AIR_QUALITY);
  SensorReading *co2  = buf.find(SENSOR_CO2);
  SensorReading *gas  = buf.find(SENSOR_GAS);
  SensorReading *light = buf.find(SENSOR_LIGHT);
  SensorReading *vib  = buf.find(SENSOR_VIBRATION);
  SensorReading *volt = buf.find(SENSOR_VOLTAGE);
  SensorReading *sound = buf.find(SENSOR_SOUND_LEVEL);
  SensorReading *press = buf.find(SENSOR_PRESSURE);
  SensorReading *soil  = buf.find(SENSOR_SOIL_MOISTURE);
  SensorReading *cur   = buf.find(SENSOR_CURRENT);
  SensorReading *motion = buf.find(SENSOR_MOTION);
  SensorReading *door   = buf.find(SENSOR_DOOR_CONTACT);

  // Helper: check if a reading is valid and above threshold
  auto valid_above = [](SensorReading *r, float thr) -> bool {
    return r && r->quality == QUALITY_VALID && r->value >= thr;
  };
  auto valid_below = [](SensorReading *r, float thr) -> bool {
    return r && r->quality == QUALITY_VALID && r->value < thr;
  };

  // Temperature + humidity (highest priority environmental)
  bool hot = valid_above(temp, POLICY_HOT_C);
  bool cold = valid_below(temp, POLICY_COLD_C);
  bool humid = valid_above(hum, POLICY_HUMID_PCT);
  bool dry = valid_below(hum, POLICY_DRY_PCT);

  // Air quality / CO2 / Gas (safety critical)
  if (valid_above(co2, POLICY_CO2_HIGH_PPM))
    return { POLICY_CO2_HIGH, "co2 is high. action is ", "ventilate now.", true, 0.95f, "co2_above_threshold" };
  if (valid_above(gas, POLICY_GAS_HIGH_PPM))
    return { POLICY_GAS_DETECTED, "gas detected. action is ", "ventilate.", true, 0.90f, "gas_above_threshold" };
  if (valid_above(aq, POLICY_AQ_HIGH_PPM))
    return { POLICY_AIR_QUALITY_HIGH, "air quality is poor. action is ", "ventilate.", true, 0.85f, "air_quality_above_threshold" };

  // Temperature/humidity combined
  if (hot && humid)
    return { POLICY_HOT_HUMID, "high heat and humidity. action is ", "escalate.", true, 0.77f, "temp_and_humidity_out_of_range" };
  if (hot)
    return { POLICY_HOT, "hot room. action is ", "check airflow.", true, 0.77f, "temperature_out_of_range" };
  if (humid)
    return { POLICY_HUMID, "humid room. action is ", "ventilate.", true, 0.77f, "humidity_out_of_range" };
  if (cold || dry)
    return { POLICY_COLD, "safe action is ", "no claim without evidence.", true, 0.50f, "unsupported_cold_or_dry" };

  // Pressure
  if (valid_below(press, POLICY_PRESSURE_LOW_HPA))
    return { POLICY_PRESSURE_LOW, "pressure is low. action is ", "log receipt.", true, 0.60f, "pressure_below_threshold" };

  // Soil moisture
  if (valid_below(soil, POLICY_SOIL_DRY_PCT))
    return { POLICY_SOIL_DRY, "soil is dry. action is ", "water.", true, 0.65f, "soil_moisture_below_threshold" };

  // Voltage
  if (valid_below(volt, POLICY_VOLTAGE_LOW_V))
    return { POLICY_LOW_VOLTAGE, "voltage is low. action is ", "check power.", true, 0.70f, "voltage_below_threshold" };

  // Light
  if (valid_below(light, POLICY_LIGHT_LOW_LUX))
    return { POLICY_LOW_LIGHT, "light is low. action is ", "log receipt.", false, 0.40f, "light_below_threshold" };
  if (valid_above(light, POLICY_LIGHT_HIGH_LUX))
    return { POLICY_HIGH_LIGHT, "light is high. action is ", "log receipt.", false, 0.40f, "light_above_threshold" };

  // Vibration
  if (valid_above(vib, POLICY_VIBRATION_HIGH_MG))
    return { POLICY_VIBRATION_HIGH, "vibration is high. action is ", "inspect.", true, 0.75f, "vibration_above_threshold" };

  // Sound
  if (valid_above(sound, POLICY_SOUND_HIGH_DB))
    return { POLICY_HIGH_SOUND, "sound is loud. action is ", "log receipt.", false, 0.50f, "sound_above_threshold" };

  // Current
  if (valid_above(cur, POLICY_HIGH_CURRENT_A))
    return { POLICY_HIGH_CURRENT, "current is high. action is ", "check load.", true, 0.70f, "current_above_threshold" };

  // Motion / door (event-driven, lower priority)
  if (valid_above(motion, 0.5f))
    return { POLICY_MOTION_DETECTED, "motion detected. action is ", "log event.", false, 0.30f, "motion_event" };
  if (valid_above(door, 0.5f))
    return { POLICY_DOOR_OPEN, "door is open. action is ", "log event.", false, 0.30f, "door_open_event" };

  // All clear
  return { POLICY_NORMAL, "normal room. action is ", "log receipt.", false, 0.95f, "local_confident" };
}