#pragma once
#include <Arduino.h>

// ── Sensor type taxonomy ──────────────────────────────────────────────
enum SensorType : uint8_t {
  SENSOR_NONE        = 0,
  SENSOR_TEMP        = 1,   // °C
  SENSOR_HUMIDITY    = 2,   // %
  SENSOR_PRESSURE    = 3,   // hPa
  SENSOR_AIR_QUALITY = 4,   // ppm (eCO2 / TVOC equivalent)
  SENSOR_LIGHT       = 5,   // lux
  SENSOR_MOTION      = 6,   // 0=none 1=detected
  SENSOR_DISTANCE    = 7,   // cm
  SENSOR_GAS         = 8,   // ppm (generic gas sensor)
  SENSOR_SOIL_MOISTURE = 9, // % (capacitive soil probe)
  SENSOR_VIBRATION   = 10,  // mg (acceleration magnitude)
  SENSOR_DOOR_CONTACT = 11, // 0=closed 1=open
  SENSOR_VOLTAGE     = 12,  // V
  SENSOR_CO2         = 13,  // ppm (dedicated CO2, e.g. SCD41)
  SENSOR_SOUND_LEVEL = 14,  // dB
  SENSOR_CURRENT     = 15,  // A
  SENSOR_TYPE_COUNT  = 16,
};

// ── Sensor quality / fault state ───────────────────────────────────────
enum SensorQuality : uint8_t {
  QUALITY_VALID        = 0,
  QUALITY_MISSING      = 1,   // no reading injected yet
  QUALITY_STALE        = 2,   // age > staleness threshold
  QUALITY_NAN          = 3,   // non-finite value
  QUALITY_IMPLAUSIBLE  = 4,   // outside plausible range
  QUALITY_SENSOR_FAULT = 5,   // explicit fault flag
};

// ── Plausibility ranges per sensor type ───────────────────────────────
struct SensorRange {
  float min_val;
  float max_val;
  const char *unit;
  const char *name;     // short name for serial protocol
};

static const SensorRange SENSOR_RANGES[SENSOR_TYPE_COUNT] = {
  {   0.0f,    0.0f, "",       "none"        },  // 0
  { -40.0f,   85.0f, "c",      "temp"        },  // 1
  {   0.0f,  100.0f, "pct",    "humidity"    },  // 2
  { 800.0f, 1200.0f, "hpa",    "pressure"    },  // 3
  {   0.0f, 5000.0f, "ppm",    "air_quality" },  // 4
  {   0.0f, 100000.0f, "lux",  "light"       },  // 5
  {   0.0f,    1.0f, "bool",   "motion"      },  // 6
  {   0.0f, 400.0f, "cm",      "distance"    },  // 7
  {   0.0f, 10000.0f, "ppm",   "gas"         },  // 8
  {   0.0f,  100.0f, "pct",    "soil"        },  // 9
  {   0.0f, 16000.0f, "mg",    "vibration"   },  // 10
  {   0.0f,    1.0f, "bool",   "door"        },  // 11
  {   0.0f,   30.0f, "v",      "voltage"     },  // 12
  {   0.0f, 10000.0f, "ppm",   "co2"         },  // 13
  {   0.0f,  140.0f, "db",     "sound"       },  // 14
  {   0.0f,   30.0f, "a",      "current"     },  // 15
};

// ── Unified sensor reading ─────────────────────────────────────────────
#define SENSOR_LABEL_LEN 16

struct SensorReading {
  SensorType    type        = SENSOR_NONE;
  SensorQuality quality    = QUALITY_MISSING;
  float         value       = NAN;
  float         secondary   = NAN;  // dual-value sensors (e.g. temp+hum)
  unsigned long age_ms      = 0;
  unsigned long last_update = 0;    // millis() at injection
};

// ── Multi-sensor buffer ────────────────────────────────────────────────
#define MAX_SENSORS 16

struct SensorBuffer {
  SensorReading readings[MAX_SENSORS];
  int count = 0;  // how many slots have been populated

  SensorReading *find(SensorType t) {
    for (int i = 0; i < count; i++)
      if (readings[i].type == t) return &readings[i];
    return nullptr;
  }

  SensorReading *find_or_alloc(SensorType t) {
    SensorReading *r = find(t);
    if (r) return r;
    if (count < MAX_SENSORS) {
      readings[count].type = t;
      readings[count].quality = QUALITY_MISSING;
      return &readings[count++];
    }
    return nullptr;
  }

  void update_age() {
    unsigned long now = millis();
    for (int i = 0; i < count; i++) {
      if (readings[i].quality == QUALITY_VALID) {
        readings[i].age_ms = now - readings[i].last_update;
      }
    }
  }
};

// ── Sensor type parsing helpers ───────────────────────────────────────
static SensorType parse_sensor_type(const char *s) {
  for (int i = 1; i < SENSOR_TYPE_COUNT; i++) {
    if (strcmp(s, SENSOR_RANGES[i].name) == 0) return (SensorType)i;
  }
  // Short aliases
  if (strcmp(s, "t") == 0) return SENSOR_TEMP;
  if (strcmp(s, "h") == 0) return SENSOR_HUMIDITY;
  if (strcmp(s, "p") == 0) return SENSOR_PRESSURE;
  if (strcmp(s, "aq") == 0) return SENSOR_AIR_QUALITY;
  if (strcmp(s, "l") == 0) return SENSOR_LIGHT;
  if (strcmp(s, "mot") == 0) return SENSOR_MOTION;
  if (strcmp(s, "dist") == 0) return SENSOR_DISTANCE;
  if (strcmp(s, "g") == 0) return SENSOR_GAS;
  if (strcmp(s, "soil") == 0) return SENSOR_SOIL_MOISTURE;
  if (strcmp(s, "vib") == 0) return SENSOR_VIBRATION;
  if (strcmp(s, "door") == 0) return SENSOR_DOOR_CONTACT;
  if (strcmp(s, "v") == 0) return SENSOR_VOLTAGE;
  if (strcmp(s, "co2") == 0) return SENSOR_CO2;
  if (strcmp(s, "snd") == 0) return SENSOR_SOUND_LEVEL;
  if (strcmp(s, "cur") == 0) return SENSOR_CURRENT;
  return SENSOR_NONE;
}

static const char *sensor_type_name(SensorType t) {
  if (t < SENSOR_TYPE_COUNT) return SENSOR_RANGES[t].name;
  return "unknown";
}

static const char *sensor_type_unit(SensorType t) {
  if (t < SENSOR_TYPE_COUNT) return SENSOR_RANGES[t].unit;
  return "";
}

static bool sensor_value_plausible(SensorType t, float v) {
  if (t >= SENSOR_TYPE_COUNT) return false;
  const SensorRange &r = SENSOR_RANGES[t];
  return !isnan(v) && isfinite(v) && v >= r.min_val && v <= r.max_val;
}

static const char *quality_str(SensorQuality q) {
  switch (q) {
    case QUALITY_VALID:        return "valid";
    case QUALITY_MISSING:      return "missing";
    case QUALITY_STALE:        return "stale";
    case QUALITY_NAN:          return "nan";
    case QUALITY_IMPLAUSIBLE:  return "implausible";
    case QUALITY_SENSOR_FAULT: return "fault";
    default:                   return "unknown";
  }
}