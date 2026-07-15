#pragma once
#include <Arduino.h>
#include <Preferences.h>
#include <mbedtls/sha256.h>
#include "sensor_types.h"
#include "sentinel_policy.h"

// ── Receipt v2: hash-chained, boot/event identity ─────────────────────
#define RECEIPT_MAX_LEN 2048
#define SENSOR_JSON_MAX 512

struct ReceiptIdentity {
  char device_id[24];      // MAC-based, e.g. "ESP32S3-AABBCC"
  uint32_t boot_id;        // random per boot
  uint32_t event_seq;      // persisted in NVS, monotonic
  char prev_hash[17];     // first 16 hex of SHA-256 of previous receipt
  bool hash_initialized;  // first receipt has no previous
};

static ReceiptIdentity g_receipt_id;
static Preferences g_receipt_prefs;
static char g_receipt_buffer[RECEIPT_MAX_LEN];

// ── Initialization ─────────────────────────────────────────────────────
static void receipt_init() {
  // Device ID from MAC
  uint64_t mac = ESP.getEfuseMac();
  snprintf(g_receipt_id.device_id, sizeof(g_receipt_id.device_id),
           "ESP32S3-%02X%02X%02X",
           (uint8_t)(mac >> 16), (uint8_t)(mac >> 8), (uint8_t)mac);
  // Boot ID: random 32-bit
  g_receipt_id.boot_id = esp_random();
  // Event seq: load from NVS, default 0
  g_receipt_prefs.begin("ri_receipt", false);
  g_receipt_id.event_seq = g_receipt_prefs.getUInt("event_seq", 0);
  // Prev hash: empty for first receipt
  g_receipt_id.prev_hash[0] = '\0';
  g_receipt_id.hash_initialized = false;
  Serial.printf("S3_SENTINEL_IDENTITY device_id=%s boot_id=0x%08lX event_seq=%lu\n",
                g_receipt_id.device_id, (unsigned long)g_receipt_id.boot_id,
                (unsigned long)g_receipt_id.event_seq);
}

static void receipt_advance_seq() {
  g_receipt_id.event_seq++;
  g_receipt_prefs.putUInt("event_seq", g_receipt_id.event_seq);
}

// ── JSON string escaping ───────────────────────────────────────────────
static size_t json_escape_to(char *dst, size_t dst_max, const char *src) {
  size_t n = 0;
  for (const char *p = src; *p && n + 2 < dst_max; p++) {
    switch (*p) {
      case '"':  if (n + 2 < dst_max) { dst[n++]='\\'; dst[n++]='"'; } break;
      case '\\': if (n + 2 < dst_max) { dst[n++]='\\'; dst[n++]='\\'; } break;
      case '\n': if (n + 2 < dst_max) { dst[n++]='\\'; dst[n++]='n';  } break;
      case '\r': if (n + 2 < dst_max) { dst[n++]='\\'; dst[n++]='r';  } break;
      case '\t': if (n + 2 < dst_max) { dst[n++]='\\'; dst[n++]='t';  } break;
      default:
        if ((uint8_t)*p < 0x20) {
          if (n + 6 < dst_max) { n += snprintf(dst + n, dst_max - n, "\\u%04x", (uint8_t)*p); }
        } else {
          dst[n++] = *p;
        }
        break;
    }
  }
  dst[n] = '\0';
  return n;
}

// ── Build sensor readings JSON array ──────────────────────────────────
static void build_sensor_json(char *buf, size_t buf_max, SensorBuffer &sensors) {
  size_t n = 0;
  n += snprintf(buf + n, buf_max - n, "[");

  bool first = true;
  for (int i = 0; i < sensors.count; i++) {
    SensorReading &r = sensors.readings[i];
    if (r.type == SENSOR_NONE) continue;

    const char *sep = first ? "" : ",";
    first = false;

    char val_str[32];
    if (isnan(r.value)) snprintf(val_str, sizeof(val_str), "null");
    else snprintf(val_str, sizeof(val_str), "%.2f", r.value);

    char sec_str[32];
    if (isnan(r.secondary)) snprintf(sec_str, sizeof(sec_str), "null");
    else snprintf(sec_str, sizeof(sec_str), "%.2f", r.secondary);

    n += snprintf(buf + n, buf_max - n,
      "%s{\"type\":\"%s\",\"quality\":\"%s\",\"value\":%s,\"secondary\":%s,\"age_ms\":%lu,\"unit\":\"%s\"}",
      sep, sensor_type_name(r.type), quality_str(r.quality),
      val_str, sec_str, (unsigned long)r.age_ms, sensor_type_unit(r.type));

    if (n >= buf_max - 10) break;  // leave room for closing
  }
  n += snprintf(buf + n, buf_max - n, "]");
}

// ── Build and emit receipt with hash chaining ──────────────────────────
// Returns pointer to the receipt string (static buffer).
static const char *build_receipt(
    SensorBuffer &sensors,
    PolicyResult &policy,
    bool local_generated,
    const char *local_output,
    uint32_t gen_elapsed_ms,
    float gen_chars_per_sec,
    bool model_hash_verified,
    const char *firmware_variant,
    const char *model_profile,
    uint32_t model_params,
    const char *weights_sha256) {

  char sensor_json[SENSOR_JSON_MAX];
  build_sensor_json(sensor_json, sizeof(sensor_json), sensors);

  // Build receipt body (without closing brace — we compute hash on body, then chain)
  int n = snprintf(g_receipt_buffer, sizeof(g_receipt_buffer),
    "{\"schema\":\"ri_esp32s3_sentinel_receipt_v2\","
    "\"device_id\":\"%s\","
    "\"boot_id\":%lu,"
    "\"event_seq\":%lu,"
    "\"prev_receipt_hash\":\"%s\","
    "\"firmware_variant\":\"%s\","
    "\"model_profile\":\"%s\","
    "\"params\":%lu,"
    "\"weights_sha256\":\"%s\","
    "\"model_hash_verified\":%s,"
    "\"readings\":%s,"
    "\"policy_decision\":\"%s\","
    "\"canonical_prompt\":\"",
    g_receipt_id.device_id,
    (unsigned long)g_receipt_id.boot_id,
    (unsigned long)g_receipt_id.event_seq,
    g_receipt_id.prev_hash,
    firmware_variant,
    model_profile,
    (unsigned long)model_params,
    weights_sha256,
    model_hash_verified ? "true" : "false",
    sensor_json,
    policy_decision_name(policy.decision));

  // Escape prompt
  char escaped[256];
  json_escape_to(escaped, sizeof(escaped), policy.prompt);
  n += snprintf(g_receipt_buffer + n, sizeof(g_receipt_buffer) - n, "%s\",", escaped);

  // Escape action
  json_escape_to(escaped, sizeof(escaped), policy.action);
  n += snprintf(g_receipt_buffer + n, sizeof(g_receipt_buffer) - n, "\"canonical_action\":\"%s\",", escaped);

  // Certainty, reason, ai_route
  n += snprintf(g_receipt_buffer + n, sizeof(g_receipt_buffer) - n,
    "\"certainty\":%.2f,"
    "\"reason\":\"%s\","
    "\"ai_route\":%s,"
    "\"local_generated\":%s,",
    policy.certainty, policy.reason,
    policy.ai_route ? "true" : "false",
    local_generated ? "true" : "false");

  // Local output (if generated)
  if (local_generated) {
    json_escape_to(escaped, sizeof(escaped), local_output);
    n += snprintf(g_receipt_buffer + n, sizeof(g_receipt_buffer) - n,
      "\"local_output\":\"%s\","
      "\"local_gen_elapsed_ms\":%lu,"
      "\"local_gen_chars_per_sec\":%.4f,",
      escaped, (unsigned long)gen_elapsed_ms, gen_chars_per_sec);
  }

  // Footer fields
  n += snprintf(g_receipt_buffer + n, sizeof(g_receipt_buffer) - n,
    "\"uptime_ms\":%lu,"
    "\"free_heap\":%lu,"
    "\"free_psram\":%lu}",
    (unsigned long)millis(),
    (unsigned long)ESP.getFreeHeap(),
    (unsigned long)ESP.getFreePsram());

  // Compute SHA-256 of this receipt for chaining
  uint8_t digest[32];
  mbedtls_sha256_context ctx;
  mbedtls_sha256_init(&ctx);
  mbedtls_sha256_starts_ret(&ctx, 0);
  // Hash in chunks (watchdog-safe)
  size_t receipt_len = strlen(g_receipt_buffer);
  for (size_t off = 0; off < receipt_len; off += 4096) {
    size_t chunk = receipt_len - off;
    if (chunk > 4096) chunk = 4096;
    mbedtls_sha256_update_ret(&ctx, (const uint8_t *)(g_receipt_buffer + off), chunk);
  }
  mbedtls_sha256_finish_ret(&ctx, digest);
  mbedtls_sha256_free(&ctx);

  // Store first 16 hex chars as prev_hash for next receipt
  static const char kHex[] = "0123456789abcdef";
  for (int i = 0; i < 8; i++) {
    g_receipt_id.prev_hash[i * 2]     = kHex[digest[i] >> 4];
    g_receipt_id.prev_hash[i * 2 + 1] = kHex[digest[i] & 0x0f];
  }
  g_receipt_id.prev_hash[16] = '\0';
  g_receipt_id.hash_initialized = true;

  // Advance event seq for next receipt
  receipt_advance_seq();

  return g_receipt_buffer;
}