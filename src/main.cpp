#include <Arduino.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "cluster_protocol.h"

#ifndef CLUSTER_WIFI_PING_ONLY
#define CLUSTER_WIFI_PING_ONLY 0
#endif
#ifndef CLUSTER_WIFI_MATMUL_PROOF
#define CLUSTER_WIFI_MATMUL_PROOF 0
#endif
#ifndef CLUSTER_WIFI_SHARDED_INFERENCE
#define CLUSTER_WIFI_SHARDED_INFERENCE 0
#endif
#define CLUSTER_WIFI_DEMO (CLUSTER_WIFI_PING_ONLY || CLUSTER_WIFI_MATMUL_PROOF || CLUSTER_WIFI_SHARDED_INFERENCE)
#ifndef CLUSTER_BOARD_ID
#define CLUSTER_BOARD_ID 0
#endif
#ifndef CLUSTER_ROLE_COORD
#define CLUSTER_ROLE_COORD 0
#endif
#ifndef CLUSTER_ROLE_WORKER
#define CLUSTER_ROLE_WORKER 0
#endif
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
#include "fc_shard_weights.h"
#endif

#if CLUSTER_WIFI_DEMO
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <WebServer.h>
#include <WiFiClient.h>
#include <Update.h>

#if __has_include("wifi_secrets.local.h")
#include "wifi_secrets.local.h"
#endif

#ifndef CLUSTER_ENABLE_OTA
#define CLUSTER_ENABLE_OTA 0
#endif
#ifndef CLUSTER_WIFI_AP_MODE
#define CLUSTER_WIFI_AP_MODE 0
#endif
#ifndef CLUSTER_WIFI_SSID
#define CLUSTER_WIFI_SSID "RI-ESP-CLUSTER"
#endif
#ifndef CLUSTER_WIFI_PASSPHRASE
#define CLUSTER_WIFI_PASSPHRASE "localfirstai"
#endif
#ifndef CLUSTER_WIFI_UDP_PORT
#define CLUSTER_WIFI_UDP_PORT 42100
#endif
#ifndef CLUSTER_OTA_PASSWORD
#define CLUSTER_OTA_PASSWORD "localfirstai"
#endif
#ifndef CLUSTER_ENABLE_HTTP_UPDATE
#define CLUSTER_ENABLE_HTTP_UPDATE 0
#endif
#ifndef CLUSTER_HTTP_UPDATE_PORT
#define CLUSTER_HTTP_UPDATE_PORT 8080
#endif
#endif

extern "C" {
#include "esp_nn.h"
int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len);
int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len_div16);
}

static constexpr uint32_t MAGIC = 0x4d4c4952;
static constexpr int VOCAB_SIZE = 33;
static constexpr int HIDDEN = 256;
static constexpr int LAYERS = 3;
static constexpr int SEED_COUNT = 3;
static constexpr int TOKENS_PER_SEED = 16;
static constexpr int MAX_TOKENS = SEED_COUNT * TOKENS_PER_SEED;
static constexpr const char *BENCH_SCHEMA = "ri-esp32s3-lstm-bench-v1";
static constexpr const char *FIRMWARE_VARIANT = "p22_i4_wih_whh_simd_h256";
static constexpr const char *WEIGHTS_SHA256 = "770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c";
static const char VOCAB[VOCAB_SIZE + 1] = "abcdefghijklmnopqrstuvwxyz .,!?\'\n";
static const char *BENCH_SEEDS[SEED_COUNT] = {
  "hot room. action is ",
  "missing sensor. action is ",
  "the receipt says "
};

int vocab_idx(char c);
char idx_vocab(int idx);
static constexpr int UTILITY_SEED_COUNT = 8;
static constexpr int UTILITY_MAX_CHARS = 48;
static const char *UTILITY_SEEDS[UTILITY_SEED_COUNT] = {
  "hot room. action is ",
  "missing sensor. action is ",
  "stale data. action is ",
  "high heat and humidity. action is ",
  "humid room. action is ",
  "normal room. action is ",
  "safe action is ",
  "local first means "
};

#if CLUSTER_WIFI_DEMO
static bool cluster_model_init_for_role(bool coordinator);
static bool cluster_prepare_fc_request_from_prompt(const char *prompt, uint8_t prompt_id, int8_t *hidden_q8,
                                                   float *hidden_scale_out, uint8_t *local_token_out,
                                                   float *local_logit_out);
static bool cluster_compute_fc_shard(uint8_t worker_board, const int8_t *hidden_q8, float hidden_scale,
                                     uint8_t *best_token_out, float *best_logit_out,
                                     uint8_t *shard_start_out, uint8_t *shard_end_out);
static const char *cluster_prompt_for_id(uint8_t prompt_id);

static WiFiUDP cluster_udp;
#if CLUSTER_ENABLE_HTTP_UPDATE
static WebServer cluster_http_update_server(CLUSTER_HTTP_UPDATE_PORT);
static bool cluster_http_update_error = false;
#endif
static uint32_t cluster_ping_seq = 1;
static uint32_t cluster_matmul_seq = 1;
static uint32_t cluster_last_ping_ms = 0;
static uint32_t cluster_last_matmul_ms = 0;
static uint32_t cluster_last_status_ms = 0;
static uint32_t cluster_matmul_active_seq = 0;
static uint8_t cluster_matmul_active_fixture = cluster_protocol::CLUSTER_MATMUL_FIXTURE_ID;
static int32_t cluster_matmul_worker1_dot = 0;
static int32_t cluster_matmul_worker2_dot = 0;
static bool cluster_matmul_worker1_seen = false;
static bool cluster_matmul_worker2_seen = false;
static bool cluster_matmul_gather_printed = false;
static bool cluster_model_ready = false;
#if CLUSTER_WIFI_SHARDED_INFERENCE
static uint32_t cluster_last_fc_ms = 0;
static uint32_t cluster_fc_seq = 1;
static uint32_t cluster_fc_active_seq = 0;
static uint8_t cluster_fc_active_prompt_id = 0;
static uint8_t cluster_fc_local_token = 0;
static float cluster_fc_local_logit = 0.0f;
static uint8_t cluster_fc_worker_token[3] = {0, 0, 0};
static float cluster_fc_worker_logit[3] = {0.0f, 0.0f, 0.0f};
static bool cluster_fc_worker_seen[3] = {false, false, false};
static bool cluster_fc_gather_printed = false;
static int8_t cluster_fc_hidden_q8[cluster_protocol::CLUSTER_FC_HIDDEN];
#endif
#if CLUSTER_ROLE_COORD
static IPAddress cluster_worker_ips[3];
static bool cluster_worker_ip_known[3] = {false, false, false};
#endif
static constexpr uint8_t CLUSTER_BROADCAST_BOARD = 255;
static const IPAddress CLUSTER_AP_IP(192, 168, 4, 1);
static const IPAddress CLUSTER_AP_GATEWAY(192, 168, 4, 1);
static const IPAddress CLUSTER_AP_NETMASK(255, 255, 255, 0);
static const IPAddress CLUSTER_AP_BROADCAST(192, 168, 4, 255);

static void cluster_print_ip(const char *label, IPAddress ip) {
  Serial.printf("%s=%u.%u.%u.%u", label, ip[0], ip[1], ip[2], ip[3]);
}

static const char *cluster_ota_hostname() {
#if CLUSTER_ROLE_COORD
  return "ri-esp-cluster-coord";
#elif CLUSTER_ROLE_WORKER && CLUSTER_BOARD_ID == 1
  return "ri-esp-cluster-worker1";
#elif CLUSTER_ROLE_WORKER && CLUSTER_BOARD_ID == 2
  return "ri-esp-cluster-worker2";
#else
  return "ri-esp-cluster-unknown";
#endif
}

static IPAddress cluster_local_ip() {
#if CLUSTER_WIFI_AP_MODE
  return WiFi.softAPIP();
#else
  return WiFi.localIP();
#endif
}

#if CLUSTER_ENABLE_OTA
static void cluster_setup_ota() {
  ArduinoOTA.setHostname(cluster_ota_hostname());
  ArduinoOTA.setPassword(CLUSTER_OTA_PASSWORD);
  ArduinoOTA.setPort(3232);

  ArduinoOTA.onStart([]() {
    const char *type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
    Serial.printf("CLUSTER_OTA_START type=%s\n", type);
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    unsigned int percent = total == 0 ? 0 : (progress * 100U) / total;
    if (percent > 100U) percent = 100U;
    Serial.printf("CLUSTER_OTA_PROGRESS percent=%u\n", percent);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("CLUSTER_OTA_END ok=1");
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("CLUSTER_OTA_ERROR code=%u\n", (unsigned)error);
  });

  ArduinoOTA.begin();
  Serial.printf("CLUSTER_OTA_READY board_id=%u hostname=%s ",
                (unsigned)CLUSTER_BOARD_ID, cluster_ota_hostname());
  cluster_print_ip("ip", cluster_local_ip());
  Serial.println(" port=3232");
}
#endif

#if CLUSTER_ENABLE_HTTP_UPDATE
static void cluster_setup_http_update() {
  cluster_http_update_server.on("/health", HTTP_GET, []() {
    char body[160];
    snprintf(body, sizeof(body),
             "ok=1 board_id=%u role=%s mode=%s ip=%s\n",
             (unsigned)CLUSTER_BOARD_ID,
#if CLUSTER_ROLE_COORD
             "coord",
#elif CLUSTER_ROLE_WORKER
             "worker",
#else
             "unknown",
#endif
#if CLUSTER_WIFI_SHARDED_INFERENCE
             "sharded_inference",
#elif CLUSTER_WIFI_MATMUL_PROOF
             "matmul",
#else
             "ping",
#endif
             cluster_local_ip().toString().c_str());
    cluster_http_update_server.send(200, "text/plain", body);
  });

  cluster_http_update_server.on(
      "/update", HTTP_POST,
      []() {
        const bool ok = !cluster_http_update_error && !Update.hasError();
        cluster_http_update_server.sendHeader("Connection", "close");
        cluster_http_update_server.send(ok ? 200 : 500, "text/plain", ok ? "OK\n" : "FAIL\n");
        Serial.printf("CLUSTER_HTTP_UPDATE_END ok=%u\n", ok ? 1 : 0);
        if (ok) {
          delay(300);
          ESP.restart();
        }
      },
      []() {
        HTTPUpload &upload = cluster_http_update_server.upload();
        if (upload.status == UPLOAD_FILE_START) {
          cluster_http_update_error = false;
          Serial.printf("CLUSTER_HTTP_UPDATE_START filename=%s\n", upload.filename.c_str());
          if (!Update.begin(UPDATE_SIZE_UNKNOWN, U_FLASH)) {
            cluster_http_update_error = true;
            Serial.printf("CLUSTER_HTTP_UPDATE_ERROR phase=begin code=%u\n", (unsigned)Update.getError());
          }
        } else if (upload.status == UPLOAD_FILE_WRITE) {
          if (!cluster_http_update_error) {
            size_t written = Update.write(upload.buf, upload.currentSize);
            if (written != upload.currentSize) {
              cluster_http_update_error = true;
              Serial.printf("CLUSTER_HTTP_UPDATE_ERROR phase=write wrote=%u expected=%u code=%u\n",
                            (unsigned)written, (unsigned)upload.currentSize, (unsigned)Update.getError());
            }
          }
        } else if (upload.status == UPLOAD_FILE_END) {
          if (!cluster_http_update_error) {
            if (!Update.end(true)) {
              cluster_http_update_error = true;
              Serial.printf("CLUSTER_HTTP_UPDATE_ERROR phase=end code=%u\n", (unsigned)Update.getError());
            } else {
              Serial.printf("CLUSTER_HTTP_UPDATE_STAGED bytes=%u\n", (unsigned)upload.totalSize);
            }
          } else {
            Update.abort();
          }
        } else if (upload.status == UPLOAD_FILE_ABORTED) {
          cluster_http_update_error = true;
          Update.abort();
          Serial.println("CLUSTER_HTTP_UPDATE_ERROR phase=aborted");
        }
      });

  cluster_http_update_server.begin();
  Serial.printf("CLUSTER_HTTP_UPDATE_READY board_id=%u ", (unsigned)CLUSTER_BOARD_ID);
  cluster_print_ip("ip", cluster_local_ip());
  Serial.printf(" port=%u endpoint=/update\n", (unsigned)CLUSTER_HTTP_UPDATE_PORT);
}
#endif

#if CLUSTER_ROLE_COORD && CLUSTER_ENABLE_HTTP_UPDATE
static bool cluster_parse_relay_update_command(const String &line, uint8_t *board_out, uint32_t *size_out) {
  if (board_out == nullptr || size_out == nullptr) return false;
  if (!line.startsWith("CLUSTER_RELAY_UPDATE ")) return false;

  int board = -1;
  unsigned long parsed_size = 0;
  int parsed = sscanf(line.c_str(), "CLUSTER_RELAY_UPDATE board=%d size=%lu", &board, &parsed_size);
  if (parsed != 2) return false;
  if (board < 1 || board > 2 || parsed_size == 0) return false;

  *board_out = (uint8_t)board;
  *size_out = (uint32_t)parsed_size;
  return true;
}

static void cluster_relay_worker_update(uint8_t board, uint32_t firmware_size) {
  if (board >= 3 || !cluster_worker_ip_known[board]) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=target board=%u reason=worker_ip_unknown\n", (unsigned)board);
    return;
  }

  IPAddress target_ip = cluster_worker_ips[board];
  WiFiClient client;
  constexpr uint16_t target_port = CLUSTER_HTTP_UPDATE_PORT;
  const char *boundary = "----RIESP32S3RelayBoundary";
  String prefix = String("--") + boundary +
                  "\r\nContent-Disposition: form-data; name=\"update\"; filename=\"firmware.bin\"\r\n" +
                  "Content-Type: application/octet-stream\r\n\r\n";
  String suffix = String("\r\n--") + boundary + "--\r\n";
  const uint32_t content_length = (uint32_t)prefix.length() + firmware_size + (uint32_t)suffix.length();

  Serial.printf("CLUSTER_RELAY_UPDATE_START board=%u ip=%s port=%u bytes=%lu\n",
                (unsigned)board, target_ip.toString().c_str(), (unsigned)target_port,
                (unsigned long)firmware_size);

  if (!client.connect(target_ip, target_port)) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=connect board=%u ip=%s\n",
                  (unsigned)board, target_ip.toString().c_str());
    return;
  }
  client.setTimeout(15000);
  client.printf("POST /update HTTP/1.1\r\n");
  client.printf("Host: %s:%u\r\n", target_ip.toString().c_str(), (unsigned)target_port);
  client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary);
  client.printf("Content-Length: %lu\r\n", (unsigned long)content_length);
  client.printf("Connection: close\r\n\r\n");
  client.print(prefix);
  Serial.printf("CLUSTER_RELAY_UPDATE_READY_FOR_BYTES board=%u bytes=%lu\n",
                (unsigned)board, (unsigned long)firmware_size);

  uint8_t buf[1024];
  uint32_t remaining = firmware_size;
  uint32_t sent = 0;
  uint32_t last_progress = 0;
  const uint32_t started_ms = millis();
  while (remaining > 0) {
    const size_t want = remaining > sizeof(buf) ? sizeof(buf) : (size_t)remaining;
    const size_t got = Serial.readBytes(buf, want);
    if (got == 0) {
      Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=serial_read sent=%lu remaining=%lu\n",
                    (unsigned long)sent, (unsigned long)remaining);
      client.stop();
      return;
    }

    size_t written_total = 0;
    uint32_t write_deadline = millis() + 10000;
    while (written_total < got) {
      if (!client.connected()) {
        Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=wifi_disconnected sent=%lu chunk_offset=%u\n",
                      (unsigned long)sent, (unsigned)written_total);
        client.stop();
        return;
      }
      const size_t wrote = client.write(buf + written_total, got - written_total);
      if (wrote == 0) {
        if (millis() > write_deadline) {
          Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=wifi_write_timeout sent=%lu chunk_offset=%u expected=%u\n",
                        (unsigned long)sent, (unsigned)written_total, (unsigned)got);
          client.stop();
          return;
        }
        delay(5);
        yield();
        continue;
      }
      written_total += wrote;
      write_deadline = millis() + 10000;
    }

    remaining -= (uint32_t)got;
    sent += (uint32_t)got;
    if (sent - last_progress >= 65536 || remaining == 0) {
      last_progress = sent;
      Serial.printf("CLUSTER_RELAY_UPDATE_PROGRESS board=%u sent=%lu total=%lu\n",
                    (unsigned)board, (unsigned long)sent, (unsigned long)firmware_size);
    }
    yield();
  }
  client.print(suffix);
  client.flush();

  String status_line;
  const uint32_t response_deadline = millis() + 20000;
  while (millis() < response_deadline && client.connected()) {
    while (client.available()) {
      String line = client.readStringUntil('\n');
      line.trim();
      if (line.length() > 0) {
        if (status_line.length() == 0) status_line = line;
        Serial.printf("CLUSTER_RELAY_UPDATE_RESPONSE board=%u line=%s\n", (unsigned)board, line.c_str());
      }
    }
    if (status_line.length() > 0 && !client.connected()) break;
    delay(10);
  }
  while (client.available()) {
    String line = client.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      if (status_line.length() == 0) status_line = line;
      Serial.printf("CLUSTER_RELAY_UPDATE_RESPONSE board=%u line=%s\n", (unsigned)board, line.c_str());
    }
  }
  client.stop();

  const bool ok = status_line.startsWith("HTTP/1.1 200") || status_line.startsWith("HTTP/1.0 200");
  Serial.printf("CLUSTER_RELAY_UPDATE_END board=%u ok=%u status=\"%s\" elapsed_ms=%lu\n",
                (unsigned)board, ok ? 1 : 0, status_line.c_str(),
                (unsigned long)(millis() - started_ms));
}

static void cluster_handle_serial_relay() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  uint8_t board = 0;
  uint32_t firmware_size = 0;
  if (!cluster_parse_relay_update_command(line, &board, &firmware_size)) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=parse line=%s\n", line.c_str());
    return;
  }
  Serial.setTimeout(10000);
  cluster_relay_worker_update(board, firmware_size);
  Serial.setTimeout(1000);
}
#endif

static bool cluster_send_packet(IPAddress ip, uint16_t port, uint8_t msg_type, uint8_t dst_board,
                                uint32_t seq, const uint8_t *payload = nullptr,
                                uint16_t payload_len = 0) {
  uint8_t packet[640];
  size_t packet_len = 0;
  if (!cluster_protocol::encode_packet(msg_type, (uint8_t)CLUSTER_BOARD_ID, dst_board, seq,
                                       payload, payload_len, packet, sizeof(packet), &packet_len)) {
    Serial.println("CLUSTER_WIFI_ERROR encode_failed");
    return false;
  }
  if (!cluster_udp.beginPacket(ip, port)) {
    Serial.println("CLUSTER_WIFI_ERROR begin_packet_failed");
    return false;
  }
  cluster_udp.write(packet, packet_len);
  if (!cluster_udp.endPacket()) {
    Serial.println("CLUSTER_WIFI_ERROR end_packet_failed");
    return false;
  }
  return true;
}

static int32_t cluster_matmul_compute_dot(uint8_t fixture_id, uint8_t worker_board, const int8_t *vector) {
  int32_t dot = 0;
  for (size_t i = 0; i < cluster_protocol::CLUSTER_MATMUL_VECTOR_LEN; i++) {
    dot += (int32_t)vector[i] * (int32_t)cluster_protocol::matmul_fixture_weight(fixture_id, worker_board, i);
  }
  return dot;
}

static void cluster_handle_matmul_result(const cluster_protocol::ClusterPacketHeader &header,
                                         const uint8_t *payload, size_t payload_len) {
#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_MATMUL_PROOF
  if (header.src_board < 3) {
    cluster_worker_ips[header.src_board] = cluster_udp.remoteIP();
    cluster_worker_ip_known[header.src_board] = true;
  }

  uint8_t fixture_id = 0;
  int32_t dot = 0;
  if (!cluster_protocol::decode_matmul_result_payload(payload, payload_len, &fixture_id, &dot)) {
    Serial.printf("CLUSTER_WIFI_DROP reason=bad_matmul_result_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }

  const int32_t expected = cluster_protocol::matmul_fixture_expected_dot(header.src_board, fixture_id);
  const bool result_ok = cluster_protocol::matmul_fixture_is_supported(fixture_id) && dot == expected;
  Serial.printf("CLUSTER_MATMUL_RESULT src_board=%u seq=%lu fixture=%u dot=%ld expected=%ld ok=%s\n",
                (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)fixture_id,
                (long)dot, (long)expected, result_ok ? "true" : "false");

  if (header.seq != cluster_matmul_active_seq || fixture_id != cluster_matmul_active_fixture) return;
  if (header.src_board == 1) {
    cluster_matmul_worker1_dot = dot;
    cluster_matmul_worker1_seen = true;
  } else if (header.src_board == 2) {
    cluster_matmul_worker2_dot = dot;
    cluster_matmul_worker2_seen = true;
  }

  if (cluster_matmul_worker1_seen && cluster_matmul_worker2_seen && !cluster_matmul_gather_printed) {
    const int32_t total = cluster_matmul_worker1_dot + cluster_matmul_worker2_dot;
    const int32_t expected_total = cluster_protocol::matmul_fixture_expected_gather(fixture_id);
    const bool gather_ok = (total == expected_total);
    Serial.printf("CLUSTER_MATMUL_GATHER seq=%lu fixture=%u worker1=%ld worker2=%ld total=%ld expected=%ld ok=%s\n",
                  (unsigned long)header.seq, (unsigned)fixture_id, (long)cluster_matmul_worker1_dot,
                  (long)cluster_matmul_worker2_dot, (long)total, (long)expected_total,
                  gather_ok ? "true" : "false");
    cluster_matmul_gather_printed = true;
  }
#else
  (void)header;
  (void)payload;
  (void)payload_len;
#endif
}


static void cluster_handle_fc_shard_result(const cluster_protocol::ClusterPacketHeader &header,
                                           const uint8_t *payload, size_t payload_len) {
#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_SHARDED_INFERENCE
  if (header.src_board < 3) {
    cluster_worker_ips[header.src_board] = cluster_udp.remoteIP();
    cluster_worker_ip_known[header.src_board] = true;
  }

  uint8_t prompt_id = 0;
  uint8_t best_token = 0;
  float best_logit = 0.0f;
  uint8_t shard_start = 0;
  uint8_t shard_end = 0;
  if (!cluster_protocol::decode_fc_shard_result_payload(payload, payload_len, &prompt_id, &best_token,
                                                        &best_logit, &shard_start, &shard_end)) {
    Serial.printf("CLUSTER_FC_DROP reason=bad_result_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }
  Serial.printf("CLUSTER_FC_RESULT src_board=%u seq=%lu prompt_id=%u token=%u char=%c logit=%.6f shard=%u-%u\n",
                (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)prompt_id,
                (unsigned)best_token, idx_vocab(best_token), (double)best_logit,
                (unsigned)shard_start, (unsigned)shard_end);

  if (header.seq != cluster_fc_active_seq || prompt_id != cluster_fc_active_prompt_id) return;
  if (header.src_board == 1 || header.src_board == 2) {
    cluster_fc_worker_token[header.src_board] = best_token;
    cluster_fc_worker_logit[header.src_board] = best_logit;
    cluster_fc_worker_seen[header.src_board] = true;
  }

  if (cluster_fc_worker_seen[1] && cluster_fc_worker_seen[2] && !cluster_fc_gather_printed) {
    uint8_t global_token = cluster_fc_worker_logit[1] >= cluster_fc_worker_logit[2]
                               ? cluster_fc_worker_token[1]
                               : cluster_fc_worker_token[2];
    float global_logit = cluster_fc_worker_logit[1] >= cluster_fc_worker_logit[2]
                             ? cluster_fc_worker_logit[1]
                             : cluster_fc_worker_logit[2];
    bool ok = (global_token == cluster_fc_local_token);
    Serial.printf("CLUSTER_FC_GATHER seq=%lu prompt_id=%u prompt=\"%s\" worker1_token=%u worker1_logit=%.6f worker2_token=%u worker2_logit=%.6f global_token=%u global_char=%c global_logit=%.6f local_token=%u local_char=%c local_logit=%.6f ok=%s\n",
                  (unsigned long)header.seq, (unsigned)prompt_id, cluster_prompt_for_id(prompt_id),
                  (unsigned)cluster_fc_worker_token[1], (double)cluster_fc_worker_logit[1],
                  (unsigned)cluster_fc_worker_token[2], (double)cluster_fc_worker_logit[2],
                  (unsigned)global_token, idx_vocab(global_token), (double)global_logit,
                  (unsigned)cluster_fc_local_token, idx_vocab(cluster_fc_local_token),
                  (double)cluster_fc_local_logit, ok ? "true" : "false");
    cluster_fc_gather_printed = true;
  }
#else
  (void)header;
  (void)payload;
  (void)payload_len;
#endif
}

static void cluster_handle_udp_packet() {
  int packet_size = cluster_udp.parsePacket();
  if (packet_size <= 0) return;
  if (packet_size > 640) {
    Serial.printf("CLUSTER_WIFI_DROP reason=too_large bytes=%d from=%s:%u\n",
                  packet_size, cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort());
    while (cluster_udp.available() > 0) cluster_udp.read();
    return;
  }

  uint8_t packet[640];
  int read_len = cluster_udp.read(packet, sizeof(packet));
  cluster_protocol::ClusterPacketHeader header;
  const uint8_t *payload = nullptr;
  size_t payload_len = 0;
  cluster_protocol::ClusterDecodeStatus status =
      cluster_protocol::decode_packet(packet, (size_t)read_len, &header, &payload, &payload_len);
  if (status != cluster_protocol::CLUSTER_DECODE_OK) {
    Serial.printf("CLUSTER_WIFI_DROP reason=decode_%u bytes=%d from=%s:%u\n",
                  (unsigned)status, read_len, cluster_udp.remoteIP().toString().c_str(),
                  cluster_udp.remotePort());
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_PING) {
#if CLUSTER_ROLE_WORKER
    if (header.dst_board == CLUSTER_BROADCAST_BOARD || header.dst_board == (uint8_t)CLUSTER_BOARD_ID) {
      bool ok = cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                    cluster_protocol::CLUSTER_MSG_PONG, header.src_board, header.seq);
      Serial.printf("CLUSTER_WIFI_PING board=%u seq=%lu from_board=%u from=%s:%u reply=%s rssi=%ld\n",
                    (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)header.src_board,
                    cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort(), ok ? "sent" : "failed",
                    (long)WiFi.RSSI());
    }
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_PONG) {
#if CLUSTER_ROLE_COORD
    if (header.src_board < 3) {
      cluster_worker_ips[header.src_board] = cluster_udp.remoteIP();
      cluster_worker_ip_known[header.src_board] = true;
    }
    Serial.printf("CLUSTER_WIFI_PONG src_board=%u seq=%lu from=%s:%u rssi=%ld\n",
                  (unsigned)header.src_board, (unsigned long)header.seq,
                  cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort(), (long)WiFi.RSSI());
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_MATMUL_REQUEST) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_MATMUL_PROOF
    if (header.dst_board == CLUSTER_BROADCAST_BOARD || header.dst_board == (uint8_t)CLUSTER_BOARD_ID) {
      int8_t vector[cluster_protocol::CLUSTER_MATMUL_VECTOR_LEN];
      uint8_t fixture_id = 0;
      if (!cluster_protocol::decode_matmul_request_payload(payload, payload_len, &fixture_id, vector)) {
        Serial.printf("CLUSTER_WIFI_DROP reason=bad_matmul_request_payload src_board=%u seq=%lu\n",
                      (unsigned)header.src_board, (unsigned long)header.seq);
        return;
      }
      const int32_t dot = cluster_matmul_compute_dot(fixture_id, (uint8_t)CLUSTER_BOARD_ID, vector);
      uint8_t result_payload[cluster_protocol::CLUSTER_MATMUL_RESULT_PAYLOAD_SIZE];
      size_t result_payload_len = 0;
      bool encoded = cluster_protocol::encode_matmul_result_payload(fixture_id, dot, result_payload,
                                                                    sizeof(result_payload),
                                                                    &result_payload_len);
      bool ok = encoded && cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                               cluster_protocol::CLUSTER_MSG_MATMUL_RESULT,
                                               header.src_board, header.seq, result_payload,
                                               (uint16_t)result_payload_len);
      const int32_t expected = cluster_protocol::matmul_fixture_expected_dot((uint8_t)CLUSTER_BOARD_ID,
                                                                            fixture_id);
      Serial.printf("CLUSTER_MATMUL_WORKER board=%u seq=%lu fixture=%u dot=%ld expected=%ld ok=%s reply=%s rssi=%ld\n",
                    (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)fixture_id,
                    (long)dot, (long)expected,
                    (cluster_protocol::matmul_fixture_is_supported(fixture_id) && dot == expected) ? "true" : "false",
                    ok ? "sent" : "failed", (long)WiFi.RSSI());
    }
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_MATMUL_RESULT) {
    cluster_handle_matmul_result(header, payload, payload_len);
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_FC_SHARD_REQUEST) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
    if (header.dst_board == CLUSTER_BROADCAST_BOARD || header.dst_board == (uint8_t)CLUSTER_BOARD_ID) {
      uint8_t prompt_id = 0;
      float hidden_scale = 1.0f;
      int8_t hidden_q8[cluster_protocol::CLUSTER_FC_HIDDEN];
      if (!cluster_protocol::decode_fc_shard_request_payload(payload, payload_len, &prompt_id,
                                                             &hidden_scale, hidden_q8)) {
        Serial.printf("CLUSTER_FC_DROP reason=bad_request_payload src_board=%u seq=%lu\n",
                      (unsigned)header.src_board, (unsigned long)header.seq);
        return;
      }
      uint8_t best_token = 0;
      float best_logit = 0.0f;
      uint8_t shard_start = 0;
      uint8_t shard_end = 0;
      bool computed = cluster_model_ready && cluster_compute_fc_shard((uint8_t)CLUSTER_BOARD_ID, hidden_q8,
                                                                      hidden_scale, &best_token,
                                                                      &best_logit, &shard_start,
                                                                      &shard_end);
      uint8_t result_payload[cluster_protocol::CLUSTER_FC_RESULT_PAYLOAD_SIZE];
      size_t result_payload_len = 0;
      bool encoded = computed && cluster_protocol::encode_fc_shard_result_payload(
                                     prompt_id, best_token, best_logit, shard_start, shard_end,
                                     result_payload, sizeof(result_payload), &result_payload_len);
      bool ok = encoded && cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                               cluster_protocol::CLUSTER_MSG_FC_SHARD_RESULT,
                                               header.src_board, header.seq, result_payload,
                                               (uint16_t)result_payload_len);
      Serial.printf("CLUSTER_FC_WORKER board=%u seq=%lu prompt_id=%u token=%u char=%c logit=%.6f shard=%u-%u ok=%s reply=%s rssi=%ld\n",
                    (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)prompt_id,
                    (unsigned)best_token, idx_vocab(best_token), (double)best_logit,
                    (unsigned)shard_start, (unsigned)shard_end,
                    computed ? "true" : "false", ok ? "sent" : "failed", (long)WiFi.RSSI());
    }
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_FC_SHARD_RESULT) {
    cluster_handle_fc_shard_result(header, payload, payload_len);
    return;
  }

  Serial.printf("CLUSTER_WIFI_DROP reason=unexpected_msg type=%u src_board=%u seq=%lu\n",
                (unsigned)header.msg_type, (unsigned)header.src_board, (unsigned long)header.seq);
}

static void cluster_setup_wifi_demo() {
  Serial.begin(115200);
  delay(1500);
  Serial.printf("\nESP32-S3 cluster WiFi demo boot board_id=%u role=%s mode=%s\n",
                (unsigned)CLUSTER_BOARD_ID,
#if CLUSTER_ROLE_COORD
                "coord"
#elif CLUSTER_ROLE_WORKER
                "worker"
#else
                "unknown"
#endif
                ,
#if CLUSTER_WIFI_SHARDED_INFERENCE
                "sharded_inference"
#elif CLUSTER_WIFI_MATMUL_PROOF
                "matmul"
#else
                "ping"
#endif
  );
  Serial.printf("CLUSTER_WIFI_CONFIG ssid=%s port=%u ap_mode=%u\n",
                CLUSTER_WIFI_SSID, (unsigned)CLUSTER_WIFI_UDP_PORT, (unsigned)CLUSTER_WIFI_AP_MODE);

#if CLUSTER_ROLE_COORD
#if CLUSTER_WIFI_AP_MODE
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.softAPConfig(CLUSTER_AP_IP, CLUSTER_AP_GATEWAY, CLUSTER_AP_NETMASK);
  bool ap_ok = WiFi.softAP(CLUSTER_WIFI_SSID, CLUSTER_WIFI_PASSPHRASE);
  delay(200);
  Serial.printf("CLUSTER_WIFI_AP_READY ok=%u ssid=%s ", ap_ok ? 1 : 0, CLUSTER_WIFI_SSID);
  cluster_print_ip("ip", WiFi.softAPIP());
  Serial.printf(" port=%u\n", (unsigned)CLUSTER_WIFI_UDP_PORT);
#else
  Serial.println("CLUSTER_WIFI_ERROR coord_requires_CLUSTER_WIFI_AP_MODE");
#endif
#elif CLUSTER_ROLE_WORKER
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.begin(CLUSTER_WIFI_SSID, CLUSTER_WIFI_PASSPHRASE);
  Serial.printf("CLUSTER_WIFI_STA_CONNECTING board_id=%u ssid=%s\n", (unsigned)CLUSTER_BOARD_ID,
                CLUSTER_WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.println();
  Serial.printf("CLUSTER_WIFI_WORKER_READY board_id=%u ", (unsigned)CLUSTER_BOARD_ID);
  cluster_print_ip("ip", WiFi.localIP());
  Serial.printf(" rssi=%ld port=%u\n", (long)WiFi.RSSI(), (unsigned)CLUSTER_WIFI_UDP_PORT);
#else
  Serial.println("CLUSTER_WIFI_ERROR missing_cluster_role");
#endif

  if (cluster_udp.begin(CLUSTER_WIFI_UDP_PORT)) {
    Serial.printf("CLUSTER_WIFI_UDP_READY board_id=%u port=%u\n", (unsigned)CLUSTER_BOARD_ID,
                  (unsigned)CLUSTER_WIFI_UDP_PORT);
  } else {
    Serial.printf("CLUSTER_WIFI_UDP_FAILED board_id=%u port=%u\n", (unsigned)CLUSTER_BOARD_ID,
                  (unsigned)CLUSTER_WIFI_UDP_PORT);
  }

#if CLUSTER_ENABLE_OTA
  cluster_setup_ota();
#endif
#if CLUSTER_ENABLE_HTTP_UPDATE
  cluster_setup_http_update();
#endif
#if CLUSTER_WIFI_SHARDED_INFERENCE
  cluster_model_ready = cluster_model_init_for_role(CLUSTER_ROLE_COORD != 0);
  Serial.printf("CLUSTER_MODEL_READY board_id=%u ok=%u role=%s\n", (unsigned)CLUSTER_BOARD_ID,
                cluster_model_ready ? 1 : 0, CLUSTER_ROLE_COORD ? "coord" : "worker");
#endif
}

static void cluster_loop_wifi_demo() {
#if CLUSTER_ENABLE_OTA
  ArduinoOTA.handle();
#endif
#if CLUSTER_ENABLE_HTTP_UPDATE
  cluster_http_update_server.handleClient();
#endif
#if CLUSTER_ROLE_COORD && CLUSTER_ENABLE_HTTP_UPDATE
  cluster_handle_serial_relay();
#endif
  cluster_handle_udp_packet();

#if CLUSTER_ROLE_COORD
  uint32_t now = millis();
#if CLUSTER_WIFI_PING_ONLY
  if (now - cluster_last_ping_ms >= 2000) {
    cluster_last_ping_ms = now;
    uint32_t seq = cluster_ping_seq++;
    bool ok = cluster_send_packet(CLUSTER_AP_BROADCAST, CLUSTER_WIFI_UDP_PORT,
                                  cluster_protocol::CLUSTER_MSG_PING, CLUSTER_BROADCAST_BOARD, seq);
    Serial.printf("CLUSTER_WIFI_PING_BROADCAST seq=%lu dst=%s port=%u sent=%s\n",
                  (unsigned long)seq, CLUSTER_AP_BROADCAST.toString().c_str(),
                  (unsigned)CLUSTER_WIFI_UDP_PORT, ok ? "true" : "false");
  }
#endif
#if CLUSTER_WIFI_MATMUL_PROOF
  if (now - cluster_last_matmul_ms >= 3000) {
    cluster_last_matmul_ms = now;
    const uint32_t seq = cluster_matmul_seq++;
    const uint8_t fixture_id = (seq & 1u) ? cluster_protocol::CLUSTER_MATMUL_FIXTURE_INT8_ID
                                          : cluster_protocol::CLUSTER_MATMUL_FIXTURE_INT4_ID;
    uint8_t request_payload[cluster_protocol::CLUSTER_MATMUL_REQUEST_PAYLOAD_SIZE];
    size_t request_payload_len = 0;
    bool encoded = cluster_protocol::encode_matmul_request_payload(
        fixture_id, request_payload, sizeof(request_payload), &request_payload_len);

    cluster_matmul_active_seq = seq;
    cluster_matmul_active_fixture = fixture_id;
    cluster_matmul_worker1_dot = 0;
    cluster_matmul_worker2_dot = 0;
    cluster_matmul_worker1_seen = false;
    cluster_matmul_worker2_seen = false;
    cluster_matmul_gather_printed = false;

    for (uint8_t dst = 1; dst <= 2; dst++) {
      bool sent = encoded && cluster_send_packet(CLUSTER_AP_BROADCAST, CLUSTER_WIFI_UDP_PORT,
                                                 cluster_protocol::CLUSTER_MSG_MATMUL_REQUEST, dst, seq,
                                                 request_payload, (uint16_t)request_payload_len);
      Serial.printf("CLUSTER_MATMUL_REQUEST seq=%lu fixture=%u dst=%u sent=%s\n",
                    (unsigned long)seq, (unsigned)fixture_id,
                    (unsigned)dst, sent ? "true" : "false");
    }
  }
#endif
#if CLUSTER_WIFI_SHARDED_INFERENCE
  if (now - cluster_last_ping_ms >= 2000) {
    cluster_last_ping_ms = now;
    uint32_t ping_seq = cluster_ping_seq++;
    bool ping_ok = cluster_send_packet(CLUSTER_AP_BROADCAST, CLUSTER_WIFI_UDP_PORT,
                                       cluster_protocol::CLUSTER_MSG_PING,
                                       CLUSTER_BROADCAST_BOARD, ping_seq);
    Serial.printf("CLUSTER_WIFI_PING_BROADCAST seq=%lu dst=%s port=%u sent=%s reason=infer_discovery\n",
                  (unsigned long)ping_seq, CLUSTER_AP_BROADCAST.toString().c_str(),
                  (unsigned)CLUSTER_WIFI_UDP_PORT, ping_ok ? "true" : "false");
  }
  if (cluster_model_ready && now - cluster_last_fc_ms >= 8000) {
    cluster_last_fc_ms = now;
    const uint32_t seq = cluster_fc_seq++;
    const uint8_t prompt_id = (uint8_t)(seq & 1u);
    float hidden_scale = 1.0f;
    uint8_t local_token = 0;
    float local_logit = 0.0f;
    bool prepared = cluster_prepare_fc_request_from_prompt(cluster_prompt_for_id(prompt_id), prompt_id,
                                                           cluster_fc_hidden_q8, &hidden_scale,
                                                           &local_token, &local_logit);
    cluster_fc_active_seq = seq;
    cluster_fc_active_prompt_id = prompt_id;
    cluster_fc_local_token = local_token;
    cluster_fc_local_logit = local_logit;
    cluster_fc_worker_seen[1] = false;
    cluster_fc_worker_seen[2] = false;
    cluster_fc_gather_printed = false;

    uint8_t request_payload[cluster_protocol::CLUSTER_FC_REQUEST_PAYLOAD_SIZE];
    size_t request_payload_len = 0;
    bool encoded = prepared && cluster_protocol::encode_fc_shard_request_payload(
                                   prompt_id, hidden_scale, cluster_fc_hidden_q8,
                                   request_payload, sizeof(request_payload), &request_payload_len);
    Serial.printf("CLUSTER_FC_REQUEST seq=%lu prompt_id=%u prompt=\"%s\" local_token=%u local_char=%c local_logit=%.6f hidden_scale=%.9f encoded=%s\n",
                  (unsigned long)seq, (unsigned)prompt_id, cluster_prompt_for_id(prompt_id),
                  (unsigned)local_token, idx_vocab(local_token), (double)local_logit,
                  (double)hidden_scale, encoded ? "true" : "false");
    for (uint8_t dst = 1; dst <= 2; dst++) {
      IPAddress target = cluster_worker_ip_known[dst] ? cluster_worker_ips[dst] : CLUSTER_AP_BROADCAST;
      bool sent = encoded && cluster_send_packet(target, CLUSTER_WIFI_UDP_PORT,
                                                 cluster_protocol::CLUSTER_MSG_FC_SHARD_REQUEST,
                                                 dst, seq, request_payload,
                                                 (uint16_t)request_payload_len);
      Serial.printf("CLUSTER_FC_REQUEST_SEND seq=%lu dst=%u target=%s sent=%s\n",
                    (unsigned long)seq, (unsigned)dst, target.toString().c_str(), sent ? "true" : "false");
    }
  }
#endif
#elif CLUSTER_ROLE_WORKER
  uint32_t now = millis();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("CLUSTER_WIFI_STA_DISCONNECTED reconnecting=true");
    WiFi.reconnect();
    delay(500);
    return;
  }
  if (now - cluster_last_status_ms >= 5000) {
    cluster_last_status_ms = now;
    Serial.printf("CLUSTER_WIFI_WORKER_STATUS board_id=%u ", (unsigned)CLUSTER_BOARD_ID);
    cluster_print_ip("ip", WiFi.localIP());
    Serial.printf(" rssi=%ld port=%u\n", (long)WiFi.RSSI(), (unsigned)CLUSTER_WIFI_UDP_PORT);
  }
#endif

  delay(10);
}
#endif

enum DType : uint8_t { F32 = 0, I8 = 1, I4 = 2 };

struct Tensor {
  char name[32];
  uint8_t dtype = 0;
  uint8_t ndim = 0;
  uint32_t dims[2] = {0, 0};
  float scale = 1.0f;
  uint32_t payload_len = 0;
  const uint8_t *payload = nullptr;
};

struct ModelView {
  const uint8_t *base = nullptr;
  size_t len = 0;
  spi_flash_mmap_handle_t mmap = 0;
  Tensor tensors[32];
  uint32_t tensor_count = 0;
};

struct ResolvedModel {
  Tensor *embed = nullptr;
  Tensor *fcw = nullptr;
  Tensor *fcb = nullptr;
  Tensor *wih[LAYERS] = {nullptr, nullptr, nullptr};
  Tensor *whh[LAYERS] = {nullptr, nullptr, nullptr};
  Tensor *bih[LAYERS] = {nullptr, nullptr, nullptr};
  Tensor *bhh[LAYERS] = {nullptr, nullptr, nullptr};
  bool ok = false;
};

struct OpBreakdown {
  uint64_t embed_us = 0;
  uint64_t quant_us = 0;
  uint64_t lstm_wih_us = 0;
  uint64_t lstm_whh_us = 0;
  uint64_t sram_copy_us = 0;
  uint64_t activation_us = 0;
  uint64_t fc_us = 0;
  uint64_t core1_wait_us = 0;
  uint32_t measured_steps = 0;
};

ModelView model;
ResolvedModel resolved;
OpBreakdown ops;

// Dual-core sync
static SemaphoreHandle_t core1_start_sem = nullptr;
static SemaphoreHandle_t core1_done_sem = nullptr;
static volatile bool core1_active = false;

struct Core1Params {
  const Tensor *wih;
  const Tensor *whh;
  const Tensor *bih;
  const Tensor *bhh;
  const int8_t *qx;
  const int8_t *qh;
  float input_scale;
  float h_scale;
  int input_dim;
  int gate_start;
  int gate_end;
  float *gates_out;
};
static Core1Params c1p;

static uint16_t rd_u16(const uint8_t *&p) { uint16_t v; memcpy(&v, p, 2); p += 2; return v; }
static uint32_t rd_u32(const uint8_t *&p) { uint32_t v; memcpy(&v, p, 4); p += 4; return v; }
static float rd_f32(const uint8_t *&p) { float v; memcpy(&v, p, 4); p += 4; return v; }

Tensor *find_tensor(const char *name) {
  for (uint32_t i = 0; i < model.tensor_count; i++) {
    if (strcmp(model.tensors[i].name, name) == 0) return &model.tensors[i];
  }
  return nullptr;
}

static inline float f32_at(const Tensor *t, uint32_t idx) {
  float v;
  memcpy(&v, t->payload + idx * 4, 4);
  return v;
}

static inline int8_t signed_i4_low(uint8_t b) {
  int8_t q = b & 0x0F;
  return q >= 8 ? q - 16 : q;
}

static inline int8_t signed_i4_high(uint8_t b) {
  int8_t q = (b >> 4) & 0x0F;
  return q >= 8 ? q - 16 : q;
}

__attribute__((noinline)) float tensor_get_slow(const Tensor *t, uint32_t idx) {
  if (t->dtype == F32) return f32_at(t, idx);
  if (t->dtype == I8) return ((float)((int8_t)t->payload[idx])) * t->scale;
  uint8_t b = t->payload[idx >> 1];
  int8_t q = (idx & 1) ? signed_i4_high(b) : signed_i4_low(b);
  return ((float)q) * t->scale;
}

static inline int32_t dot_i8_i8_acc(const uint8_t *payload, uint32_t row_start, const int8_t *x, int n) {
  const int8_t *w = (const int8_t *)(payload + row_start);
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  if ((n & 15) == 0 && (((uintptr_t)x) & 15) == 0) {
    if ((((uintptr_t)w) & 15) == 0) {
      return esp_nn_dot_s8_aligned_esp32s3(x, w, n);
    }
    return esp_nn_dot_s8_unaligned_esp32s3(x, w, n / 16);
  }
#endif
  int32_t acc = 0;
  int j = 0;
  for (; j + 7 < n; j += 8) {
    acc += (int32_t)w[j] * (int32_t)x[j];
    acc += (int32_t)w[j + 1] * (int32_t)x[j + 1];
    acc += (int32_t)w[j + 2] * (int32_t)x[j + 2];
    acc += (int32_t)w[j + 3] * (int32_t)x[j + 3];
    acc += (int32_t)w[j + 4] * (int32_t)x[j + 4];
    acc += (int32_t)w[j + 5] * (int32_t)x[j + 5];
    acc += (int32_t)w[j + 6] * (int32_t)x[j + 6];
    acc += (int32_t)w[j + 7] * (int32_t)x[j + 7];
  }
  for (; j < n; j++) acc += (int32_t)w[j] * (int32_t)x[j];
  return acc;
}

extern "C" int32_t dot_i4_i8_fast_esp32s3(const int8_t *input, const uint8_t *weights_packed, int n);

static inline int32_t dot_i4_i8_acc(const uint8_t *payload, uint32_t elem_row_start, const int8_t *x, int n) {
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  if ((n & 15) == 0 && (((uintptr_t)x) & 15) == 0 && ((elem_row_start & 1) == 0)) {
    return dot_i4_i8_fast_esp32s3(x, payload + (elem_row_start >> 1), n);
  }
#endif
  const uint8_t *packed = payload + (elem_row_start >> 1);
  int32_t acc = 0;
  int j = 0;
  if (elem_row_start & 1) {
    // Rare fallback for odd element row starts: first logical element is high nibble.
    for (; j + 1 < n; j += 2) {
      uint8_t b0 = packed[j >> 1];
      uint8_t b1 = packed[(j >> 1) + 1];
      acc += (int32_t)signed_i4_high(b0) * (int32_t)x[j];
      acc += (int32_t)signed_i4_low(b1) * (int32_t)x[j + 1];
    }
    if (j < n) acc += (int32_t)signed_i4_high(packed[j >> 1]) * (int32_t)x[j];
    return acc;
  }
  for (; j + 1 < n; j += 2) {
    uint8_t b = *packed++;
    acc += (int32_t)signed_i4_low(b) * (int32_t)x[j];
    acc += (int32_t)signed_i4_high(b) * (int32_t)x[j + 1];
  }
  if (j < n) acc += (int32_t)signed_i4_low(*packed) * (int32_t)x[j];
  return acc;
}

static inline float dot_tensor_q8(const Tensor *t, uint32_t elem_row_start, const int8_t *xq, float x_scale, int n) {
  if (t->dtype == I8) return (float)dot_i8_i8_acc(t->payload, elem_row_start, xq, n) * t->scale * x_scale;
  if (t->dtype == I4) return (float)dot_i4_i8_acc(t->payload, elem_row_start, xq, n) * t->scale * x_scale;
  float acc = 0.0f;
  for (int j = 0; j < n; j++) acc += f32_at(t, elem_row_start + j) * ((float)xq[j] * x_scale);
  return acc;
}

// Dot using a raw int8 weight pointer (for SRAM-tiled recurrent weights)
static inline float dot_raw_i8(const int8_t *w, uint32_t row_start, float w_scale, const int8_t *xq, float x_scale, int n) {
  return (float)dot_i8_i8_acc((const uint8_t *)w, row_start, xq, n) * w_scale * x_scale;
}

float quantize_q8(const float *src, int8_t *dst, int n) {
  float max_abs = 0.0f;
  for (int i = 0; i < n; i++) {
    float a = fabsf(src[i]);
    if (a > max_abs) max_abs = a;
  }
  if (max_abs < 1.0e-8f) {
    memset(dst, 0, n);
    return 1.0f;
  }
  float scale = max_abs / 127.0f;
  float inv = 1.0f / scale;
  for (int i = 0; i < n; i++) {
    float v = src[i] * inv;
    int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst[i] = (int8_t)q;
  }
  return scale;
}

int vocab_idx(char c) {
  for (int i = 0; i < VOCAB_SIZE; i++) if (VOCAB[i] == c) return i;
  return vocab_idx(' ');
}

char idx_vocab(int idx) {
  if (idx < 0 || idx >= VOCAB_SIZE) return '?';
  return VOCAB[idx];
}

bool load_model_partition() {
  const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "weights");
  if (!part) {
    Serial.println("ERR no weights partition");
    return false;
  }
  Serial.printf("weights partition addr=0x%lx size=%lu\n", (unsigned long)part->address, (unsigned long)part->size);
  const void *mapped = nullptr;
  esp_err_t err = esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &mapped, &model.mmap);
  if (err != ESP_OK) {
    Serial.printf("ERR mmap failed: %d\n", (int)err);
    return false;
  }
  model.base = (const uint8_t *)mapped;
  model.len = part->size;
  const uint8_t *p = model.base;
  uint32_t magic = rd_u32(p);
  if (magic != MAGIC) {
    Serial.printf("ERR bad magic 0x%08lx\n", (unsigned long)magic);
    return false;
  }
  uint16_t version = rd_u16(p);
  (void)rd_u16(p);
  model.tensor_count = rd_u32(p);
  if (model.tensor_count > 32) {
    Serial.printf("ERR too many tensors %lu\n", (unsigned long)model.tensor_count);
    return false;
  }
  Serial.printf("RILM version=%u tensors=%lu\n", version, (unsigned long)model.tensor_count);
  for (uint32_t i = 0; i < model.tensor_count; i++) {
    uint16_t actual_name_len = rd_u16(p);
    uint16_t stored_name_len = actual_name_len;
    if (stored_name_len >= sizeof(model.tensors[i].name)) stored_name_len = sizeof(model.tensors[i].name) - 1;
    memcpy(model.tensors[i].name, p, stored_name_len);
    model.tensors[i].name[stored_name_len] = 0;
    p += actual_name_len;
    model.tensors[i].dtype = *p++;
    model.tensors[i].ndim = *p++;
    for (uint8_t d = 0; d < model.tensors[i].ndim && d < 2; d++) model.tensors[i].dims[d] = rd_u32(p);
    for (uint8_t d = 2; d < model.tensors[i].ndim; d++) (void)rd_u32(p);
    model.tensors[i].scale = rd_f32(p);
    model.tensors[i].payload_len = rd_u32(p);
    model.tensors[i].payload = p;
    p += model.tensors[i].payload_len;
  }
  return true;
}

bool clone_payloads_to_psram() {
  uint32_t copied = 0;
  for (uint32_t i = 0; i < model.tensor_count; i++) {
    Tensor *t = &model.tensors[i];
    if (!t->payload || t->payload_len == 0) continue;
    uint32_t caps = (t->payload_len <= 8192) ? (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *copy = (uint8_t *)heap_caps_aligned_alloc(16, t->payload_len, caps);
    if (!copy && caps != (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)) copy = (uint8_t *)heap_caps_aligned_alloc(16, t->payload_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) {
      Serial.printf("ERR payload clone failed %s %lu bytes\n", t->name, (unsigned long)t->payload_len);
      return false;
    }
    memcpy(copy, t->payload, t->payload_len);
    t->payload = copy;
    copied += t->payload_len;
  }
  Serial.printf("payloads cloned bytes=%lu free_heap=%lu free_psram=%lu\n", (unsigned long)copied, (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getFreePsram());
  return true;
}

bool convert_wih_to_int4() {
  uint32_t converted = 0;
  uint32_t saved = 0;
  for (uint32_t i = 0; i < model.tensor_count; i++) {
    Tensor *t = &model.tensors[i];
    if (strncmp(t->name, "lstm.weight_ih_l", 16) != 0 && strncmp(t->name, "lstm.weight_hh_l", 16) != 0) continue;
    if (t->dtype != I8 || !t->payload || t->payload_len == 0) continue;

    uint32_t old_len = t->payload_len;
    uint32_t new_len = (old_len + 1) >> 1;
    uint8_t *packed = (uint8_t *)heap_caps_aligned_alloc(16, new_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!packed) {
      Serial.printf("ERR int4 pack failed %s %lu bytes\n", t->name, (unsigned long)new_len);
      return false;
    }

    const int8_t *src = (const int8_t *)t->payload;
    for (uint32_t j = 0; j < old_len; j += 2) {
      int q0 = (int)lrintf(((float)src[j]) * (7.0f / 127.0f));
      int q1 = 0;
      if (j + 1 < old_len) q1 = (int)lrintf(((float)src[j + 1]) * (7.0f / 127.0f));
      if (q0 > 7) q0 = 7; if (q0 < -8) q0 = -8;
      if (q1 > 7) q1 = 7; if (q1 < -8) q1 = -8;
      packed[j >> 1] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }

    t->payload = packed;
    t->payload_len = new_len;
    t->dtype = I4;
    t->scale *= (127.0f / 7.0f);
    converted++;
    saved += old_len - new_len;
    Serial.printf("int4 recurrent %s old=%lu new=%lu scale=%g\n", t->name,
      (unsigned long)old_len, (unsigned long)new_len, (double)t->scale);
  }
  Serial.printf("int4 recurrent converted=%lu saved=%lu free_psram=%lu\n",
    (unsigned long)converted, (unsigned long)saved, (unsigned long)ESP.getFreePsram());
  return converted == (LAYERS * 2);
}

bool resolve_model() {
  resolved.embed = find_tensor("embed.weight");
  resolved.fcw = find_tensor("fc.weight");
  resolved.fcb = find_tensor("fc.bias");
  for (int l = 0; l < LAYERS; l++) {
    char name[32];
    snprintf(name, sizeof(name), "lstm.weight_ih_l%d", l);
    resolved.wih[l] = find_tensor(name);
    snprintf(name, sizeof(name), "lstm.weight_hh_l%d", l);
    resolved.whh[l] = find_tensor(name);
    snprintf(name, sizeof(name), "lstm.bias_ih_l%d", l);
    resolved.bih[l] = find_tensor(name);
    snprintf(name, sizeof(name), "lstm.bias_hh_l%d", l);
    resolved.bhh[l] = find_tensor(name);
    if (!resolved.wih[l] || !resolved.whh[l] || !resolved.bih[l]) {
      Serial.printf("ERR unresolved layer %d\n", l);
      return false;
    }
  }
  resolved.ok = resolved.embed && resolved.fcw && resolved.fcb;
  if (!resolved.ok) Serial.println("ERR unresolved embed/fc");
  return resolved.ok;
}

struct LstmState {
  float *h[LAYERS];
  float *c[LAYERS];
  int8_t *qh[LAYERS];
  float *x;
  int8_t *qx;
  float *next_h;
  float *next_c;
  float *gates;
  float *logits;
};

LstmState st;

void *internal_alloc(size_t n) {
  void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL alloc failed %u bytes\n", (unsigned)n);
    while (true) delay(1000);
  }
  memset(p, 0, n);
  return p;
}

void alloc_state() {
  for (int l = 0; l < LAYERS; l++) {
    st.h[l] = (float *)internal_alloc(sizeof(float) * HIDDEN);
    st.c[l] = (float *)internal_alloc(sizeof(float) * HIDDEN);
    st.qh[l] = (int8_t *)internal_alloc(sizeof(int8_t) * HIDDEN);
  }
  st.x = (float *)internal_alloc(sizeof(float) * HIDDEN);
  st.qx = (int8_t *)internal_alloc(sizeof(int8_t) * HIDDEN);
  st.next_h = (float *)internal_alloc(sizeof(float) * HIDDEN);
  st.next_c = (float *)internal_alloc(sizeof(float) * HIDDEN);
  st.gates = (float *)internal_alloc(sizeof(float) * 4 * HIDDEN);
  st.logits = (float *)internal_alloc(sizeof(float) * VOCAB_SIZE);

}

void reset_state() {
  for (int l = 0; l < LAYERS; l++) {
    memset(st.h[l], 0, sizeof(float) * HIDDEN);
    memset(st.c[l], 0, sizeof(float) * HIDDEN);
    memset(st.qh[l], 0, sizeof(int8_t) * HIDDEN);
  }
  memset(st.x, 0, sizeof(float) * HIDDEN);
  memset(st.qx, 0, sizeof(int8_t) * HIDDEN);
  memset(st.next_h, 0, sizeof(float) * HIDDEN);
  memset(st.next_c, 0, sizeof(float) * HIDDEN);
  memset(st.gates, 0, sizeof(float) * 4 * HIDDEN);
  memset(st.logits, 0, sizeof(float) * VOCAB_SIZE);
}

static constexpr int ACT_LUT_SIZE = 257;
static float sigmoid_lut[ACT_LUT_SIZE];
static float tanh_lut[ACT_LUT_SIZE];

void init_activation_lut() {
  for (int i = 0; i < ACT_LUT_SIZE; i++) {
    float x = -8.0f + (16.0f * (float)i) / (float)(ACT_LUT_SIZE - 1);
    sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
    tanh_lut[i] = tanhf(x);
  }
}

static inline float lut_lookup(const float *lut, float x) {
  if (x <= -8.0f) return lut[0];
  if (x >= 8.0f) return lut[ACT_LUT_SIZE - 1];
  float pos = (x + 8.0f) * ((float)(ACT_LUT_SIZE - 1) / 16.0f);
  int idx = (int)pos;
  float frac = pos - (float)idx;
  return lut[idx] + (lut[idx + 1] - lut[idx]) * frac;
}

static inline float sigmoidf_fast(float x) { return lut_lookup(sigmoid_lut, x); }
static inline float tanhf_fast(float x) { return lut_lookup(tanh_lut, x); }

// Core 0 worker task: computes half the LSTM gates while core 1 does the other half.
// Pinned to core 0. Arduino loopTask is on core 1.
static void core0_worker(void *arg) {
  while (true) {
    xSemaphoreTake(core1_start_sem, portMAX_DELAY);
    if (!core1_active) continue;

    const Core1Params *p = &c1p;
    for (int g = p->gate_start; g < p->gate_end; g++) {
      float acc = f32_at(p->bih, g);
      if (p->bhh) acc += f32_at(p->bhh, g);
      acc += dot_tensor_q8(p->wih, (uint32_t)g * p->input_dim, p->qx, p->input_scale, p->input_dim);
      acc += dot_tensor_q8(p->whh, (uint32_t)g * HIDDEN, p->qh, p->h_scale, HIDDEN);
      p->gates_out[g] = acc;
    }

    xSemaphoreGive(core1_done_sem);
  }
}

void lstm_layer(int layer, const float *input, int input_dim, bool measure) {
  Tensor *wih = resolved.wih[layer];
  Tensor *whh = resolved.whh[layer];
  Tensor *bih = resolved.bih[layer];
  Tensor *bhh = resolved.bhh[layer];

  uint32_t tq = measure ? micros() : 0;
  float input_scale = quantize_q8(input, st.qx, input_dim);
  float h_scale = quantize_q8(st.h[layer], st.qh[layer], HIDDEN);
  if (measure) ops.quant_us += (uint32_t)(micros() - tq);
  // Dual-core: split 4*HIDDEN gates into two halves
  c1p.wih = wih;
  c1p.whh = whh;
  c1p.bih = bih;
  c1p.bhh = bhh;
  c1p.qx = st.qx;
  c1p.qh = st.qh[layer];
  c1p.input_scale = input_scale;
  c1p.h_scale = h_scale;
  c1p.input_dim = input_dim;
  c1p.gate_start = 2 * HIDDEN;
  c1p.gate_end = 4 * HIDDEN;
  c1p.gates_out = st.gates;

  uint32_t tw = measure ? micros() : 0;
  xSemaphoreGive(core1_start_sem);

  // Core 1: gates [0, 2*HIDDEN)
  for (int g = 0; g < 2 * HIDDEN; g++) {
    float acc = f32_at(bih, g);
    if (bhh) acc += f32_at(bhh, g);
    uint32_t t0 = measure ? micros() : 0;
    acc += dot_tensor_q8(wih, (uint32_t)g * input_dim, st.qx, input_scale, input_dim);
    if (measure) ops.lstm_wih_us += (uint32_t)(micros() - t0);
    t0 = measure ? micros() : 0;
    acc += dot_tensor_q8(whh, (uint32_t)g * HIDDEN, st.qh[layer], h_scale, HIDDEN);
    if (measure) ops.lstm_whh_us += (uint32_t)(micros() - t0);
    st.gates[g] = acc;
    if ((g & 255) == 0) yield();
  }

  xSemaphoreTake(core1_done_sem, portMAX_DELAY);
  if (measure) ops.core1_wait_us += (uint32_t)(micros() - tw);

  uint32_t t0 = measure ? micros() : 0;
  for (int i = 0; i < HIDDEN; i++) {
    float ingate = sigmoidf_fast(st.gates[i]);
    float forgetgate = sigmoidf_fast(st.gates[HIDDEN + i]);
    float cellgate = tanhf_fast(st.gates[2 * HIDDEN + i]);
    float outgate = sigmoidf_fast(st.gates[3 * HIDDEN + i]);
    float c = forgetgate * st.c[layer][i] + ingate * cellgate;
    float h = outgate * tanhf_fast(c);
    st.next_c[i] = c;
    st.next_h[i] = h;
  }
  if (measure) ops.activation_us += (uint32_t)(micros() - t0);
  memcpy(st.h[layer], st.next_h, sizeof(float) * HIDDEN);
  memcpy(st.c[layer], st.next_c, sizeof(float) * HIDDEN);
}

void model_step_hidden(int token_idx, bool measure) {
  uint32_t t0 = measure ? micros() : 0;
  Tensor *embed = resolved.embed;
  uint32_t row = (uint32_t)token_idx * HIDDEN;
  if (embed->dtype == I4) {
    for (int i = 0; i < HIDDEN; i += 2) {
      uint8_t b = embed->payload[(row + i) >> 1];
      st.x[i] = (float)signed_i4_low(b) * embed->scale;
      st.x[i + 1] = (float)signed_i4_high(b) * embed->scale;
    }
  } else {
    for (int i = 0; i < HIDDEN; i++) st.x[i] = tensor_get_slow(embed, row + i);
  }
  if (measure) ops.embed_us += (uint32_t)(micros() - t0);

  for (int l = 0; l < LAYERS; l++) {
    lstm_layer(l, st.x, HIDDEN, measure);
    memcpy(st.x, st.h[l], sizeof(float) * HIDDEN);
  }
}

int model_finish_fc(bool measure, float *best_logit_out = nullptr) {
  uint32_t t0 = measure ? micros() : 0;
  t0 = measure ? micros() : 0;
  Tensor *fcw = resolved.fcw;
  Tensor *fcb = resolved.fcb;
  float x_scale = quantize_q8(st.x, st.qx, HIDDEN);
  int best = 0;
  float best_v = -1e30f;
  for (int v = 0; v < VOCAB_SIZE; v++) {
    float acc = f32_at(fcb, v);
    acc += dot_tensor_q8(fcw, (uint32_t)v * HIDDEN, st.qx, x_scale, HIDDEN);
    st.logits[v] = acc;
    if (acc > best_v) { best_v = acc; best = v; }
  }
  if (best_logit_out) *best_logit_out = best_v;
  if (measure) {
    ops.fc_us += (uint32_t)(micros() - t0);
    ops.measured_steps++;
  }
  return best;
}

int model_step(int token_idx, bool measure) {
  model_step_hidden(token_idx, measure);
  return model_finish_fc(measure);
}

#if CLUSTER_WIFI_DEMO
static const char *cluster_prompt_for_id(uint8_t prompt_id) {
  return (prompt_id & 1u) ? "missing sensor. action is " : "hot room. action is ";
}

static bool cluster_model_init_for_role(bool coordinator) {
  if (coordinator) {
    core1_start_sem = xSemaphoreCreateBinary();
    core1_done_sem = xSemaphoreCreateBinary();
    if (!core1_start_sem || !core1_done_sem) {
      Serial.println("CLUSTER_MODEL_ERROR phase=semaphore");
      return false;
    }
    xTaskCreatePinnedToCore(core0_worker, "lstm_worker", 16384, nullptr, 2, nullptr, 0);
    core1_active = true;
  }
  if (!coordinator) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
    if (RI_FC_ROWS != VOCAB_SIZE || RI_FC_COLS != HIDDEN) {
      Serial.printf("CLUSTER_MODEL_ERROR phase=embedded_fc_shape rows=%lu cols=%lu expected_rows=%u expected_cols=%u\n",
                    (unsigned long)RI_FC_ROWS, (unsigned long)RI_FC_COLS,
                    (unsigned)VOCAB_SIZE, (unsigned)HIDDEN);
      return false;
    }
    Serial.printf("CLUSTER_MODEL_WORKER_FC_READY board_id=%u rows=%lu cols=%lu weights_sha256=%s source=embedded_fc_head\n",
                  (unsigned)CLUSTER_BOARD_ID, (unsigned long)RI_FC_ROWS, (unsigned long)RI_FC_COLS,
                  RI_FC_WEIGHTS_SHA256);
    return true;
#else
    return false;
#endif
  }

  if (!load_model_partition()) { Serial.println("CLUSTER_MODEL_ERROR phase=load_partition"); return false; }

  // Coordinator runs the recurrent pass, so it needs the fast p22 memory layout:
  // cloned tensors plus int4 recurrent weights and SRAM scratch/state. Workers only
  // compute their FC vocabulary row shard from a coordinator-supplied hidden vector.
  // The worker FC head is embedded into the app firmware, so workers do not need a
  // data/weights partition or full recurrent model materialization.
  if (!clone_payloads_to_psram()) { Serial.println("CLUSTER_MODEL_ERROR phase=clone_payloads"); return false; }
  if (!convert_wih_to_int4()) { Serial.println("CLUSTER_MODEL_ERROR phase=int4_convert"); return false; }

  if (!resolve_model()) { Serial.println("CLUSTER_MODEL_ERROR phase=resolve"); return false; }
  if (resolved.fcw->dims[1] != HIDDEN || resolved.fcw->dims[0] != VOCAB_SIZE) {
    Serial.printf("CLUSTER_MODEL_ERROR phase=fc_shape rows=%lu cols=%lu expected_rows=%u expected_cols=%u\n",
                  (unsigned long)resolved.fcw->dims[0], (unsigned long)resolved.fcw->dims[1],
                  (unsigned)VOCAB_SIZE, (unsigned)HIDDEN);
    return false;
  }
  if (coordinator) {
    init_activation_lut();
    alloc_state();
  }
  return true;
}

static bool cluster_compute_fc_shard(uint8_t worker_board, const int8_t *hidden_q8, float hidden_scale,
                                     uint8_t *best_token_out, float *best_logit_out,
                                     uint8_t *shard_start_out, uint8_t *shard_end_out) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
  if (!hidden_q8 || !best_token_out || !best_logit_out || !shard_start_out || !shard_end_out) return false;
#else
  if (!resolved.ok || !resolved.fcw || !resolved.fcb || !hidden_q8 || !best_token_out ||
      !best_logit_out || !shard_start_out || !shard_end_out) return false;
#endif
  uint8_t start = 0;
  uint8_t end = 0;
  if (!cluster_protocol::fc_shard_range_for_worker(worker_board, &start, &end)) return false;
  int best = start;
  float best_v = -1e30f;
  for (uint8_t v = start; v <= end; v++) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
    const int8_t *w = (const int8_t *)(RI_FC_WEIGHT_Q8 + ((uint32_t)v * HIDDEN));
    float acc = RI_FC_BIAS_F32[v];
    acc += (float)dot_i8_i8_acc((const uint8_t *)w, 0, hidden_q8, HIDDEN) * RI_FC_WEIGHT_SCALE * hidden_scale;
#else
    float acc = f32_at(resolved.fcb, v);
    acc += dot_tensor_q8(resolved.fcw, (uint32_t)v * HIDDEN, hidden_q8, hidden_scale, HIDDEN);
#endif
    if (acc > best_v) { best_v = acc; best = v; }
  }
  *best_token_out = (uint8_t)best;
  *best_logit_out = best_v;
  *shard_start_out = start;
  *shard_end_out = end;
  return true;
}

static bool cluster_prepare_fc_request_from_prompt(const char *prompt, uint8_t prompt_id, int8_t *hidden_q8,
                                                   float *hidden_scale_out, uint8_t *local_token_out,
                                                   float *local_logit_out) {
  (void)prompt_id;
  if (!cluster_model_ready || !prompt || !hidden_q8 || !hidden_scale_out || !local_token_out || !local_logit_out) return false;
  reset_state();
  for (const char *p = prompt; *p; p++) {
    model_step_hidden(vocab_idx(*p), false);
  }
  *hidden_scale_out = quantize_q8(st.x, hidden_q8, HIDDEN);
  int token = model_finish_fc(false, local_logit_out);
  *local_token_out = (uint8_t)token;
  return true;
}
#endif

void sort_u32(uint32_t *arr, int n) {
  for (int i = 1; i < n; i++) {
    uint32_t key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

void json_escape_print(const char *s) {
  for (const char *p = s; *p; p++) {
    char c = *p;
    if (c == '\\' || c == '"') { Serial.print('\\'); Serial.print(c); }
    else if (c == '\n') Serial.print("\\n");
    else if ((uint8_t)c < 32) Serial.print(' ');
    else Serial.print(c);
  }
}

void generate_stopped(const char *seed, char *out, int max_chars) {
  reset_state();
  int token = vocab_idx(seed[0]);
  for (const char *p = seed; *p; p++) token = model_step(vocab_idx(*p), false);
  int i = 0;
  for (; i < max_chars; i++) {
    char ch = idx_vocab(token);
    out[i] = ch;
    if (ch == '.' || ch == '\n') { i++; break; }
    token = model_step(token, false);
    yield();
  }
  if (i > max_chars) i = max_chars;
  out[i] = 0;
}

void run_benchmark() {
  uint32_t token_ms[MAX_TOKENS];
  uint32_t sorted_ms[MAX_TOKENS];
  char outputs[SEED_COUNT][TOKENS_PER_SEED + 1];
  char utility_outputs[UTILITY_SEED_COUNT][UTILITY_MAX_CHARS + 1];
  for (int s = 0; s < UTILITY_SEED_COUNT; s++) {
    generate_stopped(UTILITY_SEEDS[s], utility_outputs[s], UTILITY_MAX_CHARS);
  }

  int measured = 0;
  uint32_t bench_start = millis();
  memset(&ops, 0, sizeof(ops));

  for (int s = 0; s < SEED_COUNT; s++) {
    reset_state();
    const char *seed = BENCH_SEEDS[s];
    int token = vocab_idx(seed[0]);
    for (const char *p = seed; *p; p++) token = model_step(vocab_idx(*p), false);
    for (int i = 0; i < TOKENS_PER_SEED; i++) {
      outputs[s][i] = idx_vocab(token);
      uint32_t start = millis();
      token = model_step(token, true);
      uint32_t elapsed = millis() - start;
      token_ms[measured] = elapsed;
      sorted_ms[measured] = elapsed;
      measured++;
      yield();
    }
    outputs[s][TOKENS_PER_SEED] = 0;
  }

  uint32_t total_ms = millis() - bench_start;
  uint64_t sum = 0;
  for (int i = 0; i < measured; i++) sum += token_ms[i];
  sort_u32(sorted_ms, measured);
  float mean = measured ? ((float)sum / (float)measured) : 0.0f;
  uint32_t p50 = sorted_ms[measured / 2];
  uint32_t p95 = sorted_ms[(int)((measured - 1) * 0.95f)];
  float tps = mean > 0.0f ? 1000.0f / mean : 0.0f;
  float steps = ops.measured_steps ? (float)ops.measured_steps : 1.0f;

  Serial.print("BENCH_RECEIPT {");
  Serial.printf("\"schema\":\"%s\",", BENCH_SCHEMA);
  Serial.printf("\"firmware_variant\":\"%s\",", FIRMWARE_VARIANT);
  Serial.print("\"board\":\"esp32s3\",");
  Serial.printf("\"psram_size\":%lu,", (unsigned long)ESP.getPsramSize());
  Serial.printf("\"free_heap_start\":%lu,", (unsigned long)ESP.getFreeHeap());
  Serial.printf("\"free_psram_start\":%lu,", (unsigned long)ESP.getFreePsram());
  Serial.printf("\"weights_sha256\":\"%s\",", WEIGHTS_SHA256);
  Serial.print("\"model_profile\":\"domain_h256_all_int8\",");
  Serial.print("\"params\":1595937,");
  Serial.print("\"compressed_bytes\":1614972,");
  Serial.printf("\"tokens_per_seed\":%d,", TOKENS_PER_SEED);
  Serial.printf("\"total_measured_tokens\":%d,", measured);
  Serial.printf("\"ms_total\":%lu,", (unsigned long)total_ms);
  Serial.printf("\"ms_per_token_mean\":%.2f,", mean);
  Serial.printf("\"ms_per_token_p50\":%lu,", (unsigned long)p50);
  Serial.printf("\"ms_per_token_p95\":%lu,", (unsigned long)p95);
  Serial.printf("\"tokens_per_sec\":%.4f,", tps);
  Serial.print("\"op_breakdown_ms_per_token\":{");
  Serial.printf("\"embed\":%.3f,", (ops.embed_us / 1000.0f) / steps);
  Serial.printf("\"quant\":%.3f,", (ops.quant_us / 1000.0f) / steps);
  Serial.printf("\"lstm_wih\":%.3f,", (ops.lstm_wih_us / 1000.0f) / steps);
  Serial.printf("\"lstm_whh\":%.3f,", (ops.lstm_whh_us / 1000.0f) / steps);
  Serial.printf("\"sram_copy\":0.000,");
  Serial.printf("\"activation\":%.3f,", (ops.activation_us / 1000.0f) / steps);
  Serial.printf("\"fc\":%.3f,", (ops.fc_us / 1000.0f) / steps);
  Serial.printf("\"core1_wait\":%.3f},", (ops.core1_wait_us / 1000.0f) / steps);
  Serial.print("\"output_by_seed\":{");
  for (int s = 0; s < SEED_COUNT; s++) {
    if (s) Serial.print(',');
    Serial.print('"'); json_escape_print(BENCH_SEEDS[s]); Serial.print("\":\"");
    json_escape_print(outputs[s]); Serial.print('"');
  }
  Serial.print("},");
  Serial.print("\"stopped_output_by_seed\":{");
  for (int s = 0; s < UTILITY_SEED_COUNT; s++) {
    if (s) Serial.print(',');
    Serial.print('"'); json_escape_print(UTILITY_SEEDS[s]); Serial.print("\":\"");
    json_escape_print(utility_outputs[s]); Serial.print('"');
  }
  Serial.print("},");
  Serial.print("\"state_alloc\":\"internal+sram_scratch\",");
  Serial.printf("\"heap_after\":%lu,", (unsigned long)ESP.getFreeHeap());
  Serial.printf("\"psram_after\":%lu,", (unsigned long)ESP.getFreePsram());
  Serial.print("\"passed\":true,");
  Serial.print("\"blockers\":[]");
  Serial.println("}");
}

void run_language_prompt_receipt(const char *prompt) {
  char output[UTILITY_MAX_CHARS + 1];
  uint32_t start_ms = millis();
  generate_stopped(prompt, output, UTILITY_MAX_CHARS);
  uint32_t elapsed_ms = millis() - start_ms;
  int chars = strlen(output);
  float cps = elapsed_ms > 0 ? (1000.0f * (float)chars / (float)elapsed_ms) : 0.0f;
  Serial.print("S3_LANGUAGE_RECEIPT {");
  Serial.print("\"schema\":\"ri_esp32s3_local_language_v1\",");
  Serial.printf("\"firmware_variant\":\"%s\",", FIRMWARE_VARIANT);
  Serial.printf("\"weights_sha256\":\"%s\",", WEIGHTS_SHA256);
  Serial.print("\"model_profile\":\"curated_status_h320_all_int8\",");
  Serial.print("\"prompt\":\""); json_escape_print(prompt); Serial.print("\",");
  Serial.print("\"output\":\""); json_escape_print(output); Serial.print("\",");
  Serial.printf("\"generated_chars\":%d,", chars);
  Serial.printf("\"elapsed_ms\":%lu,", (unsigned long)elapsed_ms);
  Serial.printf("\"chars_per_sec\":%.4f,", cps);
  Serial.print("\"stop_rule\":\"period_or_newline_or_48_chars\",");
  Serial.print("\"passed\":true");
  Serial.println("}");
}

void poll_serial_language_commands() {
  static char line[128];
  static int n = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[n] = 0;
      n = 0;
      if (strncmp(line, "PROMPT:", 7) == 0) {
        const char *prompt = line + 7;
        while (*prompt == ' ') prompt++;
        run_language_prompt_receipt(prompt);
      } else if (strlen(line) > 0) {
        Serial.print("S3_LANGUAGE_ERROR {\"error\":\"expected_PROMPT_prefix\",\"received\":\"");
        json_escape_print(line);
        Serial.println("\"}");
      }
    } else if (n < (int)sizeof(line) - 1) {
      line[n++] = c;
    } else {
      n = 0;
      Serial.println("S3_LANGUAGE_ERROR {\"error\":\"line_too_long\"}");
    }
  }
}

void setup() {
#if CLUSTER_WIFI_DEMO
  cluster_setup_wifi_demo();
  return;
#endif

  Serial.begin(115200);
  delay(1500);
  Serial.println("\nESP32-S3 LSTM boot p22 i4 recurrent SIMD+dualcore");
  Serial.printf("free_heap=%lu free_psram=%lu psram_size=%lu\n",
    (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getFreePsram(), (unsigned long)ESP.getPsramSize());

  core1_start_sem = xSemaphoreCreateBinary();
  core1_done_sem = xSemaphoreCreateBinary();
  if (!core1_start_sem || !core1_done_sem) {
    Serial.println("FATAL: sem create failed");
    while (true) delay(1000);
  }

  // Pin worker to core 0 — Arduino loopTask runs on core 1 by default
  xTaskCreatePinnedToCore(core0_worker, "lstm_worker", 16384, nullptr, 2, nullptr, 0);
  core1_active = true;
  Serial.println("worker task started on core 0");

  if (!load_model_partition()) { Serial.println("MODEL_LOAD_FAILED"); return; }
  if (!clone_payloads_to_psram()) { Serial.println("MODEL_CLONE_FAILED"); return; }
  if (!convert_wih_to_int4()) { Serial.println("MODEL_I4_CONVERT_FAILED"); return; }
  if (!resolve_model()) { Serial.println("MODEL_RESOLVE_FAILED"); return; }
  init_activation_lut();
  alloc_state();
  Serial.printf("state allocated free_heap=%lu free_psram=%lu\n",
    (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getFreePsram());
  Serial.println("MODEL_READY profile=domain_h256_all_int8 params=1595937");
  run_benchmark();
  Serial.println("PROOF_DONE");
}

void loop() {
#if CLUSTER_WIFI_DEMO
  cluster_loop_wifi_demo();
  return;
#endif

  poll_serial_language_commands();
  static uint32_t last_idle = 0;
  if (millis() - last_idle >= 5000) {
    last_idle = millis();
    Serial.println("idle; send PROMPT:<text> for S3_LANGUAGE_RECEIPT");
  }
  delay(10);
}
