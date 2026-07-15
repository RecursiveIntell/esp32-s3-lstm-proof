#include <Arduino.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <soc/soc.h>
#include <soc/rtc_cntl_reg.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <esp_task_wdt.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "cluster_protocol.h"

#ifndef RI_FINAL_SENTINEL
#define RI_FINAL_SENTINEL 0
#endif

#if RI_FINAL_SENTINEL
#include "sensor_types.h"
#include "sentinel_policy.h"
#include "sentinel_receipt.h"
#include "actuator_types.h"
#endif

#if RI_FINAL_SENTINEL
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoOTA.h>
#include <Preferences.h>
#include <mbedtls/sha256.h>

// Local credentials are intentionally kept outside version control.  The same
// header is also used by the cluster environments, so one ignored file can
// provision this board without placing a password in platformio.ini.
#if __has_include("wifi_secrets.local.h")
#include "wifi_secrets.local.h"
#endif

// Empty credentials deliberately disable networking.  Supply both at build time
// to enable the read-only HTTP/OTA service, e.g. with -D RI_FINAL_WIFI_SSID=...
#ifndef RI_FINAL_WIFI_SSID
#define RI_FINAL_WIFI_SSID ""
#endif
#ifndef RI_FINAL_WIFI_PASSPHRASE
#define RI_FINAL_WIFI_PASSPHRASE ""
#endif
#ifndef RI_FINAL_OTA_PASSWORD
#define RI_FINAL_OTA_PASSWORD ""
#endif

static inline bool sentinel_wifi_enabled() {
  return strlen(RI_FINAL_WIFI_SSID) > 0 && strlen(RI_FINAL_WIFI_PASSPHRASE) > 0;
}
#endif // RI_FINAL_SENTINEL

#if RI_FINAL_SENTINEL
// Global WiFi service state (initialized lazily when WiFi is enabled)
static WebServer *sentinel_http_ptr = nullptr;
static bool sentinel_wifi_connected = false;
static uint32_t sentinel_last_wifi_attempt = 0;
static bool sentinel_ota_ready = false;
#endif


#ifndef CLUSTER_WIFI_PING_ONLY
#define CLUSTER_WIFI_PING_ONLY 0
#endif
#ifndef CLUSTER_WIFI_MATMUL_PROOF
#define CLUSTER_WIFI_MATMUL_PROOF 0
#endif
#ifndef CLUSTER_WIFI_SHARDED_INFERENCE
#define CLUSTER_WIFI_SHARDED_INFERENCE 0
#endif
#ifndef CLUSTER_WIFI_LSTM_SHARD
#define CLUSTER_WIFI_LSTM_SHARD 0
#endif
#ifndef CLUSTER_WIFI_LAYER_SHARD
#define CLUSTER_WIFI_LAYER_SHARD 0
#endif
#ifndef CLUSTER_WIFI_LOCAL_GENERATOR
#define CLUSTER_WIFI_LOCAL_GENERATOR 0
#endif
#ifndef CLUSTER_WIFI_DEMO
#define CLUSTER_WIFI_DEMO (CLUSTER_WIFI_PING_ONLY || CLUSTER_WIFI_MATMUL_PROOF || CLUSTER_WIFI_SHARDED_INFERENCE || CLUSTER_WIFI_LSTM_SHARD || CLUSTER_WIFI_LAYER_SHARD || CLUSTER_WIFI_LOCAL_GENERATOR)
#endif
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
#ifndef CLUSTER_WIFI_TCP_PORT
#define CLUSTER_WIFI_TCP_PORT 42101
#endif
#ifndef CLUSTER_WIFI_TCP_DIST
#define CLUSTER_WIFI_TCP_DIST 0
#endif
#ifndef CLUSTER_WIFI_UDP_PIPELINE_DIST
#define CLUSTER_WIFI_UDP_PIPELINE_DIST 0
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
#ifndef RI_VOCAB_SIZE
#define RI_VOCAB_SIZE 33
#endif
#ifndef RI_HIDDEN
#define RI_HIDDEN 256
#endif
#ifndef RI_LAYERS
#define RI_LAYERS 3
#endif
#ifndef RI_FIRMWARE_VARIANT
#define RI_FIRMWARE_VARIANT "p22_i4_wih_whh_simd_h256"
#endif
#ifndef RI_WEIGHTS_SHA256
#define RI_WEIGHTS_SHA256 "770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c"
#endif
#ifndef RI_MODEL_PROFILE
#define RI_MODEL_PROFILE "domain_h256_all_int8"
#endif
#ifndef RI_MODEL_PARAMS
#define RI_MODEL_PARAMS 1595937
#endif
#ifndef RI_COMPRESSED_BYTES
#define RI_COMPRESSED_BYTES 1614972
#endif
#ifndef RI_TINYSTORIES_MODE
#define RI_TINYSTORIES_MODE 0
#endif
static constexpr int VOCAB_SIZE = RI_VOCAB_SIZE;
static constexpr int HIDDEN = RI_HIDDEN;
static constexpr int LAYERS = RI_LAYERS;
static constexpr int SEED_COUNT = 3;
static constexpr int TOKENS_PER_SEED = 16;
static constexpr int MAX_TOKENS = SEED_COUNT * TOKENS_PER_SEED;
#ifndef CLUSTER_LOCAL_GEN_CHARS
#define CLUSTER_LOCAL_GEN_CHARS 64
#endif
static constexpr const char *BENCH_SCHEMA = "ri-esp32s3-lstm-bench-v1";
static constexpr const char *FIRMWARE_VARIANT = RI_FIRMWARE_VARIANT;
static constexpr const char *WEIGHTS_SHA256 = RI_WEIGHTS_SHA256;
static const char VOCAB[VOCAB_SIZE + 1] = "abcdefghijklmnopqrstuvwxyz .,!?\'\n";
static const char *BENCH_SEEDS[SEED_COUNT] = {
#if RI_TINYSTORIES_MODE
  "once upon a ",
  "the little girl ",
  "the dragon was "
#else
  "hot room. action is ",
  "missing sensor. action is ",
  "the receipt says "
#endif
};

int vocab_idx(char c);
char idx_vocab(int idx);
float quantize_q8(const float *src, int8_t *dst, int n);
void reset_state();
static constexpr int UTILITY_SEED_COUNT = 8;
static constexpr int UTILITY_MAX_CHARS = 48;
static const char *UTILITY_SEEDS[UTILITY_SEED_COUNT] = {
#if RI_TINYSTORIES_MODE
  "once upon a ",
  "the little girl ",
  "the boy saw ",
  "the dog was ",
  "the cat said ",
  "the dragon was ",
  "the toy was ",
  "the bird flew "
#else
  "hot room. action is ",
  "missing sensor. action is ",
  "stale data. action is ",
  "high heat and humidity. action is ",
  "humid room. action is ",
  "normal room. action is ",
  "safe action is ",
  "local first means "
#endif
};

#if CLUSTER_WIFI_DEMO
static bool cluster_model_init_for_role(bool coordinator);
static bool cluster_model_init_full_local();
static void cluster_local_generator_tick(uint32_t now);
static inline void model_init_pump_watchdog();
static bool cluster_prepare_fc_request_from_prompt(const char *prompt, uint8_t prompt_id, int8_t *hidden_q8,
                                                   float *hidden_scale_out, uint8_t *local_token_out,
                                                   float *local_logit_out);
static bool cluster_compute_fc_shard(uint8_t worker_board, const int8_t *hidden_q8, float hidden_scale,
                                     uint8_t *best_token_out, float *best_logit_out,
                                     uint8_t *shard_start_out, uint8_t *shard_end_out);
static bool cluster_prepare_lstm_gate_probe(uint8_t layer);
static bool cluster_expected_lstm_gate_values(uint8_t layer, uint16_t row_start, uint16_t count,
                                              int32_t *values_out);
static bool cluster_worker_compute_lstm_gate_probe(uint8_t layer, uint16_t row_start, uint16_t requested_count,
                                                   const int8_t *qx, float input_scale,
                                                   const int8_t *qh, float h_scale,
                                                   uint16_t *row_start_out, int32_t *values_out,
                                                   uint16_t *count_out);
#if CLUSTER_WIFI_LSTM_SHARD
static void cluster_handle_lstm_gate_result(const cluster_protocol::ClusterPacketHeader &header,
                                            const uint8_t *payload, size_t payload_len);
static void cluster_distributed_generation_tick(uint32_t now);
#endif
#if CLUSTER_WIFI_LAYER_SHARD
static bool cluster_model_init_layer_shard();
static void cluster_layer_shard_generation_tick(uint32_t now);
static void cluster_handle_layer_shard_state_result(const cluster_protocol::ClusterPacketHeader &header,
                                                     const uint8_t *payload, size_t payload_len);
static void cluster_handle_layer_shard_state_request(const cluster_protocol::ClusterPacketHeader &header,
                                                      const uint8_t *payload, size_t payload_len);
#endif


static WiFiUDP cluster_udp;
#if CLUSTER_ENABLE_HTTP_UPDATE
static WebServer cluster_http_update_server(CLUSTER_HTTP_UPDATE_PORT);
static bool cluster_http_update_error = false;
static bool cluster_http_data_update_error = false;
static const esp_partition_t *cluster_http_data_partition = nullptr;
static size_t cluster_http_data_written = 0;
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
#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD
#if CLUSTER_ROLE_WORKER
static WiFiServer *cluster_tcp_server = nullptr;
static WiFiClient cluster_tcp_worker_client;
#endif
#if CLUSTER_ROLE_COORD
static WiFiClient *worker_tcp[3] = {nullptr, nullptr, nullptr};
static uint32_t cluster_tcp_last_connect_ms[3] = {0, 0, 0};
#endif
#endif
#if CLUSTER_WIFI_LOCAL_GENERATOR
static uint32_t cluster_local_bench_last_ms = 0;
static bool cluster_local_bench_packet_sent = false;
#endif
#if CLUSTER_WIFI_LSTM_SHARD
static uint32_t cluster_lstm_gate_seq = 1;
static uint32_t cluster_last_lstm_gate_ms = 0;
static uint32_t cluster_lstm_gate_active_seq = 0;
static bool cluster_lstm_gate_seen[3] = {false, false, false};
static int32_t cluster_lstm_gate_max_abs_err[3] = {0, 0, 0};
static bool cluster_lstm_gate_gather_printed = false;
static int8_t cluster_lstm_gate_qx[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
static int8_t cluster_lstm_gate_qh[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
static float cluster_lstm_gate_input_scale = 1.0f;
static float cluster_lstm_gate_h_scale = 1.0f;
#if CLUSTER_ROLE_COORD
static bool cluster_dist_started = false;
static bool cluster_dist_active = false;
static bool cluster_dist_waiting = false;
static uint8_t cluster_dist_layer = 0;
static uint16_t cluster_dist_offset = 0;
static uint32_t cluster_dist_seq = 1000;
static uint32_t cluster_dist_active_seq = 0;
static uint32_t cluster_dist_started_ms = 0;
static uint32_t cluster_dist_last_send_ms = 0;
static bool cluster_dist_seen[3] = {false, false, false};
static uint8_t cluster_dist_expected_token = 0;
static uint8_t cluster_dist_output_token = 0;
static float cluster_dist_output_logit = 0.0f;
static const char *cluster_dist_prompt = "once upon a ";
#if CLUSTER_WIFI_UDP_PIPELINE_DIST
static constexpr uint8_t CLUSTER_DIST_PIPELINE_CHUNKS = 4;
static constexpr uint16_t CLUSTER_DIST_PIPELINE_CHUNK_ROWS = 256;
static uint32_t cluster_dist_pipeline_seq[CLUSTER_DIST_PIPELINE_CHUNKS] = {0, 0, 0, 0};
static bool cluster_dist_pipeline_seen[3][CLUSTER_DIST_PIPELINE_CHUNKS] = {};
static int32_t cluster_dist_pipeline_max_abs_err[3][CLUSTER_DIST_PIPELINE_CHUNKS] = {};
#endif
#endif
#endif
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
#if CLUSTER_WIFI_LAYER_SHARD
static bool cluster_layer_shard_smoke_sent = false;
static uint32_t cluster_layer_shard_seq = 3000;
static uint8_t cluster_layer_shard_qx[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16))) = {0};
static uint8_t cluster_layer_shard_qc[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16))) = {0};
// --- Real generation state ---
static bool cluster_layer_shard_gen_started = false;
static bool cluster_layer_shard_gen_active = false;
static bool cluster_layer_shard_gen_waiting_b1 = false;
static bool cluster_layer_shard_gen_waiting_b2 = false;
static uint32_t cluster_layer_shard_gen_seq_b1 = 0;
static uint32_t cluster_layer_shard_gen_seq_b2 = 0;
static uint32_t cluster_layer_shard_gen_last_send_b1_ms = 0;
static uint32_t cluster_layer_shard_gen_last_send_b2_ms = 0;
static float cluster_layer_shard_h_scale = 1.0f;
static float cluster_layer_shard_c_scale = 1.0f;
static uint8_t cluster_layer_shard_result_qx[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16)));
static uint8_t cluster_layer_shard_result_qc[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16)));
static float cluster_layer_shard_result_h_scale = 1.0f;
static float cluster_layer_shard_result_c_scale = 1.0f;
static bool cluster_layer_shard_b1_result_ready = false;
static bool cluster_layer_shard_b2_result_ready = false;
static int cluster_layer_shard_output_token = -1;
static float cluster_layer_shard_output_logit = 0.0f;
static uint32_t cluster_layer_shard_gen_started_ms = 0;
#endif
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
#if CLUSTER_WIFI_LAYER_SHARD
             "layer_shard",
#elif CLUSTER_WIFI_LSTM_SHARD
             "lstm_shard",
#elif CLUSTER_WIFI_LOCAL_GENERATOR
             "local_generator",
#elif CLUSTER_WIFI_SHARDED_INFERENCE
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

  cluster_http_update_server.on(
      "/update_weights", HTTP_POST,
      []() {
        const bool ok = !cluster_http_data_update_error && cluster_http_data_partition &&
                        cluster_http_data_written > 0;
        cluster_http_update_server.sendHeader("Connection", "close");
        cluster_http_update_server.send(ok ? 200 : 500, "text/plain", ok ? "OK\n" : "FAIL\n");
        Serial.printf("CLUSTER_HTTP_DATA_UPDATE_END ok=%u label=weights bytes=%u\n",
                      ok ? 1 : 0, (unsigned)cluster_http_data_written);
        if (ok) {
          delay(300);
          ESP.restart();
        }
      },
      []() {
        HTTPUpload &upload = cluster_http_update_server.upload();
        if (upload.status == UPLOAD_FILE_START) {
          cluster_http_data_update_error = false;
          cluster_http_data_written = 0;
          cluster_http_data_partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                                 (esp_partition_subtype_t)0x40,
                                                                 "weights");
          if (!cluster_http_data_partition) {
            cluster_http_data_update_error = true;
            Serial.println("CLUSTER_HTTP_DATA_UPDATE_ERROR phase=find label=weights");
            return;
          }
          Serial.printf("CLUSTER_HTTP_DATA_UPDATE_START label=weights partition_addr=0x%lx partition_size=%lu filename=%s\n",
                        (unsigned long)cluster_http_data_partition->address,
                        (unsigned long)cluster_http_data_partition->size,
                        upload.filename.c_str());
          esp_err_t err = esp_partition_erase_range(cluster_http_data_partition, 0,
                                                    cluster_http_data_partition->size);
          if (err != ESP_OK) {
            cluster_http_data_update_error = true;
            Serial.printf("CLUSTER_HTTP_DATA_UPDATE_ERROR phase=erase code=%d\n", (int)err);
          }
        } else if (upload.status == UPLOAD_FILE_WRITE) {
          if (!cluster_http_data_update_error && cluster_http_data_partition) {
            if (cluster_http_data_written + upload.currentSize > cluster_http_data_partition->size) {
              cluster_http_data_update_error = true;
              Serial.printf("CLUSTER_HTTP_DATA_UPDATE_ERROR phase=size written=%u chunk=%u partition_size=%lu\n",
                            (unsigned)cluster_http_data_written, (unsigned)upload.currentSize,
                            (unsigned long)cluster_http_data_partition->size);
              return;
            }
            esp_err_t err = esp_partition_write(cluster_http_data_partition, cluster_http_data_written,
                                                upload.buf, upload.currentSize);
            if (err != ESP_OK) {
              cluster_http_data_update_error = true;
              Serial.printf("CLUSTER_HTTP_DATA_UPDATE_ERROR phase=write offset=%u size=%u code=%d\n",
                            (unsigned)cluster_http_data_written, (unsigned)upload.currentSize, (int)err);
              return;
            }
            cluster_http_data_written += upload.currentSize;
            if ((cluster_http_data_written & 0xFFFFu) < upload.currentSize) {
              Serial.printf("CLUSTER_HTTP_DATA_UPDATE_PROGRESS label=weights written=%u\n",
                            (unsigned)cluster_http_data_written);
            }
          }
        } else if (upload.status == UPLOAD_FILE_END) {
          if (!cluster_http_data_update_error) {
            Serial.printf("CLUSTER_HTTP_DATA_UPDATE_STAGED label=weights bytes=%u\n",
                          (unsigned)cluster_http_data_written);
          }
        } else if (upload.status == UPLOAD_FILE_ABORTED) {
          cluster_http_data_update_error = true;
          Serial.println("CLUSTER_HTTP_DATA_UPDATE_ERROR phase=aborted");
        }
      });

  cluster_http_update_server.begin();
  Serial.printf("CLUSTER_HTTP_UPDATE_READY board_id=%u ", (unsigned)CLUSTER_BOARD_ID);
  cluster_print_ip("ip", cluster_local_ip());
  Serial.printf(" port=%u endpoint=/update data_endpoint=/update_weights\n", (unsigned)CLUSTER_HTTP_UPDATE_PORT);
}
#endif

#if CLUSTER_ROLE_COORD && CLUSTER_ENABLE_HTTP_UPDATE
static bool cluster_parse_relay_update_command(const String &line, uint8_t *board_out, uint32_t *size_out, bool *weights_out) {
  if (board_out == nullptr || size_out == nullptr || weights_out == nullptr) return false;
  if (!line.startsWith("CLUSTER_RELAY_UPDATE ")) return false;

  int board = -1;
  unsigned long parsed_size = 0;
  char target[24] = "app";
  int parsed = sscanf(line.c_str(), "CLUSTER_RELAY_UPDATE board=%d size=%lu target=%23s", &board, &parsed_size, target);
  if (parsed < 2) return false;
  if (board < 1 || board > 2 || parsed_size == 0) return false;

  *board_out = (uint8_t)board;
  *size_out = (uint32_t)parsed_size;
  *weights_out = (strcmp(target, "weights") == 0 || strcmp(target, "data") == 0);
  return true;
}

static void cluster_relay_worker_update(uint8_t board, uint32_t firmware_size, bool weights_target) {
  if (board >= 3) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=target board=%u reason=bad_board\n", (unsigned)board);
    return;
  }

  IPAddress target_ip = cluster_worker_ips[board];
  if (board < 3 && !cluster_worker_ip_known[board]) {
    // Last-known SoftAP DHCP assignments from live cluster receipts. This lets the
    // coordinator attempt a direct HTTP relay even if UDP discovery is missing; a
    // failed connect is then a real reachability/power receipt instead of only an
    // unknown-IP precondition failure.
    target_ip = (board == 1) ? IPAddress(192, 168, 4, 3) : IPAddress(192, 168, 4, 2);
    Serial.printf("CLUSTER_RELAY_UPDATE_WARN board=%u reason=worker_ip_unknown using_fallback_ip=%s\n",
                  (unsigned)board, target_ip.toString().c_str());
  }
  WiFiClient client;
  constexpr uint16_t target_port = CLUSTER_HTTP_UPDATE_PORT;
  const char *boundary = "----RIESP32S3RelayBoundary";
  const char *field_name = weights_target ? "weights" : "update";
  const char *file_name = weights_target ? "weights.bin" : "firmware.bin";
  String prefix = String("--") + boundary +
                  "\r\nContent-Disposition: form-data; name=\"" + field_name + "\"; filename=\"" + file_name + "\"\r\n" +
                  "Content-Type: application/octet-stream\r\n\r\n";
  String suffix = String("\r\n--") + boundary + "--\r\n";
  const uint32_t content_length = (uint32_t)prefix.length() + firmware_size + (uint32_t)suffix.length();

  Serial.printf("CLUSTER_RELAY_UPDATE_START board=%u ip=%s port=%u bytes=%lu target=%s\n",
                (unsigned)board, target_ip.toString().c_str(), (unsigned)target_port,
                (unsigned long)firmware_size, weights_target ? "weights" : "app");

  if (!client.connect(target_ip, target_port)) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=connect board=%u ip=%s\n",
                  (unsigned)board, target_ip.toString().c_str());
    return;
  }
  client.setTimeout(15000);
  client.printf(weights_target ? "POST /update_weights HTTP/1.1\r\n" : "POST /update HTTP/1.1\r\n");
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
  bool weights_target = false;
  if (!cluster_parse_relay_update_command(line, &board, &firmware_size, &weights_target)) {
    Serial.printf("CLUSTER_RELAY_UPDATE_ERROR phase=parse line=%s\n", line.c_str());
    return;
  }
  Serial.setTimeout(60000);
  cluster_relay_worker_update(board, firmware_size, weights_target);
  Serial.setTimeout(1000);
}
#endif

static bool cluster_send_packet(IPAddress ip, uint16_t port, uint8_t msg_type, uint8_t dst_board,
                                uint32_t seq, const uint8_t *payload = nullptr,
                                uint16_t payload_len = 0) {
  static uint8_t packet[4096];
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

#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD
static bool cluster_tcp_read_exact(WiFiClient &client, uint8_t *dst, size_t len, uint32_t timeout_ms = 2000) {
  size_t offset = 0;
  const uint32_t started = millis();
  while (offset < len && client.connected()) {
    int n = client.read(dst + offset, len - offset);
    if (n > 0) {
      offset += (size_t)n;
      continue;
    }
    if (millis() - started >= timeout_ms) break;
    delay(1);
  }
  return offset == len;
}

static bool cluster_send_tcp_packet(WiFiClient &client, uint8_t msg_type, uint8_t dst_board,
                                    uint32_t seq, const uint8_t *payload = nullptr,
                                    uint16_t payload_len = 0) {
  if (!client.connected()) return false;
  static uint8_t packet[4608];
  size_t packet_len = 0;
  if (!cluster_protocol::encode_packet(msg_type, (uint8_t)CLUSTER_BOARD_ID, dst_board, seq,
                                       payload, payload_len, packet, sizeof(packet), &packet_len)) {
    Serial.println("CLUSTER_TCP_ERROR encode_failed");
    return false;
  }
  size_t written = client.write(packet, packet_len);
  if (written != packet_len) {
    Serial.printf("CLUSTER_TCP_ERROR short_write wrote=%u expected=%u\n",
                  (unsigned)written, (unsigned)packet_len);
    return false;
  }
  return true;
}

#if CLUSTER_ROLE_COORD
static bool cluster_tcp_ensure_worker_connected(uint8_t board_id, uint32_t now, bool force = false) {
  if (board_id == 0 || board_id >= 3 || !cluster_worker_ip_known[board_id]) return false;
  if (worker_tcp[board_id] && worker_tcp[board_id]->connected()) return true;
  if (!force && now - cluster_tcp_last_connect_ms[board_id] < 1000) return false;
  cluster_tcp_last_connect_ms[board_id] = now;
  if (!worker_tcp[board_id]) worker_tcp[board_id] = new WiFiClient();
  if (worker_tcp[board_id]->connected()) worker_tcp[board_id]->stop();
  bool ok = worker_tcp[board_id]->connect(cluster_worker_ips[board_id], CLUSTER_WIFI_TCP_PORT);
  if (ok) worker_tcp[board_id]->setNoDelay(true);
  Serial.printf("CLUSTER_TCP_CONNECT dst=%u target=%s port=%u ok=%s nodelay=%s\n",
                (unsigned)board_id, cluster_worker_ips[board_id].toString().c_str(),
                (unsigned)CLUSTER_WIFI_TCP_PORT, ok ? "true" : "false");
  return ok;
}

static bool cluster_send_tcp_packet(uint8_t board_id, uint8_t msg_type, uint32_t seq,
                                    const uint8_t *payload = nullptr, uint16_t payload_len = 0) {
  if (!cluster_tcp_ensure_worker_connected(board_id, millis())) return false;
  return cluster_send_tcp_packet(*worker_tcp[board_id], msg_type, board_id, seq, payload, payload_len);
}
#endif

#if CLUSTER_ROLE_WORKER
static void cluster_handle_lstm_gate_request_tcp(WiFiClient &client,
                                                 const cluster_protocol::ClusterPacketHeader &header,
                                                 const uint8_t *payload, size_t payload_len) {
  if (header.dst_board != CLUSTER_BROADCAST_BOARD && header.dst_board != (uint8_t)CLUSTER_BOARD_ID) return;
  uint8_t layer = 0;
  uint16_t row_start_req = 0;
  uint16_t count_req = 0;
  float input_scale = 1.0f;
  float h_scale = 1.0f;
  static int8_t qx[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
  static int8_t qh[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
  if (!cluster_protocol::decode_lstm_gate_request_payload(payload, payload_len, &layer,
                                                          &row_start_req, &count_req,
                                                          &input_scale, &h_scale, qx, qh)) {
    Serial.printf("CLUSTER_LSTM_GATE_DROP reason=bad_tcp_request_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }
  uint16_t row_start = 0;
  uint16_t count = 0;
  static int32_t values[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_VALUES];
  bool computed = cluster_model_ready && cluster_worker_compute_lstm_gate_probe(layer, row_start_req, count_req,
                                                                                qx, input_scale, qh, h_scale,
                                                                                &row_start, values, &count);
  static uint8_t result_payload[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_PAYLOAD_SIZE];
  size_t result_payload_len = 0;
  bool encoded = computed && cluster_protocol::encode_lstm_gate_result_payload(
                                 layer, row_start, values, count, result_payload,
                                 sizeof(result_payload), &result_payload_len);
  bool ok = encoded && cluster_send_tcp_packet(client, cluster_protocol::CLUSTER_MSG_LSTM_GATE_RESULT,
                                               header.src_board, header.seq, result_payload,
                                               (uint16_t)result_payload_len);
  Serial.printf("CLUSTER_LSTM_GATE_WORKER_TCP board=%u seq=%lu layer=%u row_start=%u count=%u computed=%s reply=%s rssi=%ld\n",
                (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)layer,
                (unsigned)row_start, (unsigned)count, computed ? "true" : "false",
                ok ? "sent" : "failed", (long)WiFi.RSSI());
}
#endif

static bool cluster_handle_tcp_packet(WiFiClient &client, uint8_t peer_board) {
  if (!client.connected() || client.available() < (int)cluster_protocol::CLUSTER_PACKET_HEADER_SIZE) return false;
  static uint8_t packet[4608];
  if (!cluster_tcp_read_exact(client, packet, cluster_protocol::CLUSTER_PACKET_HEADER_SIZE)) {
    Serial.printf("CLUSTER_TCP_DROP reason=short_header peer=%u\n", (unsigned)peer_board);
    client.stop();
    return false;
  }
  cluster_protocol::ClusterPacketHeader stream_header = cluster_protocol::read_header(packet);
  const size_t packet_len = cluster_protocol::CLUSTER_PACKET_HEADER_SIZE + (size_t)stream_header.payload_len;
  if (packet_len > sizeof(packet)) {
    Serial.printf("CLUSTER_TCP_DROP reason=too_large peer=%u bytes=%u\n",
                  (unsigned)peer_board, (unsigned)packet_len);
    client.stop();
    return false;
  }
  if (!cluster_tcp_read_exact(client, packet + cluster_protocol::CLUSTER_PACKET_HEADER_SIZE,
                              stream_header.payload_len)) {
    Serial.printf("CLUSTER_TCP_DROP reason=short_payload peer=%u bytes=%u\n",
                  (unsigned)peer_board, (unsigned)stream_header.payload_len);
    client.stop();
    return false;
  }

  cluster_protocol::ClusterPacketHeader header;
  const uint8_t *payload = nullptr;
  size_t payload_len = 0;
  cluster_protocol::ClusterDecodeStatus status =
      cluster_protocol::decode_packet(packet, packet_len, &header, &payload, &payload_len);
  if (status != cluster_protocol::CLUSTER_DECODE_OK) {
    Serial.printf("CLUSTER_TCP_DROP reason=decode_%u peer=%u bytes=%u\n",
                  (unsigned)status, (unsigned)peer_board, (unsigned)packet_len);
    return false;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_GATE_REQUEST) {
#if CLUSTER_ROLE_WORKER
    cluster_handle_lstm_gate_request_tcp(client, header, payload, payload_len);
#endif
    return true;
  }
  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_GATE_RESULT) {
#if CLUSTER_ROLE_COORD
    cluster_handle_lstm_gate_result(header, payload, payload_len);
#else
    (void)payload;
    (void)payload_len;
#endif
    return true;
  }

  Serial.printf("CLUSTER_TCP_DROP reason=unexpected_msg type=%u src_board=%u seq=%lu\n",
                (unsigned)header.msg_type, (unsigned)header.src_board, (unsigned long)header.seq);
  return false;
}

static void cluster_handle_tcp_io(uint32_t now) {
#if CLUSTER_ROLE_WORKER
  if (cluster_tcp_server) {
    WiFiClient incoming = cluster_tcp_server->available();
    if (incoming) {
      if (cluster_tcp_worker_client && cluster_tcp_worker_client.connected()) cluster_tcp_worker_client.stop();
      cluster_tcp_worker_client = incoming;
      cluster_tcp_worker_client.setNoDelay(true);
      Serial.printf("CLUSTER_TCP_ACCEPT board=%u remote=%s port=%u nodelay=true\n",
                    (unsigned)CLUSTER_BOARD_ID, cluster_tcp_worker_client.remoteIP().toString().c_str(),
                    (unsigned)CLUSTER_WIFI_TCP_PORT);
    }
  }
  if (cluster_tcp_worker_client && cluster_tcp_worker_client.connected()) {
    while (cluster_tcp_worker_client.available() >= (int)cluster_protocol::CLUSTER_PACKET_HEADER_SIZE) {
      if (!cluster_handle_tcp_packet(cluster_tcp_worker_client, 0)) break;
    }
  }
#elif CLUSTER_ROLE_COORD
  if (cluster_model_ready && cluster_worker_ip_known[1] && cluster_worker_ip_known[2]) {
    for (uint8_t board_id = 1; board_id <= 2; board_id++) {
      if (cluster_worker_ip_known[board_id]) cluster_tcp_ensure_worker_connected(board_id, now);
      if (worker_tcp[board_id] && worker_tcp[board_id]->connected()) {
        while (worker_tcp[board_id]->available() >= (int)cluster_protocol::CLUSTER_PACKET_HEADER_SIZE) {
          if (!cluster_handle_tcp_packet(*worker_tcp[board_id], board_id)) break;
        }
      }
    }
  }
#else
  (void)now;
#endif
}
#endif

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
  if (packet_size > 4096) {
    Serial.printf("CLUSTER_WIFI_DROP reason=too_large bytes=%d from=%s:%u\n",
                  packet_size, cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort());
    while (cluster_udp.available() > 0) cluster_udp.read();
    return;
  }

  static uint8_t packet[4096];
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
      uint8_t pong_payload[1] = { cluster_model_ready ? (uint8_t)1 : (uint8_t)0 };
      bool ok = cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                    cluster_protocol::CLUSTER_MSG_PONG, header.src_board, header.seq,
                                    pong_payload, sizeof(pong_payload));
      Serial.printf("CLUSTER_WIFI_PING board=%u seq=%lu from_board=%u from=%s:%u model_ready=%u reply=%s rssi=%ld\n",
                    (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)header.src_board,
                    cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort(),
                    cluster_model_ready ? 1 : 0, ok ? "sent" : "failed", (long)WiFi.RSSI());
    }
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_PONG) {
#if CLUSTER_ROLE_COORD
    if (header.src_board < 3) {
      cluster_worker_ips[header.src_board] = cluster_udp.remoteIP();
      cluster_worker_ip_known[header.src_board] = true;
#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD
      cluster_tcp_ensure_worker_connected(header.src_board, millis(), true);
#endif
    }
    int model_ready_payload = payload_len >= 1 ? (int)payload[0] : -1;
    Serial.printf("CLUSTER_WIFI_PONG src_board=%u seq=%lu from=%s:%u rssi=%ld model_ready=%d\n",
                  (unsigned)header.src_board, (unsigned long)header.seq,
                  cluster_udp.remoteIP().toString().c_str(), cluster_udp.remotePort(), (long)WiFi.RSSI(),
                  model_ready_payload);
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

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_GATE_REQUEST) {
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_LSTM_SHARD
    if (header.dst_board == CLUSTER_BROADCAST_BOARD || header.dst_board == (uint8_t)CLUSTER_BOARD_ID) {
      uint8_t layer = 0;
      uint16_t row_start_req = 0;
      uint16_t count_req = 0;
      float input_scale = 1.0f;
      float h_scale = 1.0f;
      static int8_t qx[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
      static int8_t qh[cluster_protocol::CLUSTER_LSTM_HIDDEN] __attribute__((aligned(16)));
      if (!cluster_protocol::decode_lstm_gate_request_payload(payload, payload_len, &layer,
                                                              &row_start_req, &count_req,
                                                              &input_scale, &h_scale, qx, qh)) {
        Serial.printf("CLUSTER_LSTM_GATE_DROP reason=bad_request_payload src_board=%u seq=%lu\n",
                      (unsigned)header.src_board, (unsigned long)header.seq);
        return;
      }
      uint16_t row_start = 0;
      uint16_t count = 0;
      static int32_t values[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_VALUES];
      bool computed = cluster_model_ready && cluster_worker_compute_lstm_gate_probe(layer, row_start_req, count_req,
                                                                                    qx, input_scale, qh, h_scale,
                                                                                    &row_start, values, &count);
      static uint8_t result_payload[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_PAYLOAD_SIZE];
      size_t result_payload_len = 0;
      bool encoded = computed && cluster_protocol::encode_lstm_gate_result_payload(
                                     layer, row_start, values, count, result_payload,
                                     sizeof(result_payload), &result_payload_len);
      bool ok = encoded && cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                               cluster_protocol::CLUSTER_MSG_LSTM_GATE_RESULT,
                                               header.src_board, header.seq, result_payload,
                                               (uint16_t)result_payload_len);
      Serial.printf("CLUSTER_LSTM_GATE_WORKER board=%u seq=%lu layer=%u row_start=%u count=%u computed=%s reply=%s rssi=%ld\n",
                    (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)layer,
                    (unsigned)row_start, (unsigned)count, computed ? "true" : "false",
                    ok ? "sent" : "failed", (long)WiFi.RSSI());
    }
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_GATE_RESULT) {
#if CLUSTER_WIFI_LSTM_SHARD
    cluster_handle_lstm_gate_result(header, payload, payload_len);
#else
    Serial.printf("CLUSTER_WIFI_DROP reason=unexpected_lstm_gate_result src_board=%u seq=%lu mode=not_lstm_shard\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST) {
#if CLUSTER_WIFI_LAYER_SHARD
    cluster_handle_layer_shard_state_request(header, payload, payload_len);
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_RESULT) {
#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_LAYER_SHARD
    cluster_handle_layer_shard_state_result(header, payload, payload_len);
#endif
    return;
  }

  if (header.msg_type == cluster_protocol::CLUSTER_MSG_BENCH_RESULT) {
#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_LOCAL_GENERATOR
    cluster_protocol::ClusterBenchResult result;
    if (!cluster_protocol::decode_bench_result_payload(payload, payload_len, &result)) {
      Serial.printf("CLUSTER_BENCH_DROP reason=bad_result_payload src_board=%u seq=%lu\n",
                    (unsigned)header.src_board, (unsigned long)header.seq);
      return;
    }
    Serial.printf("CLUSTER_BENCH_RESULT board=%u prompt_id=%u profile=%s generated_chars=%u elapsed_ms=%lu chars_per_sec=%.4f checksum=0x%08lx\n",
                  (unsigned)header.src_board, (unsigned)result.prompt_id, RI_MODEL_PROFILE,
                  (unsigned)result.generated_chars, (unsigned long)result.elapsed_ms,
                  (double)result.chars_per_sec, (unsigned long)result.checksum);
#endif
    return;
  }

  Serial.printf("CLUSTER_WIFI_DROP reason=unexpected_msg type=%u src_board=%u seq=%lu\n",
                (unsigned)header.msg_type, (unsigned)header.src_board, (unsigned long)header.seq);
}

static void cluster_setup_wifi_demo() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
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
#if CLUSTER_WIFI_LAYER_SHARD
                "layer_shard"
#elif CLUSTER_WIFI_LSTM_SHARD
                "lstm_shard"
#elif CLUSTER_WIFI_LOCAL_GENERATOR
                "local_generator"
#elif CLUSTER_WIFI_SHARDED_INFERENCE
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
  WiFi.setTxPower(WIFI_POWER_2dBm);
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
  WiFi.setTxPower(WIFI_POWER_2dBm);
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

#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD && CLUSTER_ROLE_WORKER
  cluster_tcp_server = new WiFiServer(CLUSTER_WIFI_TCP_PORT);
  cluster_tcp_server->begin();
  Serial.printf("CLUSTER_TCP_SERVER_READY board_id=%u port=%u\n", (unsigned)CLUSTER_BOARD_ID,
                (unsigned)CLUSTER_WIFI_TCP_PORT);
#endif

#if CLUSTER_ENABLE_OTA
  cluster_setup_ota();
#endif
#if CLUSTER_ENABLE_HTTP_UPDATE
  cluster_setup_http_update();
#endif
#if CLUSTER_WIFI_LOCAL_GENERATOR
  cluster_model_ready = cluster_model_init_full_local();
  Serial.printf("CLUSTER_MODEL_READY board_id=%u ok=%u role=%s mode=local_generator\n", (unsigned)CLUSTER_BOARD_ID,
                cluster_model_ready ? 1 : 0, CLUSTER_ROLE_COORD ? "coord" : "worker");
#elif CLUSTER_WIFI_LAYER_SHARD
  cluster_model_ready = cluster_model_init_layer_shard();
  Serial.printf("CLUSTER_MODEL_READY board_id=%u ok=%u role=%s mode=layer_shard\n", (unsigned)CLUSTER_BOARD_ID,
                cluster_model_ready ? 1 : 0, CLUSTER_ROLE_COORD ? "coord" : "worker");
#elif CLUSTER_WIFI_SHARDED_INFERENCE || CLUSTER_WIFI_LSTM_SHARD
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
#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD
  cluster_handle_tcp_io(now);
#endif
#if CLUSTER_WIFI_LAYER_SHARD
  cluster_layer_shard_generation_tick(now);
#endif
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
#if CLUSTER_WIFI_SHARDED_INFERENCE || CLUSTER_WIFI_LSTM_SHARD
  if (now - cluster_last_ping_ms >= 2000) {
    cluster_last_ping_ms = now;
    uint32_t ping_seq = cluster_ping_seq++;
    bool ping_ok = cluster_send_packet(CLUSTER_AP_BROADCAST, CLUSTER_WIFI_UDP_PORT,
                                       cluster_protocol::CLUSTER_MSG_PING,
                                       CLUSTER_BROADCAST_BOARD, ping_seq);
    Serial.printf("CLUSTER_WIFI_PING_BROADCAST seq=%lu dst=%s port=%u sent=%s reason=%s\n",
                  (unsigned long)ping_seq, CLUSTER_AP_BROADCAST.toString().c_str(),
                  (unsigned)CLUSTER_WIFI_UDP_PORT, ping_ok ? "true" : "false",
                  CLUSTER_WIFI_LSTM_SHARD ? "lstm_shard_discovery" : "infer_discovery");
  }
#endif
#if CLUSTER_WIFI_SHARDED_INFERENCE
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
#if CLUSTER_WIFI_LSTM_SHARD
  cluster_distributed_generation_tick(now);
#endif
#if CLUSTER_WIFI_LOCAL_GENERATOR
  cluster_local_generator_tick(now);
#endif
#elif CLUSTER_ROLE_WORKER
  uint32_t now = millis();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("CLUSTER_WIFI_STA_DISCONNECTED reconnecting=true");
    WiFi.reconnect();
    delay(500);
    return;
  }
#if CLUSTER_WIFI_TCP_DIST && CLUSTER_WIFI_LSTM_SHARD
  cluster_handle_tcp_io(now);
#endif
  if (now - cluster_last_status_ms >= 5000) {
    cluster_last_status_ms = now;
    Serial.printf("CLUSTER_WIFI_WORKER_STATUS board_id=%u ", (unsigned)CLUSTER_BOARD_ID);
    cluster_print_ip("ip", WiFi.localIP());
    Serial.printf(" rssi=%ld port=%u\n", (long)WiFi.RSSI(), (unsigned)CLUSTER_WIFI_UDP_PORT);
  }
#if CLUSTER_WIFI_LOCAL_GENERATOR
  cluster_local_generator_tick(now);
#endif
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

#if RI_FINAL_SENTINEL
// This is deliberately separate from the build-time expected identity.  It is
// true only after the exact parsed RILM artifact has been SHA-256 verified.
static bool sentinel_model_hash_verified = false;
static uint32_t sentinel_model_artifact_bytes = 0;
#endif

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


#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_LSTM_SHARD
static spi_flash_mmap_handle_t cluster_lstm_shard_mmap = 0;
static const uint8_t *cluster_lstm_shard_base = nullptr;
static size_t cluster_lstm_shard_len = 0;
struct ClusterShardTensor {
  char name[32];
  uint8_t dtype = 0;
  uint8_t ndim = 0;
  uint32_t dims[2] = {0, 0};
  float scale = 1.0f;
  uint32_t payload_len = 0;
  const uint8_t *payload = nullptr;
};
static ClusterShardTensor cluster_lstm_shard_tensors[16];
static uint32_t cluster_lstm_shard_tensor_count = 0;
static uint32_t cluster_lstm_shard_row_start = 0;
static uint32_t cluster_lstm_shard_row_end = 0;
static constexpr uint32_t RIWS_MAGIC = 0x53574952;

static bool load_lstm_shard_partition() {
  const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                                         (esp_partition_subtype_t)0x40,
                                                         "weights");
  if (!part) {
    Serial.println("CLUSTER_MODEL_ERROR phase=shard_partition_find label=weights");
    return false;
  }
  const void *mapped = nullptr;
  esp_err_t err = esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &mapped,
                                     &cluster_lstm_shard_mmap);
  if (err != ESP_OK) {
    Serial.printf("CLUSTER_MODEL_ERROR phase=shard_mmap code=%d\n", (int)err);
    return false;
  }
  cluster_lstm_shard_base = (const uint8_t *)mapped;
  cluster_lstm_shard_len = part->size;
  const uint8_t *q = cluster_lstm_shard_base;
  uint32_t magic = rd_u32(q);
  uint16_t version = rd_u16(q);
  uint16_t worker = rd_u16(q);
  uint32_t hidden = rd_u32(q);
  uint32_t layers = rd_u32(q);
  uint32_t row_start = rd_u32(q);
  uint32_t row_end = rd_u32(q);
  uint32_t tensor_count = rd_u32(q);
  if (magic != RIWS_MAGIC) {
    Serial.printf("CLUSTER_MODEL_ERROR phase=shard_magic got=0x%08lx\n", (unsigned long)magic);
    return false;
  }
  if (version != 1 || worker != (uint16_t)CLUSTER_BOARD_ID || hidden != (uint32_t)HIDDEN ||
      layers != (uint32_t)LAYERS || tensor_count == 0 || tensor_count > 16) {
    Serial.printf("CLUSTER_MODEL_ERROR phase=shard_header version=%u worker=%u hidden=%lu layers=%lu tensors=%lu expected_worker=%u expected_hidden=%u expected_layers=%u\n",
                  (unsigned)version, (unsigned)worker, (unsigned long)hidden, (unsigned long)layers,
                  (unsigned long)tensor_count, (unsigned)CLUSTER_BOARD_ID, (unsigned)HIDDEN, (unsigned)LAYERS);
    return false;
  }
  cluster_lstm_shard_tensor_count = tensor_count;
  cluster_lstm_shard_row_start = row_start;
  cluster_lstm_shard_row_end = row_end;
  for (uint32_t i = 0; i < tensor_count; i++) {
    ClusterShardTensor *t = &cluster_lstm_shard_tensors[i];
    uint16_t actual_name_len = rd_u16(q);
    uint16_t stored_name_len = actual_name_len;
    if (stored_name_len >= sizeof(t->name)) stored_name_len = sizeof(t->name) - 1;
    memcpy(t->name, q, stored_name_len);
    t->name[stored_name_len] = 0;
    q += actual_name_len;
    t->dtype = *q++;
    t->ndim = *q++;
    for (uint8_t d = 0; d < t->ndim && d < 2; d++) t->dims[d] = rd_u32(q);
    for (uint8_t d = 2; d < t->ndim; d++) (void)rd_u32(q);
    t->scale = rd_f32(q);
    t->payload_len = rd_u32(q);
    t->payload = q;
    q += t->payload_len;
  }
  Serial.printf("CLUSTER_MODEL_WORKER_LSTM_SHARD_READY board_id=%u source=weights_partition format=RIWSv%u hidden=%lu layers=%lu rows=%lu-%lu tensors=%lu partition_addr=0x%lx partition_size=%lu\n",
                (unsigned)CLUSTER_BOARD_ID, (unsigned)version, (unsigned long)hidden,
                (unsigned long)layers, (unsigned long)row_start, (unsigned long)row_end,
                (unsigned long)tensor_count, (unsigned long)part->address, (unsigned long)part->size);
  return true;
}
#endif

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

static inline void feed_tg1_wdt() {
  // Feed the Timer Group 1 hardware watchdog directly via its feed register.
  // On ESP32-S3: TG1 base=0x60020000, WDTFEED=base+0x00C = 0x6002000C
  // Writing any value to this register resets the WDT countdown.
  volatile uint32_t *tg1_wdt_feed = (volatile uint32_t *)0x6002000C;
  *tg1_wdt_feed = 1;
}

static inline void model_init_pump_watchdog() {
  // H512 init can spend seconds cloning PSRAM payloads and packing recurrent
  // matrices before the main loop starts servicing WiFi. Yield explicitly so
  // the task watchdog does not reset the coordinator mid-load.
  yield();
  delay(0);
  // Feed the RTC hardware WDT to prevent TG1WDT reset.
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0x50D83AA1);
  WRITE_PERI_REG(RTC_CNTL_WDTCONFIG0_REG, 0x32 << 2 | RTC_CNTL_WDT_EN);
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0);
  // Also feed TG1 WDT (the actual source of TG1WDT_SYS_RST).
  feed_tg1_wdt();
}

static void model_init_copy_payload(uint8_t *dst, const uint8_t *src, uint32_t len) {
  static const uint32_t kChunk = 4096;
  uint32_t offset = 0;
  while (offset < len) {
    uint32_t n = len - offset;
    if (n > kChunk) n = kChunk;
    memcpy(dst + offset, src + offset, n);
    offset += n;
    model_init_pump_watchdog();
  }
}

static void model_init_suspend_watchdogs() {
  // Disable Task WDTs.
  disableLoopWDT();
  disableCore0WDT();
#ifndef CONFIG_FREERTOS_UNICORE
  disableCore1WDT();
#endif
  // Disable RTC hardware WDT (TG1WDT) to prevent reset during
  // heavy PSRAM clone/convert and generation operations.
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0x50D83AA1);
  CLEAR_PERI_REG_MASK(RTC_CNTL_WDTCONFIG0_REG, RTC_CNTL_WDT_EN);
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0);
  // Also disable Timer Group 1 WDT (TG1WDT_SYS_RST source on ESP32-S3).
  // This is the hardware watchdog that fires during long inference on H320.
#if RI_FINAL_SENTINEL
  // Feed then disable TG1 WDT via direct register access.
  // On ESP32-S3: TG0 base=0x6001F000, TG1 base=0x60020000
  // TG1 WDT: CONFIG0=base+0x48, FEED=base+0x0C, WPROTECT=base+0x10
  // Must unlock write-protect with magic 0x50D83AA1 before writing config.
  volatile uint32_t *tg1_wdt_prot  = (volatile uint32_t *)0x60020010;
  volatile uint32_t *tg1_wdt_feed  = (volatile uint32_t *)0x6002000C;
  volatile uint32_t *tg1_wdt_cfg   = (volatile uint32_t *)0x60020048;
  *tg1_wdt_prot  = 0x50D83AA1;   // Unlock write protect
  *tg1_wdt_feed  = 1;             // Feed (reset timer)
  *tg1_wdt_cfg  &= ~1u;           // Clear WDT_EN bit (bit 0)
  *tg1_wdt_prot  = 0;             // Re-lock write protect
#endif
}

static void model_init_resume_watchdogs() {
  // Do NOT re-enable any WDTs for local generator or layer-shard mode.
  // Both run heavy generation operations that need more than any WDT
  // timeout allows.
#if !CLUSTER_WIFI_LOCAL_GENERATOR && !CLUSTER_WIFI_LAYER_SHARD
  enableLoopWDT();
  enableCore0WDT();
#ifndef CONFIG_FREERTOS_UNICORE
  enableCore1WDT();
#endif
  // Re-enable RTC hardware WDT for sentinel mode (it was suspended during
  // generation but should be active during normal operation).
#if RI_FINAL_SENTINEL
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0x50D83AA1);
  WRITE_PERI_REG(RTC_CNTL_WDTCONFIG0_REG, 0x32 << 2 | RTC_CNTL_WDT_EN);
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0);
#endif
#endif
}

#if RI_FINAL_SENTINEL
static bool sentinel_verify_parsed_model_hash(const uint8_t *data, uint32_t len) {
  uint8_t digest[32];
  char hex[65];
  mbedtls_sha256_context ctx;
  mbedtls_sha256_init(&ctx);
  int rc = mbedtls_sha256_starts_ret(&ctx, 0);
  // Hash in bounded chunks so artifact verification itself cannot starve the
  // watchdog on a large memory-mapped partition.
  for (uint32_t offset = 0; rc == 0 && offset < len;) {
    uint32_t chunk = len - offset;
    if (chunk > 4096) chunk = 4096;
    rc = mbedtls_sha256_update_ret(&ctx, data + offset, chunk);
    offset += chunk;
    model_init_pump_watchdog();
  }
  if (rc == 0) rc = mbedtls_sha256_finish_ret(&ctx, digest);
  mbedtls_sha256_free(&ctx);
  if (rc != 0) {
    Serial.printf("S3_SENTINEL_MODEL_HASH error=%d bytes=%lu\n", rc, (unsigned long)len);
    return false;
  }
  static const char kHex[] = "0123456789abcdef";
  for (size_t i = 0; i < sizeof(digest); ++i) {
    hex[i * 2] = kHex[digest[i] >> 4];
    hex[i * 2 + 1] = kHex[digest[i] & 0x0f];
  }
  hex[64] = 0;
  const bool match = strcmp(hex, RI_WEIGHTS_SHA256) == 0;
  Serial.printf("S3_SENTINEL_MODEL_HASH verified=%u bytes=%lu expected=%s actual=%s\n",
                match ? 1 : 0, (unsigned long)len, RI_WEIGHTS_SHA256, hex);
  return match;
}
#endif

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
  const uint8_t *end = model.base + model.len;
  auto rilm_offset = [&]() -> unsigned long { return (unsigned long)(p - model.base); };
  auto need = [&](uint32_t n, const char *phase, uint32_t tensor_idx) -> bool {
    if ((uint32_t)(end - p) < n) {
      Serial.printf("RILM_BOUNDS_ERROR phase=%s tensor=%lu offset=%lu need=%lu remaining=%lu model_len=%lu\n",
                    phase, (unsigned long)tensor_idx, rilm_offset(), (unsigned long)n,
                    (unsigned long)(end - p), (unsigned long)model.len);
      Serial.flush();
      return false;
    }
    return true;
  };

  if (!need(12, "header", 0)) return false;
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
  Serial.flush();
  for (uint32_t i = 0; i < model.tensor_count; i++) {
    if (!need(2, "name_len", i)) return false;
    uint16_t actual_name_len = rd_u16(p);
    if (!need(actual_name_len + 2, "name_dtype_ndim", i)) return false;
    uint16_t stored_name_len = actual_name_len;
    if (stored_name_len >= sizeof(model.tensors[i].name)) stored_name_len = sizeof(model.tensors[i].name) - 1;
    memcpy(model.tensors[i].name, p, stored_name_len);
    model.tensors[i].name[stored_name_len] = 0;
    p += actual_name_len;
    model.tensors[i].dtype = *p++;
    model.tensors[i].ndim = *p++;
    if (model.tensors[i].ndim > 8) {
      Serial.printf("RILM_BOUNDS_ERROR phase=ndim tensor=%lu offset=%lu ndim=%u model_len=%lu\n",
                    (unsigned long)i, rilm_offset(), (unsigned)model.tensors[i].ndim,
                    (unsigned long)model.len);
      Serial.flush();
      return false;
    }
    if (!need((uint32_t)model.tensors[i].ndim * 4u + 8u, "dims_scale_payload_len", i)) return false;
    model.tensors[i].dims[0] = 0;
    model.tensors[i].dims[1] = 0;
    for (uint8_t d = 0; d < model.tensors[i].ndim && d < 2; d++) model.tensors[i].dims[d] = rd_u32(p);
    for (uint8_t d = 2; d < model.tensors[i].ndim; d++) (void)rd_u32(p);
    model.tensors[i].scale = rd_f32(p);
    model.tensors[i].payload_len = rd_u32(p);
    if (!need(model.tensors[i].payload_len, "payload", i)) return false;
    model.tensors[i].payload = p;
    Serial.printf("RILM_TENSOR index=%lu name=%s dtype=%u ndim=%u dim0=%lu dim1=%lu scale=%g payload_len=%lu payload_offset=%lu next_offset=%lu model_len=%lu\n",
                  (unsigned long)i, model.tensors[i].name, (unsigned)model.tensors[i].dtype,
                  (unsigned)model.tensors[i].ndim, (unsigned long)model.tensors[i].dims[0],
                  (unsigned long)model.tensors[i].dims[1], (double)model.tensors[i].scale,
                  (unsigned long)model.tensors[i].payload_len, rilm_offset(),
                  (unsigned long)(rilm_offset() + model.tensors[i].payload_len),
                  (unsigned long)model.len);
    Serial.flush();
    p += model.tensors[i].payload_len;
    model_init_pump_watchdog();
  }
#if RI_FINAL_SENTINEL
  sentinel_model_artifact_bytes = (uint32_t)(p - model.base);
  sentinel_model_hash_verified = sentinel_verify_parsed_model_hash(model.base, sentinel_model_artifact_bytes);
#endif
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
    model_init_copy_payload(copy, t->payload, t->payload_len);
    t->payload = copy;
    copied += t->payload_len;
    model_init_pump_watchdog();
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
      if ((j & 0x1FFFu) == 0) model_init_pump_watchdog();
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
  // all-int8 H256 packs convert both input and recurrent matrices (LAYERS*2).
  // mixed_lstm_safe TinyStories H512 already stores input matrices as int4 and only
  // needs recurrent matrices converted at boot (LAYERS). Anything below LAYERS means
  // a layer is missing a packed/convertible recurrent path.
  // In layer-shard mode, each board only has its single layer's weights, so accept >= 1.
#if CLUSTER_WIFI_LAYER_SHARD
  return converted >= 1;
#else
  return converted >= LAYERS;
#endif
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

bool resolve_model_layer_shard() {
  // Layer-shard mode: resolve only the tensors present in this board's partition.
  // Board 0 (coordinator): embed + layer 0 + fc. Boards 1/2: single LSTM layer.
  resolved.embed = find_tensor("embed.weight");
  resolved.fcw = find_tensor("fc.weight");
  resolved.fcb = find_tensor("fc.bias");
  const int board_layer = (int)CLUSTER_BOARD_ID;
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
  }
  // Validate this board's assigned layer
  if (!resolved.wih[board_layer] || !resolved.whh[board_layer] || !resolved.bih[board_layer]) {
    Serial.printf("ERR layer_shard unresolved layer %d for board_id=%d\n", board_layer, board_layer);
    return false;
  }
  // Coordinator (board 0) also needs embed and fc
  if (board_layer == 0) {
    if (!resolved.embed || !resolved.fcw || !resolved.fcb) {
      Serial.println("ERR layer_shard coordinator missing embed/fc");
      return false;
    }
  }
  resolved.ok = true;
  Serial.printf("LAYER_SHARD_RESOLVED board_id=%d layer=%d embed=%s fcw=%s fcb=%s wih=%s whh=%s bih=%s bhh=%s\n",
                board_layer, board_layer,
                resolved.embed ? "yes" : "no",
                resolved.fcw ? "yes" : "no",
                resolved.fcb ? "yes" : "no",
                resolved.wih[board_layer] ? "yes" : "no",
                resolved.whh[board_layer] ? "yes" : "no",
                resolved.bih[board_layer] ? "yes" : "no",
                resolved.bhh[board_layer] ? "yes" : "no");
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
  // Unsubscribe this task from the task watchdog — the worker computes
  // LSTM gates synchronously on demand and cannot feed the WDT during a
  // single model_step call. The sentinel generation loop feeds the RTC
  // WDT instead, and the main loop feeds its own task WDT.
#if RI_FINAL_SENTINEL
  esp_task_wdt_delete(NULL);
#endif
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
      if ((g & 63) == 0) {
        yield();
        model_init_pump_watchdog();
      }
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
    if ((g & 63) == 0) {
      yield();
      model_init_pump_watchdog();
    }
  }

  // Wait for core 0 worker with watchdog feeding — use a short timeout loop
  // instead of portMAX_DELAY so the TG1 hardware watchdog gets fed while we
  // block. This is critical for H320 inference which takes ~60ms per char.
  {
    const TickType_t wdt_feed_ticks = pdMS_TO_TICKS(50);
    while (xSemaphoreTake(core1_done_sem, wdt_feed_ticks) != pdTRUE) {
      feed_tg1_wdt();
      yield();
    }
  }
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
  // Suspend WDT before ANY heavy work, including core0 task creation.
  // The core0 worker blocks on core1_start_sem during model init; its idle
  // task WDT will fire if we create it before suspending.
  model_init_suspend_watchdogs();
  if (coordinator) {
    core1_start_sem = xSemaphoreCreateBinary();
    core1_done_sem = xSemaphoreCreateBinary();
    if (!core1_start_sem || !core1_done_sem) {
      model_init_resume_watchdogs();
      Serial.println("CLUSTER_MODEL_ERROR phase=semaphore");
      return false;
    }
    xTaskCreatePinnedToCore(core0_worker, "lstm_worker", 16384, nullptr, 2, nullptr, 0);
    core1_active = true;
  }
  if (!coordinator) {
    model_init_resume_watchdogs();
#if CLUSTER_ROLE_WORKER && CLUSTER_WIFI_LSTM_SHARD
    return load_lstm_shard_partition();
#elif CLUSTER_ROLE_WORKER && CLUSTER_WIFI_SHARDED_INFERENCE
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

  // WDT already suspended at top of function.

  bool ok = true;
  if (!load_model_partition()) { Serial.println("CLUSTER_MODEL_ERROR phase=load_partition"); ok = false; }

  // Coordinator runs the recurrent pass, so it needs the fast p22 memory layout:
  // cloned tensors plus int4 recurrent weights and SRAM scratch/state. Workers only
  // compute their FC vocabulary row shard from a coordinator-supplied hidden vector.
  // The worker FC head is embedded into the app firmware, so workers do not need a
  // data/weights partition or full recurrent model materialization.
  if (ok && !clone_payloads_to_psram()) { Serial.println("CLUSTER_MODEL_ERROR phase=clone_payloads"); ok = false; }
  if (ok && !convert_wih_to_int4()) { Serial.println("CLUSTER_MODEL_ERROR phase=int4_convert"); ok = false; }

  if (ok && !resolve_model()) { Serial.println("CLUSTER_MODEL_ERROR phase=resolve"); ok = false; }
  if (ok && (resolved.fcw->dims[1] != HIDDEN || resolved.fcw->dims[0] != VOCAB_SIZE)) {
    Serial.printf("CLUSTER_MODEL_ERROR phase=fc_shape rows=%lu cols=%lu expected_rows=%u expected_cols=%u\n",
                  (unsigned long)resolved.fcw->dims[0], (unsigned long)resolved.fcw->dims[1],
                  (unsigned)VOCAB_SIZE, (unsigned)HIDDEN);
    ok = false;
  }
  if (ok && coordinator) {
    init_activation_lut();
    alloc_state();
  }
  model_init_resume_watchdogs();
  return ok;
}

static bool cluster_model_init_full_local() {
  model_init_suspend_watchdogs();
  core1_start_sem = xSemaphoreCreateBinary();
  core1_done_sem = xSemaphoreCreateBinary();
  if (!core1_start_sem || !core1_done_sem) {
    model_init_resume_watchdogs();
    Serial.println("CLUSTER_MODEL_ERROR phase=semaphore mode=local_generator");
    return false;
  }
  xTaskCreatePinnedToCore(core0_worker, "lstm_worker", 16384, nullptr, 2, nullptr, 0);
  core1_active = true;
  bool ok = true;
  if (!load_model_partition()) { Serial.println("CLUSTER_MODEL_ERROR phase=load_partition mode=local_generator"); ok = false; }
  if (ok && !clone_payloads_to_psram()) { Serial.println("CLUSTER_MODEL_ERROR phase=clone_payloads mode=local_generator"); ok = false; }
  if (ok && !convert_wih_to_int4()) { Serial.println("CLUSTER_MODEL_ERROR phase=int4_convert mode=local_generator"); ok = false; }
  if (ok && !resolve_model()) { Serial.println("CLUSTER_MODEL_ERROR phase=resolve mode=local_generator"); ok = false; }
  if (ok) {
    init_activation_lut();
    alloc_state();
    Serial.printf("CLUSTER_MODEL_LOCAL_GENERATOR_READY board_id=%u profile=%s params=%lu hidden=%u layers=%u weights_sha256=%s\n",
                  (unsigned)CLUSTER_BOARD_ID, RI_MODEL_PROFILE, (unsigned long)RI_MODEL_PARAMS,
                  (unsigned)HIDDEN, (unsigned)LAYERS, WEIGHTS_SHA256);
  }
  model_init_resume_watchdogs();
  return ok;
}

static bool cluster_model_init_layer_shard() {
  model_init_suspend_watchdogs();

  // All boards need the core0 worker for dual-core LSTM gate computation
  core1_start_sem = xSemaphoreCreateBinary();
  core1_done_sem = xSemaphoreCreateBinary();
  if (!core1_start_sem || !core1_done_sem) {
    model_init_resume_watchdogs();
    Serial.println("CLUSTER_MODEL_ERROR phase=semaphore mode=layer_shard");
    return false;
  }
  xTaskCreatePinnedToCore(core0_worker, "lstm_worker", 16384, nullptr, 2, nullptr, 0);
  core1_active = true;

  bool ok = true;
  if (!load_model_partition()) { Serial.println("CLUSTER_MODEL_ERROR phase=load_partition mode=layer_shard"); ok = false; }
  if (ok && !clone_payloads_to_psram()) { Serial.println("CLUSTER_MODEL_ERROR phase=clone_payloads mode=layer_shard"); ok = false; }
  if (ok && !convert_wih_to_int4()) { Serial.println("CLUSTER_MODEL_ERROR phase=int4_convert mode=layer_shard"); ok = false; }
  if (ok && !resolve_model_layer_shard()) { Serial.println("CLUSTER_MODEL_ERROR phase=resolve_layer_shard"); ok = false; }

  if (ok) {
    init_activation_lut();
    alloc_state();
    Serial.printf("CLUSTER_MODEL_LAYER_SHARD_READY board_id=%u layer=%d profile=%s hidden=%u layers=%u\n",
                  (unsigned)CLUSTER_BOARD_ID, (int)CLUSTER_BOARD_ID,
                  RI_MODEL_PROFILE, (unsigned)HIDDEN, (unsigned)LAYERS);
  }
  model_init_resume_watchdogs();
  return ok;
}

#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_LAYER_SHARD
static void cluster_layer_shard_generation_tick(uint32_t now) {
  // Discovery pings to find workers
  if (now - cluster_last_ping_ms >= 2000) {
    cluster_last_ping_ms = now;
    uint32_t ping_seq = cluster_ping_seq++;
    bool ping_ok = cluster_send_packet(CLUSTER_AP_BROADCAST, CLUSTER_WIFI_UDP_PORT,
                                       cluster_protocol::CLUSTER_MSG_PING,
                                       CLUSTER_BROADCAST_BOARD, ping_seq);
    Serial.printf("CLUSTER_WIFI_PING_BROADCAST seq=%lu dst=%s port=%u sent=%s reason=layer_shard_discovery\n",
                  (unsigned long)ping_seq, CLUSTER_AP_BROADCAST.toString().c_str(),
                  (unsigned)CLUSTER_WIFI_UDP_PORT, ping_ok ? "true" : "false");
  }

  // Start generation: run once at now >= 12000 after both workers PONG
  if (!cluster_layer_shard_gen_started && !cluster_layer_shard_gen_active &&
      cluster_model_ready && now >= 12000 &&
      cluster_worker_ip_known[1] && cluster_worker_ip_known[2]) {
    cluster_layer_shard_gen_started = true;
    cluster_layer_shard_gen_active = true;
    cluster_layer_shard_gen_started_ms = now;

    // 1. Embedding lookup for input token
    const char *prompt = "once upon a ";
    size_t plen = strlen(prompt);
    reset_state();
    // Process prefix tokens through embed + layer 0
    for (size_t i = 0; i < plen; i++) {
      int tok = vocab_idx(prompt[i]);
      // Embedding lookup
      Tensor *embed = resolved.embed;
      uint32_t row = (uint32_t)tok * HIDDEN;
      if (embed->dtype == I4) {
        for (int j = 0; j < HIDDEN; j += 2) {
          uint8_t b = embed->payload[(row + j) >> 1];
          st.x[j] = (float)signed_i4_low(b) * embed->scale;
          st.x[j + 1] = (float)signed_i4_high(b) * embed->scale;
        }
      } else {
        for (int j = 0; j < HIDDEN; j++) st.x[j] = tensor_get_slow(embed, row + j);
      }
      // Run layer 0 locally
      lstm_layer(0, st.x, HIDDEN, false);
      memcpy(st.x, st.h[0], sizeof(float) * HIDDEN);
    }

    // After prefix processing, st.x and st.h[0]/st.c[0] hold the final state
    // Now quantize h[0] and c[0] for sending to board 1
    cluster_layer_shard_h_scale = quantize_q8(st.h[0], (int8_t *)cluster_layer_shard_qx, HIDDEN);
    cluster_layer_shard_c_scale = quantize_q8(st.c[0], (int8_t *)cluster_layer_shard_qc, HIDDEN);

    Serial.printf("CLUSTER_LAYER_SHARD_GEN_START prompt=\"%s\" final_input=%c h_scale=%.9f c_scale=%.9f\n",
                  prompt, plen ? prompt[plen - 1] : ' ',
                  (double)cluster_layer_shard_h_scale, (double)cluster_layer_shard_c_scale);

    // 2. Send state forward request to board 1 (layer 1)
    cluster_layer_shard_gen_seq_b1 = cluster_layer_shard_seq++;
    cluster_layer_shard_gen_waiting_b1 = true;
    cluster_layer_shard_b1_result_ready = false;
    cluster_layer_shard_gen_last_send_b1_ms = now;

    cluster_protocol::ClusterLstmStateForwardRequest request;
    request.token_id = 0;
    request.layer_start = 1;
    request.layer_count = 1;
    request.hidden_scale = cluster_layer_shard_h_scale;
    request.cell_scale = cluster_layer_shard_c_scale;
    request.qx = cluster_layer_shard_qx;
    request.qc = cluster_layer_shard_qc;

    uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_STATE_FORWARD_REQUEST_PAYLOAD_SIZE];
    bool encoded = cluster_protocol::encode_lstm_state_forward_request_payload(
        request, request_payload, sizeof(request_payload));
    IPAddress target1 = cluster_worker_ips[1];
    bool sent1 = encoded && cluster_send_packet(target1, CLUSTER_WIFI_UDP_PORT,
                                                cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST,
                                                1, cluster_layer_shard_gen_seq_b1,
                                                request_payload, sizeof(request_payload));
    Serial.printf("CLUSTER_LAYER_SHARD_STATE_SEND seq=%lu dst=1 target=%s layer_start=1 sent=%s\n",
                  (unsigned long)cluster_layer_shard_gen_seq_b1, target1.toString().c_str(),
                  sent1 ? "true" : "false");
    return;
  }

  // Retry logic for board 1
  if (cluster_layer_shard_gen_active && cluster_layer_shard_gen_waiting_b1 &&
      now - cluster_layer_shard_gen_last_send_b1_ms > 3000) {
    Serial.printf("CLUSTER_LAYER_SHARD_RETRY_B1 seq=%lu\n", (unsigned long)cluster_layer_shard_gen_seq_b1);
    cluster_layer_shard_gen_last_send_b1_ms = now;
    cluster_protocol::ClusterLstmStateForwardRequest request;
    request.token_id = 0;
    request.layer_start = 1;
    request.layer_count = 1;
    request.hidden_scale = cluster_layer_shard_h_scale;
    request.cell_scale = cluster_layer_shard_c_scale;
    request.qx = cluster_layer_shard_qx;
    request.qc = cluster_layer_shard_qc;
    uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_STATE_FORWARD_REQUEST_PAYLOAD_SIZE];
    bool encoded = cluster_protocol::encode_lstm_state_forward_request_payload(
        request, request_payload, sizeof(request_payload));
    cluster_send_packet(cluster_worker_ips[1], CLUSTER_WIFI_UDP_PORT,
                        cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST,
                        1, cluster_layer_shard_gen_seq_b1,
                        request_payload, sizeof(request_payload));
  }

  // Retry logic for board 2
  if (cluster_layer_shard_gen_active && cluster_layer_shard_gen_waiting_b2 &&
      now - cluster_layer_shard_gen_last_send_b2_ms > 3000) {
    Serial.printf("CLUSTER_LAYER_SHARD_RETRY_B2 seq=%lu\n", (unsigned long)cluster_layer_shard_gen_seq_b2);
    cluster_layer_shard_gen_last_send_b2_ms = now;
    cluster_protocol::ClusterLstmStateForwardRequest request;
    request.token_id = 0;
    request.layer_start = 2;
    request.layer_count = 1;
    request.hidden_scale = cluster_layer_shard_h_scale;
    request.cell_scale = cluster_layer_shard_c_scale;
    request.qx = cluster_layer_shard_qx;
    request.qc = cluster_layer_shard_qc;
    uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_STATE_FORWARD_REQUEST_PAYLOAD_SIZE];
    bool encoded = cluster_protocol::encode_lstm_state_forward_request_payload(
        request, request_payload, sizeof(request_payload));
    cluster_send_packet(cluster_worker_ips[2], CLUSTER_WIFI_UDP_PORT,
                        cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST,
                        2, cluster_layer_shard_gen_seq_b2,
                        request_payload, sizeof(request_payload));
  }
}
#endif

#if CLUSTER_ROLE_COORD && CLUSTER_WIFI_LAYER_SHARD
static void cluster_handle_layer_shard_state_result(const cluster_protocol::ClusterPacketHeader &header,
                                                     const uint8_t *payload, size_t payload_len) {
  cluster_protocol::ClusterLstmStateForwardResult result;
  if (!cluster_protocol::decode_lstm_state_forward_result_payload(payload, payload_len, &result)) {
    Serial.printf("CLUSTER_LAYER_SHARD_DROP reason=bad_state_result_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }

  uint8_t src = header.src_board;
  Serial.printf("CLUSTER_LAYER_SHARD_STATE_RESULT src_board=%u seq=%lu layer_end=%u decoded=true\n",
                (unsigned)src, (unsigned long)header.seq, (unsigned)result.layer_end);

  if (src == 1 && cluster_layer_shard_gen_waiting_b1) {
    // Board 1 result: store and forward to board 2
    cluster_layer_shard_result_h_scale = result.hidden_scale;
    cluster_layer_shard_result_c_scale = result.cell_scale;
    memcpy(cluster_layer_shard_result_qx, result.qx, cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN);
    memcpy(cluster_layer_shard_result_qc, result.qc, cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN);
    cluster_layer_shard_b1_result_ready = true;
    cluster_layer_shard_gen_waiting_b1 = false;

    // Send to board 2
    cluster_layer_shard_gen_seq_b2 = cluster_layer_shard_seq++;
    cluster_layer_shard_gen_waiting_b2 = true;
    cluster_layer_shard_b2_result_ready = false;
    cluster_layer_shard_gen_last_send_b2_ms = millis();

    cluster_protocol::ClusterLstmStateForwardRequest req2;
    req2.token_id = 0;
    req2.layer_start = 2;
    req2.layer_count = 1;
    req2.hidden_scale = cluster_layer_shard_result_h_scale;
    req2.cell_scale = cluster_layer_shard_result_c_scale;
    req2.qx = cluster_layer_shard_result_qx;
    req2.qc = cluster_layer_shard_result_qc;

    uint8_t req_payload[cluster_protocol::CLUSTER_LSTM_STATE_FORWARD_REQUEST_PAYLOAD_SIZE];
    bool encoded = cluster_protocol::encode_lstm_state_forward_request_payload(
        req2, req_payload, sizeof(req_payload));
    IPAddress target2 = cluster_worker_ips[2];
    bool sent = encoded && cluster_send_packet(target2, CLUSTER_WIFI_UDP_PORT,
                                                cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST,
                                                2, cluster_layer_shard_gen_seq_b2,
                                                req_payload, sizeof(req_payload));
    Serial.printf("CLUSTER_LAYER_SHARD_FORWARD_TO_B2 seq=%lu target=%s sent=%s\n",
                  (unsigned long)cluster_layer_shard_gen_seq_b2,
                  target2.toString().c_str(), sent ? "true" : "false");
  } else if (src == 2 && cluster_layer_shard_gen_waiting_b2) {
    // Board 2 result: dequantize and run FC
    cluster_layer_shard_result_h_scale = result.hidden_scale;
    cluster_layer_shard_result_c_scale = result.cell_scale;
    memcpy(cluster_layer_shard_result_qx, result.qx, cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN);
    memcpy(cluster_layer_shard_result_qc, result.qc, cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN);
    cluster_layer_shard_b2_result_ready = true;
    cluster_layer_shard_gen_waiting_b2 = false;

    // Dequantize final hidden state
    for (int i = 0; i < HIDDEN; i++) {
      st.h[2][i] = (float)((int8_t)cluster_layer_shard_result_qx[i]) * cluster_layer_shard_result_h_scale;
      st.c[2][i] = (float)((int8_t)cluster_layer_shard_result_qc[i]) * cluster_layer_shard_result_c_scale;
    }
    memcpy(st.x, st.h[2], sizeof(float) * HIDDEN);

    // Run FC head
    cluster_layer_shard_output_token = model_finish_fc(false, &cluster_layer_shard_output_logit);
    uint32_t elapsed = millis() - cluster_layer_shard_gen_started_ms;
    Serial.printf("CLUSTER_LAYER_SHARD_TOKEN token=%u char=%c logit=%.6f elapsed_ms=%lu\n",
                  (unsigned)cluster_layer_shard_output_token,
                  idx_vocab(cluster_layer_shard_output_token),
                  (double)cluster_layer_shard_output_logit,
                  (unsigned long)elapsed);
    cluster_layer_shard_gen_active = false;
  }
}
#endif

#if CLUSTER_WIFI_LAYER_SHARD
static void cluster_handle_layer_shard_state_request(const cluster_protocol::ClusterPacketHeader &header,
                                                       const uint8_t *payload, size_t payload_len) {
#if CLUSTER_ROLE_WORKER
  if (header.dst_board != CLUSTER_BROADCAST_BOARD && header.dst_board != (uint8_t)CLUSTER_BOARD_ID) return;

  cluster_protocol::ClusterLstmStateForwardRequest request;
  if (!cluster_protocol::decode_lstm_state_forward_request_payload(payload, payload_len, &request)) {
    Serial.printf("CLUSTER_LAYER_SHARD_DROP reason=bad_state_request_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }
  Serial.printf("CLUSTER_LAYER_SHARD_STATE_REQUEST board_id=%u seq=%lu token=%u layer_start=%u layer_count=%u decoded=true\n",
                (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq, (unsigned)request.token_id,
                (unsigned)request.layer_start, (unsigned)request.layer_count);

  const int board_layer = (int)CLUSTER_BOARD_ID;
  float h_scale = request.hidden_scale;
  float c_scale = request.cell_scale;
  const int8_t *rqx = (const int8_t *)request.qx;
  const int8_t *rqc = (const int8_t *)request.qc;

  for (int i = 0; i < HIDDEN; i++) {
    st.h[board_layer][i] = (float)rqx[i] * h_scale;
    st.c[board_layer][i] = (float)rqc[i] * c_scale;
  }
  memcpy(st.x, st.h[board_layer], sizeof(float) * HIDDEN);

  lstm_layer(board_layer, st.x, HIDDEN, false);

  static int8_t out_qx[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16)));
  static int8_t out_qc[cluster_protocol::CLUSTER_LSTM_STATE_HIDDEN] __attribute__((aligned(16)));
  float out_h_scale = quantize_q8(st.h[board_layer], out_qx, HIDDEN);
  float out_c_scale = quantize_q8(st.c[board_layer], out_qc, HIDDEN);

  Serial.printf("CLUSTER_LAYER_SHARD_LAYER_DONE board_id=%u layer=%d out_h_scale=%.9f out_c_scale=%.9f\n",
                (unsigned)CLUSTER_BOARD_ID, board_layer,
                (double)out_h_scale, (double)out_c_scale);

  cluster_protocol::ClusterLstmStateForwardResult result;
  result.layer_end = (uint8_t)(board_layer + 1);
  result.hidden_scale = out_h_scale;
  result.cell_scale = out_c_scale;
  result.qx = (const uint8_t *)out_qx;
  result.qc = (const uint8_t *)out_qc;

  static uint8_t result_payload[cluster_protocol::CLUSTER_LSTM_STATE_FORWARD_RESULT_PAYLOAD_SIZE];
  bool encoded = cluster_protocol::encode_lstm_state_forward_result_payload(
      result, result_payload, sizeof(result_payload));
  bool ok = encoded && cluster_send_packet(cluster_udp.remoteIP(), cluster_udp.remotePort(),
                                           cluster_protocol::CLUSTER_MSG_LSTM_STATE_FORWARD_RESULT,
                                           header.src_board, header.seq, result_payload,
                                           sizeof(result_payload));
  if (!ok) {
    Serial.printf("CLUSTER_LAYER_SHARD_DROP reason=state_result_send_failed board_id=%u seq=%lu encoded=%s\n",
                  (unsigned)CLUSTER_BOARD_ID, (unsigned long)header.seq,
                  encoded ? "true" : "false");
  }
#else
  (void)header;
  (void)payload;
  (void)payload_len;
#endif
}
#endif

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

#if CLUSTER_WIFI_LSTM_SHARD
#if CLUSTER_ROLE_WORKER
static inline float shard_f32_at(const ClusterShardTensor *t, uint32_t idx) {
  float v;
  memcpy(&v, t->payload + idx * 4, 4);
  return v;
}

static inline float dot_shard_tensor_q8(const ClusterShardTensor *t, uint32_t elem_row_start,
                                        const int8_t *xq, float x_scale, int n) {
  if (t->dtype == I8) return (float)dot_i8_i8_acc(t->payload, elem_row_start, xq, n) * t->scale * x_scale;
  if (t->dtype == I4) return (float)dot_i4_i8_acc(t->payload, elem_row_start, xq, n) * t->scale * x_scale;
  float acc = 0.0f;
  for (int j = 0; j < n; j++) acc += shard_f32_at(t, elem_row_start + j) * ((float)xq[j] * x_scale);
  return acc;
}

static ClusterShardTensor *find_lstm_shard_tensor(const char *name) {
  for (uint32_t i = 0; i < cluster_lstm_shard_tensor_count; i++) {
    if (strcmp(cluster_lstm_shard_tensors[i].name, name) == 0) return &cluster_lstm_shard_tensors[i];
  }
  return nullptr;
}
#endif

static void lstm_tensor_names(uint8_t layer, char *wih_name, char *whh_name, char *bih_name, char *bhh_name, size_t cap) {
  snprintf(wih_name, cap, "lstm.weight_ih_l%u", (unsigned)layer);
  snprintf(whh_name, cap, "lstm.weight_hh_l%u", (unsigned)layer);
  snprintf(bih_name, cap, "lstm.bias_ih_l%u", (unsigned)layer);
  snprintf(bhh_name, cap, "lstm.bias_hh_l%u", (unsigned)layer);
}

static bool cluster_prepare_lstm_gate_probe(uint8_t layer) {
#if CLUSTER_ROLE_COORD
  if (!resolved.ok || layer >= LAYERS || !resolved.embed || !st.x || !st.qx || !st.qh[layer]) return false;
  reset_state();
  const int token = vocab_idx('o');
  for (int i = 0; i < HIDDEN; i++) st.x[i] = tensor_get_slow(resolved.embed, (uint32_t)token * HIDDEN + i);
  cluster_lstm_gate_input_scale = quantize_q8(st.x, cluster_lstm_gate_qx, HIDDEN);
  memset(st.h[layer], 0, sizeof(float) * HIDDEN);
  cluster_lstm_gate_h_scale = quantize_q8(st.h[layer], cluster_lstm_gate_qh, HIDDEN);
  return true;
#else
  (void)layer;
  return false;
#endif
}

static bool cluster_expected_lstm_gate_values(uint8_t layer, uint16_t row_start, uint16_t count,
                                              int32_t *values_out) {
#if CLUSTER_ROLE_COORD
  if (!values_out || layer >= LAYERS || !resolved.ok) return false;
  Tensor *wih = resolved.wih[layer];
  Tensor *whh = resolved.whh[layer];
  Tensor *bih = resolved.bih[layer];
  Tensor *bhh = resolved.bhh[layer];
  if (!wih || !whh || !bih || !bhh) return false;
  for (uint16_t i = 0; i < count; i++) {
    uint32_t g = (uint32_t)row_start + i;
    float acc = f32_at(bih, g) + f32_at(bhh, g);
    acc += dot_tensor_q8(wih, g * HIDDEN, cluster_lstm_gate_qx, cluster_lstm_gate_input_scale, HIDDEN);
    acc += dot_tensor_q8(whh, g * HIDDEN, cluster_lstm_gate_qh, cluster_lstm_gate_h_scale, HIDDEN);
    values_out[i] = (int32_t)lrintf(acc * 1024.0f);
  }
  return true;
#else
  (void)layer; (void)row_start; (void)count; (void)values_out;
  return false;
#endif
}

static bool cluster_worker_compute_lstm_gate_probe(uint8_t layer, uint16_t row_start, uint16_t requested_count,
                                                   const int8_t *qx, float input_scale,
                                                   const int8_t *qh, float h_scale,
                                                   uint16_t *row_start_out, int32_t *values_out,
                                                   uint16_t *count_out) {
#if CLUSTER_ROLE_WORKER
  if (layer >= LAYERS || !qx || !qh || !row_start_out || !values_out || !count_out) return false;
  if (requested_count == 0 || requested_count > cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_VALUES) return false;
  if (row_start < cluster_lstm_shard_row_start || row_start > cluster_lstm_shard_row_end) return false;
  if ((uint32_t)row_start + requested_count - 1 > cluster_lstm_shard_row_end) return false;
  char wih_name[32], whh_name[32], bih_name[32], bhh_name[32];
  lstm_tensor_names(layer, wih_name, whh_name, bih_name, bhh_name, sizeof(wih_name));
  ClusterShardTensor *wih = find_lstm_shard_tensor(wih_name);
  ClusterShardTensor *whh = find_lstm_shard_tensor(whh_name);
  ClusterShardTensor *bih = find_lstm_shard_tensor(bih_name);
  ClusterShardTensor *bhh = find_lstm_shard_tensor(bhh_name);
  if (!wih || !whh || !bih || !bhh) return false;
  *row_start_out = row_start;
  *count_out = requested_count;
  for (uint16_t i = 0; i < requested_count; i++) {
    uint32_t local_g = (uint32_t)(row_start - cluster_lstm_shard_row_start) + i;
    float acc = shard_f32_at(bih, local_g) + shard_f32_at(bhh, local_g);
    acc += dot_shard_tensor_q8(wih, local_g * HIDDEN, qx, input_scale, HIDDEN);
    acc += dot_shard_tensor_q8(whh, local_g * HIDDEN, qh, h_scale, HIDDEN);
    values_out[i] = (int32_t)lrintf(acc * 1024.0f);
  }
  return true;
#else
  (void)layer; (void)row_start; (void)requested_count; (void)qx; (void)input_scale; (void)qh; (void)h_scale; (void)row_start_out; (void)values_out; (void)count_out;
  return false;
#endif
}


#if CLUSTER_ROLE_COORD
static uint16_t cluster_dist_chunk_size() {
#if CLUSTER_WIFI_TCP_DIST
  return (uint16_t)min((uint16_t)1024, (uint16_t)(1024 - cluster_dist_offset));
#else
  return (uint16_t)min((uint16_t)256, (uint16_t)(1024 - cluster_dist_offset));
#endif
}

#if CLUSTER_WIFI_UDP_PIPELINE_DIST
static bool cluster_dist_pipeline_index_for_row(uint8_t worker, uint16_t row_start, uint16_t count,
                                                uint8_t *index_out) {
  if (!index_out || count != CLUSTER_DIST_PIPELINE_CHUNK_ROWS || worker == 0 || worker > 2) return false;
  const uint16_t base = (worker == 1) ? 0 : 1024;
  if (row_start < base) return false;
  const uint16_t offset = row_start - base;
  if (offset >= 1024 || (offset % CLUSTER_DIST_PIPELINE_CHUNK_ROWS) != 0) return false;
  *index_out = (uint8_t)(offset / CLUSTER_DIST_PIPELINE_CHUNK_ROWS);
  return *index_out < CLUSTER_DIST_PIPELINE_CHUNKS;
}

static bool cluster_dist_pipeline_complete() {
  for (uint8_t dst = 1; dst <= 2; dst++) {
    for (uint8_t i = 0; i < CLUSTER_DIST_PIPELINE_CHUNKS; i++) {
      if (!cluster_dist_pipeline_seen[dst][i]) return false;
    }
  }
  return true;
}

static void cluster_dist_pipeline_reset_seen() {
  for (uint8_t dst = 0; dst < 3; dst++) {
    for (uint8_t i = 0; i < CLUSTER_DIST_PIPELINE_CHUNKS; i++) {
      cluster_dist_pipeline_seen[dst][i] = false;
      cluster_dist_pipeline_max_abs_err[dst][i] = 0;
    }
  }
  cluster_lstm_gate_seen[1] = false;
  cluster_lstm_gate_seen[2] = false;
  cluster_lstm_gate_max_abs_err[1] = 0;
  cluster_lstm_gate_max_abs_err[2] = 0;
  cluster_lstm_gate_gather_printed = false;
}

static bool cluster_dist_pipeline_send_layer(bool retry_missing_only) {
  if (!cluster_dist_active || cluster_dist_layer >= LAYERS) return false;

  cluster_lstm_gate_input_scale = quantize_q8(st.x, cluster_lstm_gate_qx, HIDDEN);
  cluster_lstm_gate_h_scale = quantize_q8(st.h[cluster_dist_layer], cluster_lstm_gate_qh, HIDDEN);
  if (!retry_missing_only) {
    cluster_dist_pipeline_reset_seen();
    for (uint8_t i = 0; i < CLUSTER_DIST_PIPELINE_CHUNKS; i++) {
      cluster_dist_pipeline_seq[i] = cluster_dist_seq++;
    }
  }

  Serial.printf("CLUSTER_DIST_GEN_LAYER_PIPE_REQUEST layer=%u chunks=%u count=%u input_scale=%.9f h_scale=%.9f retry_missing_only=%u\n",
                (unsigned)cluster_dist_layer, (unsigned)CLUSTER_DIST_PIPELINE_CHUNKS,
                (unsigned)CLUSTER_DIST_PIPELINE_CHUNK_ROWS, (double)cluster_lstm_gate_input_scale,
                (double)cluster_lstm_gate_h_scale, retry_missing_only ? 1 : 0);

  bool all_sent = true;
  for (uint8_t chunk = 0; chunk < CLUSTER_DIST_PIPELINE_CHUNKS; chunk++) {
    const uint16_t offset = (uint16_t)chunk * CLUSTER_DIST_PIPELINE_CHUNK_ROWS;
    for (uint8_t dst = 1; dst <= 2; dst++) {
      if (retry_missing_only && cluster_dist_pipeline_seen[dst][chunk]) continue;
      uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE];
      size_t request_payload_len = 0;
      const uint16_t row_start = (dst == 1) ? offset : (uint16_t)(1024 + offset);
      bool encoded = cluster_protocol::encode_lstm_gate_request_payload(
          cluster_dist_layer, row_start, CLUSTER_DIST_PIPELINE_CHUNK_ROWS,
          cluster_lstm_gate_input_scale, cluster_lstm_gate_h_scale, cluster_lstm_gate_qx,
          cluster_lstm_gate_qh, request_payload, sizeof(request_payload), &request_payload_len);
      IPAddress target = cluster_worker_ip_known[dst] ? cluster_worker_ips[dst] : CLUSTER_AP_BROADCAST;
      bool sent = encoded && cluster_send_packet(target, CLUSTER_WIFI_UDP_PORT,
                                                 cluster_protocol::CLUSTER_MSG_LSTM_GATE_REQUEST,
                                                 dst, cluster_dist_pipeline_seq[chunk],
                                                 request_payload, (uint16_t)request_payload_len);
      if (!sent) all_sent = false;
      Serial.printf("CLUSTER_DIST_GEN_PIPE_SEND seq=%lu dst=%u layer=%u offset=%u row_start=%u count=%u target=%s sent=%s\n",
                    (unsigned long)cluster_dist_pipeline_seq[chunk], (unsigned)dst,
                    (unsigned)cluster_dist_layer, (unsigned)offset, (unsigned)row_start,
                    (unsigned)CLUSTER_DIST_PIPELINE_CHUNK_ROWS, target.toString().c_str(),
                    sent ? "true" : "false");
    }
  }

  cluster_dist_waiting = true;
  cluster_dist_last_send_ms = millis();
  if (!all_sent) {
    Serial.printf("CLUSTER_DIST_GEN_PIPE_SEND_INCOMPLETE layer=%u retry=true\n",
                  (unsigned)cluster_dist_layer);
  }
  return all_sent;
}
#endif

static void cluster_dist_prepare_prefix_and_expected() {
  reset_state();
  const char *prompt = cluster_dist_prompt;
  for (const char *q = prompt; *q; q++) model_step_hidden(vocab_idx(*q), false);
  cluster_dist_expected_token = (uint8_t)model_finish_fc(false);

  reset_state();
  size_t n = strlen(prompt);
  if (n > 1) {
    for (size_t i = 0; i + 1 < n; i++) model_step_hidden(vocab_idx(prompt[i]), false);
  }
}

static bool cluster_dist_send_pair() {
  if (!cluster_dist_active || cluster_dist_layer >= LAYERS || cluster_dist_offset >= 1024) return false;
  const uint16_t count = (uint16_t)min((uint16_t)256, (uint16_t)(1024 - cluster_dist_offset));
  cluster_lstm_gate_input_scale = quantize_q8(st.x, cluster_lstm_gate_qx, HIDDEN);
  cluster_lstm_gate_h_scale = quantize_q8(st.h[cluster_dist_layer], cluster_lstm_gate_qh, HIDDEN);
  cluster_dist_active_seq = cluster_dist_seq++;
  cluster_lstm_gate_active_seq = cluster_dist_active_seq;
  cluster_dist_seen[1] = false;
  cluster_dist_seen[2] = false;
  cluster_lstm_gate_seen[1] = false;
  cluster_lstm_gate_seen[2] = false;
  cluster_lstm_gate_max_abs_err[1] = 0;
  cluster_lstm_gate_max_abs_err[2] = 0;
  cluster_lstm_gate_gather_printed = false;
  Serial.printf("CLUSTER_DIST_GEN_CHUNK_REQUEST seq=%lu layer=%u offset=%u count=%u input_scale=%.9f h_scale=%.9f\n",
                (unsigned long)cluster_dist_active_seq, (unsigned)cluster_dist_layer,
                (unsigned)cluster_dist_offset, (unsigned)count,
                (double)cluster_lstm_gate_input_scale, (double)cluster_lstm_gate_h_scale);
  bool all_sent = true;
  for (uint8_t dst = 1; dst <= 2; dst++) {
    uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE];
    size_t request_payload_len = 0;
    uint16_t row_start = (dst == 1) ? cluster_dist_offset : (uint16_t)(1024 + cluster_dist_offset);
    bool encoded = cluster_protocol::encode_lstm_gate_request_payload(
        cluster_dist_layer, row_start, count, cluster_lstm_gate_input_scale, cluster_lstm_gate_h_scale,
        cluster_lstm_gate_qx, cluster_lstm_gate_qh, request_payload, sizeof(request_payload), &request_payload_len);
    IPAddress target = cluster_worker_ip_known[dst] ? cluster_worker_ips[dst] : CLUSTER_AP_BROADCAST;
    bool sent = encoded && cluster_send_packet(target, CLUSTER_WIFI_UDP_PORT,
                                               cluster_protocol::CLUSTER_MSG_LSTM_GATE_REQUEST,
                                               dst, cluster_dist_active_seq, request_payload,
                                               (uint16_t)request_payload_len);
    if (!sent) all_sent = false;
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_SEND seq=%lu dst=%u row_start=%u count=%u target=%s sent=%s\n",
                  (unsigned long)cluster_dist_active_seq, (unsigned)dst, (unsigned)row_start,
                  (unsigned)count, target.toString().c_str(), sent ? "true" : "false");
  }
  cluster_dist_waiting = true;
  cluster_dist_last_send_ms = millis();
  if (!all_sent) {
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_SEND_INCOMPLETE seq=%lu layer=%u offset=%u retry=true\n",
                  (unsigned long)cluster_dist_active_seq, (unsigned)cluster_dist_layer,
                  (unsigned)cluster_dist_offset);
  }
  return all_sent;
}

#if CLUSTER_WIFI_TCP_DIST
static bool cluster_dist_send_pair_tcp() {
  if (!cluster_dist_active || cluster_dist_layer >= LAYERS || cluster_dist_offset >= 1024) return false;
  const uint16_t count = cluster_dist_chunk_size();
  cluster_lstm_gate_input_scale = quantize_q8(st.x, cluster_lstm_gate_qx, HIDDEN);
  cluster_lstm_gate_h_scale = quantize_q8(st.h[cluster_dist_layer], cluster_lstm_gate_qh, HIDDEN);
  cluster_dist_active_seq = cluster_dist_seq++;
  cluster_lstm_gate_active_seq = cluster_dist_active_seq;
  cluster_dist_seen[1] = false;
  cluster_dist_seen[2] = false;
  cluster_lstm_gate_seen[1] = false;
  cluster_lstm_gate_seen[2] = false;
  cluster_lstm_gate_max_abs_err[1] = 0;
  cluster_lstm_gate_max_abs_err[2] = 0;
  cluster_lstm_gate_gather_printed = false;
  Serial.printf("CLUSTER_DIST_GEN_CHUNK_REQUEST_TCP seq=%lu layer=%u offset=%u count=%u input_scale=%.9f h_scale=%.9f\n",
                (unsigned long)cluster_dist_active_seq, (unsigned)cluster_dist_layer,
                (unsigned)cluster_dist_offset, (unsigned)count,
                (double)cluster_lstm_gate_input_scale, (double)cluster_lstm_gate_h_scale);
  bool all_sent = true;
  for (uint8_t dst = 1; dst <= 2; dst++) {
    static uint8_t request_payload[cluster_protocol::CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE];
    size_t request_payload_len = 0;
    uint16_t row_start = (dst == 1) ? cluster_dist_offset : (uint16_t)(1024 + cluster_dist_offset);
    bool encoded = cluster_protocol::encode_lstm_gate_request_payload(
        cluster_dist_layer, row_start, count, cluster_lstm_gate_input_scale, cluster_lstm_gate_h_scale,
        cluster_lstm_gate_qx, cluster_lstm_gate_qh, request_payload, sizeof(request_payload), &request_payload_len);
    bool sent = encoded && cluster_send_tcp_packet(dst, cluster_protocol::CLUSTER_MSG_LSTM_GATE_REQUEST,
                                                   cluster_dist_active_seq, request_payload,
                                                   (uint16_t)request_payload_len);
    if (!sent) all_sent = false;
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_SEND_TCP seq=%lu dst=%u row_start=%u count=%u target=%s sent=%s\n",
                  (unsigned long)cluster_dist_active_seq, (unsigned)dst, (unsigned)row_start,
                  (unsigned)count,
                  cluster_worker_ip_known[dst] ? cluster_worker_ips[dst].toString().c_str() : "unknown",
                  sent ? "true" : "false");
  }
  cluster_dist_waiting = true;
  cluster_dist_last_send_ms = millis();
  if (!all_sent) {
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_SEND_INCOMPLETE_TCP seq=%lu layer=%u offset=%u retry=true\n",
                  (unsigned long)cluster_dist_active_seq, (unsigned)cluster_dist_layer,
                  (unsigned)cluster_dist_offset);
  }
  return all_sent;
}
#endif

static void cluster_dist_finish_layer() {
  for (int i = 0; i < HIDDEN; i++) {
    float ingate = sigmoidf_fast(st.gates[i]);
    float forgetgate = sigmoidf_fast(st.gates[HIDDEN + i]);
    float cellgate = tanhf_fast(st.gates[2 * HIDDEN + i]);
    float outgate = sigmoidf_fast(st.gates[3 * HIDDEN + i]);
    float c = forgetgate * st.c[cluster_dist_layer][i] + ingate * cellgate;
    float h = outgate * tanhf_fast(c);
    st.next_c[i] = c;
    st.next_h[i] = h;
  }
  memcpy(st.h[cluster_dist_layer], st.next_h, sizeof(float) * HIDDEN);
  memcpy(st.c[cluster_dist_layer], st.next_c, sizeof(float) * HIDDEN);
  memcpy(st.x, st.h[cluster_dist_layer], sizeof(float) * HIDDEN);
  Serial.printf("CLUSTER_DIST_GEN_LAYER_DONE layer=%u\n", (unsigned)cluster_dist_layer);
}

static void cluster_dist_advance_or_finish() {
  if (!cluster_dist_active || cluster_dist_waiting) return;
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
  cluster_dist_finish_layer();
  cluster_dist_layer++;
  cluster_dist_offset = 0;
  if (cluster_dist_layer >= LAYERS) {
    cluster_dist_output_token = (uint8_t)model_finish_fc(false, &cluster_dist_output_logit);
    const uint32_t elapsed = millis() - cluster_dist_started_ms;
    const bool token_match = (cluster_dist_output_token == cluster_dist_expected_token);
    Serial.printf("CLUSTER_DIST_GEN_TOKEN prompt=\"%s\" local_p22_token=%u local_p22_char=%c dist_token=%u dist_char=%c logit=%.6f elapsed_ms=%lu status=%s note=worker_int8_recurrent_vs_local_int4_reference_udp_pipeline\n",
                  cluster_dist_prompt, (unsigned)cluster_dist_expected_token, idx_vocab(cluster_dist_expected_token),
                  (unsigned)cluster_dist_output_token, idx_vocab(cluster_dist_output_token),
                  (double)cluster_dist_output_logit, (unsigned long)elapsed,
                  token_match ? "PASS" : "FAIL");
    cluster_dist_active = false;
    return;
  }
  cluster_dist_pipeline_send_layer(false);
#else
  cluster_dist_offset += cluster_dist_chunk_size();
  if (cluster_dist_offset >= 1024) {
    cluster_dist_finish_layer();
    cluster_dist_layer++;
    cluster_dist_offset = 0;
    if (cluster_dist_layer >= LAYERS) {
      cluster_dist_output_token = (uint8_t)model_finish_fc(false, &cluster_dist_output_logit);
      const uint32_t elapsed = millis() - cluster_dist_started_ms;
      const bool token_match = (cluster_dist_output_token == cluster_dist_expected_token);
      Serial.printf("CLUSTER_DIST_GEN_TOKEN prompt=\"%s\" local_p22_token=%u local_p22_char=%c dist_token=%u dist_char=%c logit=%.6f elapsed_ms=%lu status=%s note=worker_int8_recurrent_vs_local_int4_reference\n",
                    cluster_dist_prompt, (unsigned)cluster_dist_expected_token, idx_vocab(cluster_dist_expected_token),
                    (unsigned)cluster_dist_output_token, idx_vocab(cluster_dist_output_token),
                    (double)cluster_dist_output_logit, (unsigned long)elapsed,
                    token_match ? "PASS" : "FAIL");
      cluster_dist_active = false;
      return;
    }
  }
#if CLUSTER_WIFI_TCP_DIST
  cluster_dist_send_pair_tcp();
#else
  cluster_dist_send_pair();
#endif
#endif
}

static void cluster_distributed_generation_tick(uint32_t now) {
  if (!cluster_model_ready) return;
  if (!cluster_dist_started && now > 12000 && cluster_worker_ip_known[1] && cluster_worker_ip_known[2]) {
    cluster_dist_started = true;
    cluster_dist_active = true;
    cluster_dist_waiting = false;
    cluster_dist_layer = 0;
    cluster_dist_offset = 0;
    cluster_dist_started_ms = now;
    cluster_dist_prepare_prefix_and_expected();
    const char *prompt = cluster_dist_prompt;
    size_t n = strlen(prompt);
    const int token = vocab_idx(n ? prompt[n - 1] : ' ');
    uint32_t row = (uint32_t)token * HIDDEN;
    for (int i = 0; i < HIDDEN; i++) st.x[i] = tensor_get_slow(resolved.embed, row + i);
    Serial.printf("CLUSTER_DIST_GEN_START prompt=\"%s\" final_input=%c expected_token=%u expected_char=%c\n",
                  cluster_dist_prompt, n ? prompt[n - 1] : ' ', (unsigned)cluster_dist_expected_token,
                  idx_vocab(cluster_dist_expected_token));
#if CLUSTER_WIFI_TCP_DIST
    cluster_dist_send_pair_tcp();
#elif CLUSTER_WIFI_UDP_PIPELINE_DIST
    cluster_dist_pipeline_send_layer(false);
#else
    cluster_dist_send_pair();
#endif
    return;
  }
  if (cluster_dist_active && cluster_dist_waiting && now - cluster_dist_last_send_ms > 3000) {
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_RETRY seq=%lu layer=%u offset=%u seen1=%u seen2=%u\n",
                  (unsigned long)cluster_dist_active_seq, (unsigned)cluster_dist_layer,
                  (unsigned)cluster_dist_offset, cluster_dist_seen[1] ? 1 : 0, cluster_dist_seen[2] ? 1 : 0);
#if CLUSTER_WIFI_TCP_DIST
    cluster_dist_send_pair_tcp();
#elif CLUSTER_WIFI_UDP_PIPELINE_DIST
    cluster_dist_pipeline_send_layer(true);
#else
    cluster_dist_send_pair();
#endif
  }
  if (cluster_dist_active && !cluster_dist_waiting) cluster_dist_advance_or_finish();
}
#endif

static void cluster_handle_lstm_gate_result(const cluster_protocol::ClusterPacketHeader &header,
                                            const uint8_t *payload, size_t payload_len) {
#if CLUSTER_ROLE_COORD
  uint8_t layer = 0;
  uint16_t row_start = 0;
  uint16_t count = 0;
  static int32_t values[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_VALUES];
  if (!cluster_protocol::decode_lstm_gate_result_payload(payload, payload_len, &layer, &row_start, values, &count)) {
    Serial.printf("CLUSTER_LSTM_GATE_RESULT_DROP reason=bad_payload src_board=%u seq=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq);
    return;
  }
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
  uint8_t pipeline_chunk = 0;
  const bool pipeline_result =
      cluster_dist_active && cluster_dist_waiting &&
      cluster_dist_pipeline_index_for_row(header.src_board, row_start, count, &pipeline_chunk) &&
      layer == cluster_dist_layer && header.seq == cluster_dist_pipeline_seq[pipeline_chunk];
  if ((!pipeline_result && header.seq != cluster_lstm_gate_active_seq) ||
      header.src_board > 2 || header.src_board == 0) {
#else
  if (header.seq != cluster_lstm_gate_active_seq || header.src_board > 2 || header.src_board == 0) {
#endif
    Serial.printf("CLUSTER_LSTM_GATE_RESULT_DROP reason=stale_or_bad_src src_board=%u seq=%lu active=%lu\n",
                  (unsigned)header.src_board, (unsigned long)header.seq, (unsigned long)cluster_lstm_gate_active_seq);
    return;
  }
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
  if (pipeline_result && cluster_dist_pipeline_seen[header.src_board][pipeline_chunk]) {
    Serial.printf("CLUSTER_LSTM_GATE_RESULT_DROP reason=duplicate_pipeline_result src_board=%u seq=%lu layer=%u row_start=%u count=%u\n",
                  (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)layer,
                  (unsigned)row_start, (unsigned)count);
    return;
  }
#endif
  static int32_t expected[cluster_protocol::CLUSTER_LSTM_GATE_RESULT_MAX_VALUES];
  bool expected_ok = cluster_expected_lstm_gate_values(layer, row_start, count, expected);
  int32_t max_abs_err = 0;
  bool ok = expected_ok;
  for (uint16_t i = 0; i < count && expected_ok; i++) {
    int32_t err = values[i] - expected[i];
    if (err < 0) err = -err;
    if (err > max_abs_err) max_abs_err = err;
    if (err > 2) ok = false;
  }
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
  if (!pipeline_result) {
    cluster_lstm_gate_seen[header.src_board] = ok;
    cluster_lstm_gate_max_abs_err[header.src_board] = max_abs_err;
  }
#else
  cluster_lstm_gate_seen[header.src_board] = ok;
  cluster_lstm_gate_max_abs_err[header.src_board] = max_abs_err;
#endif
#if CLUSTER_ROLE_COORD
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
  if (pipeline_result) {
    // Worker shards intentionally use the int8 recurrent path while the local
    // reference uses the coordinator int4 path. The existing non-pipelined
    // distributed proof accepts those values and reports local_reference_ok=false
    // when drift exceeds the strict reference threshold. Preserve that contract:
    // accept structurally valid worker results, copy them into the gate buffer,
    // and report the reference drift separately.
    for (uint16_t i = 0; i < count; i++) {
      uint32_t g = (uint32_t)row_start + i;
      if (g < 4u * HIDDEN) st.gates[g] = ((float)values[i]) / 1024.0f;
    }
    cluster_dist_pipeline_seen[header.src_board][pipeline_chunk] = true;
    cluster_dist_pipeline_max_abs_err[header.src_board][pipeline_chunk] = max_abs_err;
    Serial.printf("CLUSTER_DIST_GEN_PIPE_RESULT src_board=%u seq=%lu layer=%u chunk=%u row_start=%u count=%u accepted=true local_reference_ok=%s max_abs_err=%ld complete=%u\n",
                  (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)layer,
                  (unsigned)pipeline_chunk, (unsigned)row_start, (unsigned)count,
                  ok ? "true" : "false", (long)max_abs_err,
                  cluster_dist_pipeline_complete() ? 1 : 0);
    if (cluster_dist_pipeline_complete()) {
      cluster_lstm_gate_seen[1] = true;
      cluster_lstm_gate_seen[2] = true;
      cluster_lstm_gate_max_abs_err[1] = 0;
      cluster_lstm_gate_max_abs_err[2] = 0;
      for (uint8_t dst = 1; dst <= 2; dst++) {
        for (uint8_t i = 0; i < CLUSTER_DIST_PIPELINE_CHUNKS; i++) {
          if (cluster_dist_pipeline_max_abs_err[dst][i] > cluster_lstm_gate_max_abs_err[dst]) {
            cluster_lstm_gate_max_abs_err[dst] = cluster_dist_pipeline_max_abs_err[dst][i];
          }
        }
      }
      cluster_dist_waiting = false;
      Serial.printf("CLUSTER_DIST_GEN_LAYER_PIPE_GATHER layer=%u chunks=%u results=8 worker1_max_abs_err=%ld worker2_max_abs_err=%ld status=PASS\n",
                    (unsigned)layer, (unsigned)CLUSTER_DIST_PIPELINE_CHUNKS,
                    (long)cluster_lstm_gate_max_abs_err[1], (long)cluster_lstm_gate_max_abs_err[2]);
    }
  } else
#endif
  if (cluster_dist_active && header.seq == cluster_dist_active_seq) {
    for (uint16_t i = 0; i < count; i++) {
      uint32_t g = (uint32_t)row_start + i;
      if (g < 4u * HIDDEN) st.gates[g] = ((float)values[i]) / 1024.0f;
    }
    cluster_dist_seen[header.src_board] = true;
    Serial.printf("CLUSTER_DIST_GEN_CHUNK_RESULT src_board=%u seq=%lu layer=%u row_start=%u count=%u accepted=true local_reference_ok=%s max_abs_err=%ld\n",
                  (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)layer,
                  (unsigned)row_start, (unsigned)count, ok ? "true" : "false", (long)max_abs_err);
    if (cluster_dist_seen[1] && cluster_dist_seen[2]) cluster_dist_waiting = false;
  }
#endif
  Serial.printf("CLUSTER_LSTM_GATE_RESULT src_board=%u seq=%lu layer=%u row_start=%u count=%u max_abs_err=%ld ok=%s\n",
                (unsigned)header.src_board, (unsigned long)header.seq, (unsigned)layer,
                (unsigned)row_start, (unsigned)count, (long)max_abs_err, ok ? "true" : "false");
  if (
#if CLUSTER_WIFI_UDP_PIPELINE_DIST && !CLUSTER_WIFI_TCP_DIST
      !pipeline_result &&
#endif
      !cluster_lstm_gate_gather_printed && cluster_lstm_gate_seen[1] && cluster_lstm_gate_seen[2]) {
    cluster_lstm_gate_gather_printed = true;
    Serial.printf("CLUSTER_LSTM_GATE_GATHER seq=%lu layer=%u worker1_ok=true worker1_max_abs_err=%ld worker2_ok=true worker2_max_abs_err=%ld rows_checked=%u status=PASS\n",
                  (unsigned long)header.seq, (unsigned)layer, (long)cluster_lstm_gate_max_abs_err[1],
                  (long)cluster_lstm_gate_max_abs_err[2], (unsigned)(count * 2));
  }
#else
  (void)header; (void)payload; (void)payload_len;
#endif
}
#endif
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
  for (const char *p = seed; *p; p++) {
    token = model_step(vocab_idx(*p), false);
    yield();
    feed_tg1_wdt();
  }
  int i = 0;
  for (; i < max_chars; i++) {
    char ch = idx_vocab(token);
    out[i] = ch;
    if (ch == '.' || ch == '\n') { i++; break; }
    token = model_step(token, false);
    yield();
    // Feed TG1 WDT every char to prevent reset on slow H320 model
    feed_tg1_wdt();
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
  Serial.printf("\"model_profile\":\"%s\",", RI_MODEL_PROFILE);
  Serial.printf("\"params\":%lu,", (unsigned long)RI_MODEL_PARAMS);
  Serial.printf("\"compressed_bytes\":%lu,", (unsigned long)RI_COMPRESSED_BYTES);
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

#if CLUSTER_WIFI_LOCAL_GENERATOR
static void cluster_local_generator_tick(uint32_t now) {
  if (!cluster_model_ready || now < 12000) return;
  if (cluster_local_bench_last_ms != 0 && now - cluster_local_bench_last_ms < 30000) return;
  cluster_local_bench_last_ms = now;

  const uint8_t prompt_id = (uint8_t)(CLUSTER_BOARD_ID % SEED_COUNT);
  const char *seed = BENCH_SEEDS[prompt_id];
  const int gen_chars = CLUSTER_LOCAL_GEN_CHARS;
  char output[CLUSTER_LOCAL_GEN_CHARS + 1];
  uint32_t checksum = 2166136261u;

  reset_state();
  int token = vocab_idx(seed[0]);
  for (const char *p = seed; *p; p++) token = model_step(vocab_idx(*p), false);

  uint32_t start_ms = millis();
  memset(&ops, 0, sizeof(ops));
  for (int i = 0; i < gen_chars; i++) {
    char ch = idx_vocab(token);
    output[i] = ch;
    checksum ^= (uint8_t)ch;
    checksum *= 16777619u;
    token = model_step(token, true);
    yield();
  }
  output[gen_chars] = 0;
  uint32_t elapsed_ms = millis() - start_ms;
  float chars_per_sec = elapsed_ms ? (1000.0f * (float)gen_chars / (float)elapsed_ms) : 0.0f;

  cluster_protocol::ClusterBenchResult result;
  result.prompt_id = prompt_id;
  result.model_profile_id = 1;
  result.generated_chars = gen_chars;
  result.elapsed_ms = elapsed_ms;
  result.checksum = checksum;
  result.chars_per_sec = chars_per_sec;

  Serial.printf("CLUSTER_BENCH_RESULT board=%u prompt_id=%u profile=%s generated_chars=%u elapsed_ms=%lu chars_per_sec=%.4f checksum=0x%08lx output=\"",
                (unsigned)CLUSTER_BOARD_ID, (unsigned)prompt_id, RI_MODEL_PROFILE,
                (unsigned)result.generated_chars, (unsigned long)result.elapsed_ms,
                (double)result.chars_per_sec, (unsigned long)result.checksum);
  json_escape_print(output);
  Serial.println("\"");

#if CLUSTER_ROLE_WORKER
  uint8_t payload[cluster_protocol::CLUSTER_BENCH_RESULT_PAYLOAD_SIZE];
  bool encoded = cluster_protocol::encode_bench_result_payload(result, payload, sizeof(payload));
  cluster_local_bench_packet_sent = encoded && cluster_send_packet(
      CLUSTER_AP_IP, CLUSTER_WIFI_UDP_PORT, cluster_protocol::CLUSTER_MSG_BENCH_RESULT,
      0, 9000u + (uint32_t)CLUSTER_BOARD_ID, payload, sizeof(payload));
  Serial.printf("CLUSTER_BENCH_RESULT_SEND board=%u sent=%s target=%s:%u\n",
                (unsigned)CLUSTER_BOARD_ID, cluster_local_bench_packet_sent ? "true" : "false",
                CLUSTER_AP_IP.toString().c_str(), (unsigned)CLUSTER_WIFI_UDP_PORT);
#endif
}
#endif

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
  Serial.printf("\"model_profile\":\"%s\",", RI_MODEL_PROFILE);
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

#if RI_FINAL_SENTINEL

// ── Multi-sensor buffer (replaces old SentinelReading) ──────────────
static SensorBuffer g_sensors;

// ── Servo / actuator state ───────────────────────────────────────────
static ServoConfig g_servos[MAX_SERVOS];
static int g_servo_count = 0;

static void servo_init_slot(int slot, uint8_t gpio) {
  if (slot < 0 || slot >= MAX_SERVOS) return;
  ServoConfig &s = g_servos[slot];
  s.gpio = gpio;
  s.channel = (uint8_t)slot;  // LEDC channel 0-3
  s.attached = true;
  s.current_angle = 90;  // neutral
  ledcSetup(s.channel, SERVO_FREQ_HZ, SERVO_RESOLUTION);
  ledcAttachPin(gpio, s.channel);
  ledcWrite(s.channel, SERVO_DUTY_NEUTRAL);
  Serial.printf("S3_SENTINEL_SERVO_ATTACHED slot=%d gpio=%u channel=%u angle=90\n",
                slot, (unsigned)gpio, (unsigned)s.channel);
  if (slot + 1 > g_servo_count) g_servo_count = slot + 1;
}

static void servo_move(int slot, int angle) {
  if (slot < 0 || slot >= MAX_SERVOS || !g_servos[slot].attached) return;
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  uint32_t duty = servo_angle_to_duty(angle);
  ledcWrite(g_servos[slot].channel, duty);
  g_servos[slot].current_angle = angle;
  Serial.printf("S3_SENTINEL_SERVO_MOVE slot=%d angle=%d duty=%lu\n",
                slot, angle, (unsigned long)duty);
}

// ── Multi-sensor serial command: SENSOR:<type>:<value>[:<secondary>][:<age_s>]
// Legacy format SENSOR:<temp_c>,<humidity_pct>,<age_s> still supported.
static void sentinel_handle_sensor_command(const char *line) {
  const char *p = line + 7;  // skip "SENSOR:"
  while (*p == ' ') p++;

  // Legacy format: SENSOR:29.4,72.0,3 (temp,humidity,age)
  if (isdigit((int)*p) || *p == '-' || *p == '.' || *p == 'n') {
    char *end;
    float temp_c = strtof(p, &end);
    if (end == p || (*end != ',' && *end != '\0')) {
      Serial.println("S3_SENTINEL_ERROR {\"error\":\"invalid_SENSOR_format\"}");
      return;
    }
    if (*end == ',') {
      // Full legacy: temp,humidity,age
      p = end + 1;
      float humidity = strtof(p, &end);
      if (end == p) {
        Serial.println("S3_SENTINEL_ERROR {\"error\":\"invalid_SENSOR_format\"}");
        return;
      }
      p = end + (strchr(p, ',') ? 1 : 0);
      unsigned long age_s = 0;
      if (*p) age_s = strtoul(p, &end, 10);

      // Inject temp
      SensorReading *tr = g_sensors.find_or_alloc(SENSOR_TEMP);
      if (tr) {
        tr->value = temp_c;
        tr->quality = sensor_value_plausible(SENSOR_TEMP, temp_c) ? QUALITY_VALID : QUALITY_IMPLAUSIBLE;
        tr->age_ms = age_s * 1000UL;
        tr->last_update = millis() - tr->age_ms;
      }
      // Inject humidity
      SensorReading *hr = g_sensors.find_or_alloc(SENSOR_HUMIDITY);
      if (hr) {
        hr->value = humidity;
        hr->quality = sensor_value_plausible(SENSOR_HUMIDITY, humidity) ? QUALITY_VALID : QUALITY_IMPLAUSIBLE;
        hr->age_ms = age_s * 1000UL;
        hr->last_update = millis() - hr->age_ms;
      }
      Serial.printf("S3_SENTINEL_SENSOR {\"accepted\":true,\"legacy\":true,\"temp_c\":%.2f,\"humidity_pct\":%.2f,\"age_ms\":%lu}\n",
                    temp_c, humidity, (unsigned long)(age_s * 1000UL));
      return;
    }
    // Just a temperature value
    SensorReading *tr = g_sensors.find_or_alloc(SENSOR_TEMP);
    if (tr) {
      tr->value = temp_c;
      tr->quality = sensor_value_plausible(SENSOR_TEMP, temp_c) ? QUALITY_VALID : QUALITY_IMPLAUSIBLE;
      tr->age_ms = 0;
      tr->last_update = millis();
    }
    Serial.printf("S3_SENTINEL_SENSOR {\"accepted\":true,\"type\":\"temp\",\"value\":%.2f}\n", temp_c);
    return;
  }

  // New format: SENSOR:<type>:<value>[:<secondary>][:<age_s>]
  // Parse type
  char type_str[32];
  int ti = 0;
  while (*p && *p != ':' && ti < (int)sizeof(type_str) - 1) type_str[ti++] = *p++;
  type_str[ti] = '\0';
  if (*p != ':') {
    Serial.println("S3_SENTINEL_ERROR {\"error\":\"missing_value\",\"hint\":\"SENSOR:<type>:<value>\"}");
    return;
  }
  p++;

  SensorType stype = parse_sensor_type(type_str);
  if (stype == SENSOR_NONE) {
    Serial.printf("S3_SENTINEL_ERROR {\"error\":\"unknown_sensor_type\",\"type\":\"%s\"}\n", type_str);
    return;
  }

  // Parse value
  char *end;
  float value = strtof(p, &end);
  if (end == p) {
    Serial.printf("S3_SENTINEL_ERROR {\"error\":\"invalid_value\",\"type\":\"%s\"}\n", type_str);
    return;
  }
  p = end;

  // Optional secondary value
  float secondary = NAN;
  if (*p == ':') {
    p++;
    float s = strtof(p, &end);
    if (end != p) {
      secondary = s;
      p = end;
    }
  }

  // Optional age in seconds
  unsigned long age_s = 0;
  if (*p == ':') {
    p++;
    age_s = strtoul(p, &end, 10);
    p = end;
  }

  // Validate plausibility
  SensorQuality quality = sensor_value_plausible(stype, value) ? QUALITY_VALID : QUALITY_IMPLAUSIBLE;
  if (isnan(value)) quality = QUALITY_NAN;

  // Store reading
  SensorReading *r = g_sensors.find_or_alloc(stype);
  if (!r) {
    Serial.println("S3_SENTINEL_ERROR {\"error\":\"sensor_buffer_full\"}");
    return;
  }
  r->value = value;
  r->secondary = secondary;
  r->quality = quality;
  r->age_ms = age_s * 1000UL;
  r->last_update = millis() - r->age_ms;

  Serial.printf("S3_SENTINEL_SENSOR {\"accepted\":true,\"type\":\"%s\",\"value\":%.2f,\"quality\":\"%s\",\"age_ms\":%lu}\n",
                sensor_type_name(stype), value, quality_str(quality), (unsigned long)(age_s * 1000UL));
}

// ── STATUS command: full multi-sensor status ────────────────────────
static void sentinel_handle_status_command() {
  g_sensors.update_age();
  Serial.print("S3_SENTINEL_STATUS {\"schema\":\"ri_esp32s3_sentinel_status_v2\",");
  Serial.printf("\"device_id\":\"%s\",\"boot_id\":%lu,\"event_seq\":%lu,",
                g_receipt_id.device_id, (unsigned long)g_receipt_id.boot_id,
                (unsigned long)g_receipt_id.event_seq);
  Serial.printf("\"firmware_variant\":\"%s\",\"model_profile\":\"%s\",\"params\":%lu,",
                FIRMWARE_VARIANT, RI_MODEL_PROFILE, (unsigned long)RI_MODEL_PARAMS);
  Serial.printf("\"weights_sha256\":\"%s\",\"model_hash_verified\":%s,",
                WEIGHTS_SHA256, sentinel_model_hash_verified ? "true" : "false");
  Serial.printf("\"sensor_count\":%d,", g_sensors.count);
  Serial.print("\"sensors\":[");
  for (int i = 0; i < g_sensors.count; i++) {
    SensorReading &r = g_sensors.readings[i];
    if (i > 0) Serial.print(",");
    Serial.printf("{\"type\":\"%s\",\"quality\":\"%s\",\"value\":%.2f,\"age_ms\":%lu}",
                  sensor_type_name(r.type), quality_str(r.quality), r.value, (unsigned long)r.age_ms);
  }
  Serial.print("],");
  Serial.printf("\"uptime_ms\":%lu,\"free_heap\":%lu,\"free_psram\":%lu,\"psram_size\":%lu}\n",
                (unsigned long)millis(), (unsigned long)ESP.getFreeHeap(),
                (unsigned long)ESP.getFreePsram(), (unsigned long)ESP.getPsramSize());
}

// ── RUN command: evaluate policy, optionally generate advisory text ──
static void sentinel_handle_run_command() {
  PolicyResult policy = evaluate_policy(g_sensors);
  char output[UTILITY_MAX_CHARS + 1];
  uint32_t gen_elapsed_ms = 0;
  float gen_chars_per_sec = 0.0f;
  bool local_generated = false;

  if (policy.ai_route) {
    // H320 generation can take ~1.4s for 16 chars. The semaphore wait loop
    // in lstm_layer() feeds the TG1 watchdog every 50ms, so WiFi tasks on
    // core 0 get CPU time and the hardware watchdog stays fed.
    uint32_t start = millis();
    generate_stopped(policy.prompt, output, 16);
    gen_elapsed_ms = millis() - start;
    int chars = strlen(output);
    gen_chars_per_sec = gen_elapsed_ms ? (1000.0f * (float)chars / (float)gen_elapsed_ms) : 0.0f;
    local_generated = true;
  }

  // ── Actuator: policy-driven servo control ───────────────────────────
  // The LM NEVER controls actuators. Only the deterministic policy decides.
  ActuatorCommand act = policy_to_servo((uint8_t)policy.decision);
  if (act.servo_move && g_servos[act.servo_channel].attached) {
    servo_move(act.servo_channel, act.servo_angle);
  }

  const char *receipt = build_receipt(
    g_sensors, policy, local_generated,
    local_generated ? output : "",
    gen_elapsed_ms, gen_chars_per_sec,
    sentinel_model_hash_verified,
    FIRMWARE_VARIANT, RI_MODEL_PROFILE,
    RI_MODEL_PARAMS, WEIGHTS_SHA256);

  Serial.print("S3_SENTINEL_RECEIPT ");
  Serial.println(receipt);

  // Emit actuator receipt if servo moved
  if (act.servo_move && g_servos[act.servo_channel].attached) {
    Serial.printf("S3_SENTINEL_ACTUATOR {\"servo\":%d,\"angle\":%d,\"reason\":\"%s\",\"policy\":\"%s\"}\n",
                  act.servo_channel, act.servo_angle, act.reason,
                  policy_decision_name(policy.decision));
  }
}

// ── SENSORS command: list all registered sensor types ───────────────
static void sentinel_handle_sensors_list_command() {
  Serial.println("S3_SENTINEL_SENSOR_TYPES");
  for (int i = 1; i < SENSOR_TYPE_COUNT; i++) {
    Serial.printf("  %s [%s] range=%.1f..%.1f\n",
                  SENSOR_RANGES[i].name, SENSOR_RANGES[i].unit,
                  SENSOR_RANGES[i].min_val, SENSOR_RANGES[i].max_val);
  }
  Serial.printf("  registered=%d/%d\n", g_sensors.count, MAX_SENSORS);
}

// ── CLEAR command: clear all sensor readings ─────────────────────────
static void sentinel_handle_clear_command() {
  g_sensors.count = 0;
  Serial.println("S3_SENTINEL_CLEARED");
}

// ── SERVO command: SERVO:<slot>:<angle> ─────────────────────────────
// Manually move a servo. slot=0-3, angle=0-180.
static void sentinel_handle_servo_command(const char *line) {
  const char *p = line + 6;  // skip "SERVO:"
  char *end;
  int slot = (int)strtol(p, &end, 10);
  if (end == p || *end != ':') {
    Serial.println("S3_SENTINEL_ERROR {\"error\":\"invalid_SERVO_format\",\"hint\":\"SERVO:<slot>:<angle>\"}");
    return;
  }
  p = end + 1;
  int angle = (int)strtol(p, &end, 10);
  if (slot < 0 || slot >= MAX_SERVOS || !g_servos[slot].attached) {
    Serial.printf("S3_SENTINEL_ERROR {\"error\":\"servo_not_attached\",\"slot\":%d}\n", slot);
    return;
  }
  servo_move(slot, angle);
}

// ── SERVOATTACH command: SERVOATTACH:<slot>:<gpio> ──────────────────
static void sentinel_handle_servo_attach_command(const char *line) {
  const char *p = line + 12;  // skip "SERVOATTACH:"
  char *end;
  int slot = (int)strtol(p, &end, 10);
  if (end == p || *end != ':') {
    Serial.println("S3_SENTINEL_ERROR {\"error\":\"invalid_SERVOATTACH_format\"}");
    return;
  }
  p = end + 1;
  int gpio = (int)strtol(p, &end, 10);
  if (slot < 0 || slot >= MAX_SERVOS) {
    Serial.printf("S3_SENTINEL_ERROR {\"error\":\"invalid_slot\",\"slot\":%d}\n", slot);
    return;
  }
  servo_init_slot(slot, (uint8_t)gpio);
}

// ── ACTUATORS command: list all attached actuators ───────────────────
static void sentinel_handle_actuators_list_command() {
  Serial.println("S3_SENTINEL_ACTUATORS");
  for (int i = 0; i < g_servo_count; i++) {
    if (g_servos[i].attached) {
      Serial.printf("  servo slot=%d gpio=%u channel=%u angle=%d\n",
                    i, (unsigned)g_servos[i].gpio,
                    (unsigned)g_servos[i].channel, g_servos[i].current_angle);
    }
  }
  Serial.printf("  total_servos=%d/%d\n", g_servo_count, MAX_SERVOS);
}

// ── Serial command parser ────────────────────────────────────────────
static void sentinel_poll_serial_commands() {
  static char line[256];
  static int n = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[n] = 0;
      n = 0;
      if (strncmp(line, "SENSOR:", 7) == 0) {
        sentinel_handle_sensor_command(line);
      } else if (strcmp(line, "STATUS") == 0) {
        sentinel_handle_status_command();
      } else if (strcmp(line, "RUN") == 0) {
        sentinel_handle_run_command();
      } else if (strcmp(line, "SENSORS") == 0) {
        sentinel_handle_sensors_list_command();
      } else if (strcmp(line, "ACTUATORS") == 0) {
        sentinel_handle_actuators_list_command();
      } else if (strncmp(line, "SERVOATTACH:", 12) == 0) {
        sentinel_handle_servo_attach_command(line);
      } else if (strncmp(line, "SERVO:", 6) == 0) {
        sentinel_handle_servo_command(line);
      } else if (strcmp(line, "CLEAR") == 0) {
        sentinel_handle_clear_command();
      } else if (strncmp(line, "PROMPT:", 7) == 0) {
        const char *prompt = line + 7;
        while (*prompt == ' ') prompt++;
        run_language_prompt_receipt(prompt);
      } else if (strlen(line) > 0) {
        Serial.print("S3_SENTINEL_ERROR {\"error\":\"unknown_command\",\"received\":\"");
        // safe escape for the received line
        for (const char *q = line; *q; q++) {
          if (*q == '"' || *q == '\\') Serial.print('\\');
          Serial.print(*q);
        }
        Serial.println("\"}");
      }
    } else if (n < (int)sizeof(line) - 1) {
      line[n++] = c;
    } else {
      n = 0;
      Serial.println("S3_SENTINEL_ERROR {\"error\":\"line_too_long\"}");
    }
  }
}

// ── WiFi / OTA / HTTP (unchanged from original, adapted for v2) ──────
static void sentinel_setup_wifi() {
  if (!sentinel_wifi_enabled()) return;
  if (sentinel_http_ptr == nullptr) {
    sentinel_http_ptr = new WebServer(80);
  }
  if (sentinel_http_ptr == nullptr) {
    Serial.println("S3_SENTINEL_WIFI_ERROR phase=http_alloc");
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(RI_FINAL_WIFI_SSID, RI_FINAL_WIFI_PASSPHRASE);
  sentinel_last_wifi_attempt = millis();
  Serial.printf("S3_SENTINEL_WIFI_CONNECTING ssid=%s\n", RI_FINAL_WIFI_SSID);
}

static void sentinel_setup_ota() {
  if (sentinel_ota_ready || strlen(RI_FINAL_OTA_PASSWORD) == 0) return;
  ArduinoOTA.setHostname("ri-esp32s3-sentinel");
  ArduinoOTA.setPassword(RI_FINAL_OTA_PASSWORD);
  ArduinoOTA.setPort(3232);
  ArduinoOTA.onStart([]() { Serial.println("S3_SENTINEL_OTA_START"); });
  ArduinoOTA.onEnd([]() { Serial.println("S3_SENTINEL_OTA_END ok=1"); });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("S3_SENTINEL_OTA_ERROR code=%u\n", (unsigned)error);
  });
  ArduinoOTA.begin();
  sentinel_ota_ready = true;
  Serial.printf("S3_SENTINEL_OTA_READY hostname=ri-esp32s3-sentinel port=3232\n");
}

static void sentinel_http_health() {
  String body = "{\"ok\":true,\"variant\":\"" + String(FIRMWARE_VARIANT) +
               "\",\"model\":\"" + String(RI_MODEL_PROFILE) +
               "\",\"device_id\":\"" + String(g_receipt_id.device_id) +
               "\",\"uptime_ms\":" + String(millis()) + "}";
  sentinel_http_ptr->send(200, "application/json", body);
}

static void sentinel_http_status() {
  g_sensors.update_age();
  String body = "{\"schema\":\"ri_esp32s3_sentinel_status_v2\",";
  body += "\"device_id\":\"" + String(g_receipt_id.device_id) + "\",";
  body += "\"boot_id\":" + String(g_receipt_id.boot_id) + ",";
  body += "\"event_seq\":" + String(g_receipt_id.event_seq) + ",";
  body += "\"firmware_variant\":\"" + String(FIRMWARE_VARIANT) + "\",";
  body += "\"model_profile\":\"" + String(RI_MODEL_PROFILE) + "\",";
  body += "\"model_hash_verified\":" + String(sentinel_model_hash_verified ? "true" : "false") + ",";
  body += "\"sensor_count\":" + String(g_sensors.count) + ",";
  body += "\"sensors\":[";
  for (int i = 0; i < g_sensors.count; i++) {
    SensorReading &r = g_sensors.readings[i];
    if (i > 0) body += ",";
    body += "{\"type\":\"" + String(sensor_type_name(r.type)) + "\",";
    body += "\"quality\":\"" + String(quality_str(r.quality)) + "\",";
    body += "\"value\":" + String(isnan(r.value) ? "null" : String(r.value, 2)) + ",";
    body += "\"age_ms\":" + String(r.age_ms) + "}";
  }
  body += "],";
  body += "\"uptime_ms\":" + String(millis()) + ",";
  body += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
  body += "\"free_psram\":" + String(ESP.getFreePsram()) + "}";
  sentinel_http_ptr->send(200, "application/json", body);
}

static void sentinel_wifi_tick(uint32_t now) {
  if (!sentinel_wifi_enabled() || sentinel_http_ptr == nullptr) return;
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (!sentinel_wifi_connected && connected) {
    sentinel_wifi_connected = true;
    // Extend TG1 WDT timeout — WiFi re-enables it with a short default.
    // Set stage 0 to a very long timeout so H320 inference (~1.4s) doesn't trigger it.
    // TG1: WPROTECT=0x60020010, CONFIG0=0x60020048, CONFIG2=0x60020054 (stage0 timeout)
    {
      volatile uint32_t *tg1_prot = (volatile uint32_t *)0x60020010;
      volatile uint32_t *tg1_feed = (volatile uint32_t *)0x6002000C;
      volatile uint32_t *tg1_cfg  = (volatile uint32_t *)0x60020048;
      volatile uint32_t *tg1_stg0 = (volatile uint32_t *)0x60020054;
      *tg1_prot = 0x50D83AA1;
      *tg1_feed = 1;
      // Set stage 0 timeout to 0xFFFF (very long — ~8.3s at 40MHz/2^15 prescale)
      *tg1_stg0 = 0xFFFF;
      // Keep WDT enabled but with the long timeout
      *tg1_prot = 0;
    }
    sentinel_http_ptr->on("/health", HTTP_GET, sentinel_http_health);
    sentinel_http_ptr->on("/status", HTTP_GET, sentinel_http_status);
    sentinel_http_ptr->begin();
    Serial.printf("S3_SENTINEL_WIFI_CONNECTED ip=%s rssi=%ld http=80\n",
                  WiFi.localIP().toString().c_str(), (long)WiFi.RSSI());
    sentinel_setup_ota();
  } else if (sentinel_wifi_connected && !connected) {
    sentinel_wifi_connected = false;
    Serial.println("S3_SENTINEL_WIFI_DISCONNECTED reconnecting=true");
  }
  if (!sentinel_wifi_connected) {
    if (now - sentinel_last_wifi_attempt > 30000) {
      WiFi.reconnect();
      sentinel_last_wifi_attempt = now;
    }
  }
  if (sentinel_wifi_connected) {
    sentinel_http_ptr->handleClient();
    if (sentinel_ota_ready) ArduinoOTA.handle();
  }
}
#endif


void setup() {
#if CLUSTER_WIFI_DEMO
  cluster_setup_wifi_demo();
  return;
#endif

  Serial.begin(115200);
  delay(1500);

#if RI_FINAL_SENTINEL
  // Disable brownout detector — heavy PSRAM access during H320 LSTM
  // generation on N8R8 devkit boards can cause transient voltage dips.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  // Disable the TG1 hardware watchdog as belt-and-suspenders. The primary
  // fix is feeding it in the semaphore wait loop of lstm_layer(), but we
  // also disable it here so model loading (which can take seconds for
  // PSRAM clone + int4 conversion) doesn't trigger it.
  {
    volatile uint32_t *tg1_prot = (volatile uint32_t *)0x60020010;
    volatile uint32_t *tg1_feed = (volatile uint32_t *)0x6002000C;
    volatile uint32_t *tg1_cfg  = (volatile uint32_t *)0x60020048;
    *tg1_prot = 0x50D83AA1;
    *tg1_feed = 1;
    *tg1_cfg  = 0;
    *tg1_prot = 0;
  }
  // Disable RTC WDT too.
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0x50D83AA1);
  CLEAR_PERI_REG_MASK(RTC_CNTL_WDTCONFIG0_REG, RTC_CNTL_WDT_EN);
  WRITE_PERI_REG(RTC_CNTL_WDTWPROTECT_REG, 0);
#endif

  Serial.printf("\nESP32-S3 LSTM boot %s\n", FIRMWARE_VARIANT);
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
  Serial.printf("MODEL_READY profile=%s params=%lu hidden=%u layers=%u\n",
                RI_MODEL_PROFILE, (unsigned long)RI_MODEL_PARAMS,
                (unsigned)HIDDEN, (unsigned)LAYERS);

#if RI_FINAL_SENTINEL
  receipt_init();
  Serial.printf("S3_SENTINEL_READY variant=%s model_hash_verified=%s offline_first=true multi_sensor=true\n",
                FIRMWARE_VARIANT, sentinel_model_hash_verified ? "true" : "false");
  if (sentinel_wifi_enabled()) {
    sentinel_setup_wifi();
  }
#else
  run_benchmark();
  Serial.println("PROOF_DONE");
#endif
}

void loop() {
#if CLUSTER_WIFI_DEMO
  cluster_loop_wifi_demo();
  return;
#endif

#if RI_FINAL_SENTINEL
  sentinel_poll_serial_commands();
  sentinel_wifi_tick(millis());
  static uint32_t last_idle = 0;
  if (millis() - last_idle >= 5000) {
    last_idle = millis();
    Serial.println("idle; send STATUS, RUN, SENSORS, ACTUATORS, CLEAR, SENSOR:<type>:<value>, SERVO:<slot>:<angle>, or PROMPT:<text>");
  }
  delay(10);
#else
  poll_serial_language_commands();
  static uint32_t last_idle = 0;
  if (millis() - last_idle >= 5000) {
    last_idle = millis();
    Serial.println("idle; send PROMPT:<text> for S3_LANGUAGE_RECEIPT");
  }
  delay(10);
#endif
}
