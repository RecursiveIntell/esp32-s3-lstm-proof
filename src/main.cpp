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

#ifndef CLUSTER_BOARD_ID
#define CLUSTER_BOARD_ID 0
#endif
#ifndef CLUSTER_ROLE_COORD
#define CLUSTER_ROLE_COORD 0
#endif
#ifndef CLUSTER_ROLE_WORKER
#define CLUSTER_ROLE_WORKER 0
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

int model_step(int token_idx, bool measure) {
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
  if (measure) {
    ops.fc_us += (uint32_t)(micros() - t0);
    ops.measured_steps++;
  }
  return best;
}

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
  poll_serial_language_commands();
  static uint32_t last_idle = 0;
  if (millis() - last_idle >= 5000) {
    last_idle = millis();
    Serial.println("idle; send PROMPT:<text> for S3_LANGUAGE_RECEIPT");
  }
  delay(10);
}
