#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace cluster_protocol {

// Packet bytes are little-endian. The magic value serializes as ASCII "RIEC":
// bytes 0x52 0x49 0x45 0x43, integer value 0x43454952 on little-endian hosts.
static constexpr uint32_t CLUSTER_PACKET_MAGIC = 0x43454952u;
static constexpr uint8_t CLUSTER_PACKET_VERSION = 1;
static constexpr size_t CLUSTER_PACKET_HEADER_SIZE = 16;
static constexpr uint8_t CLUSTER_MATMUL_FIXTURE_ID = 1;
static constexpr uint8_t CLUSTER_MATMUL_FIXTURE_INT8_ID = 1;
static constexpr uint8_t CLUSTER_MATMUL_FIXTURE_INT4_ID = 2;
static constexpr size_t CLUSTER_MATMUL_VECTOR_LEN = 16;
static constexpr size_t CLUSTER_MATMUL_I4_PACKED_WEIGHT_LEN = CLUSTER_MATMUL_VECTOR_LEN / 2;
static constexpr size_t CLUSTER_MATMUL_REQUEST_PAYLOAD_SIZE = 1 + CLUSTER_MATMUL_VECTOR_LEN;
static constexpr size_t CLUSTER_MATMUL_RESULT_PAYLOAD_SIZE = 1 + 4;

enum ClusterMsgType : uint8_t {
  CLUSTER_MSG_PING = 1,
  CLUSTER_MSG_PONG = 2,
  CLUSTER_MSG_BROADCAST_VECTOR = 3,
  CLUSTER_MSG_MATMUL_REQUEST = 4,
  CLUSTER_MSG_MATMUL_RESULT = 5,
  CLUSTER_MSG_ERROR = 6,
  CLUSTER_MSG_FC_SHARD_REQUEST = 7,
  CLUSTER_MSG_FC_SHARD_RESULT = 8,
  CLUSTER_MSG_LSTM_GATE_REQUEST = 9,
  CLUSTER_MSG_LSTM_GATE_RESULT = 10,
};

struct ClusterPacketHeader {
  uint32_t magic;
  uint8_t version;
  uint8_t msg_type;
  uint8_t src_board;
  uint8_t dst_board;
  uint32_t seq;
  uint16_t payload_len;
  uint16_t crc16;
};

enum ClusterDecodeStatus : uint8_t {
  CLUSTER_DECODE_OK = 0,
  CLUSTER_DECODE_TRUNCATED,
  CLUSTER_DECODE_BAD_MAGIC,
  CLUSTER_DECODE_BAD_VERSION,
  CLUSTER_DECODE_BAD_LENGTH,
  CLUSTER_DECODE_BAD_CRC,
  CLUSTER_DECODE_BAD_ARGUMENT,
};

static inline uint16_t crc16_ccitt_false_update(uint16_t crc, uint8_t byte) {
  crc ^= (uint16_t)byte << 8;
  for (uint8_t bit = 0; bit < 8; bit++) {
    if (crc & 0x8000u) {
      crc = (uint16_t)((crc << 1) ^ 0x1021u);
    } else {
      crc = (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static inline uint16_t crc16_ccitt_false(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFFu;
  for (size_t i = 0; i < len; i++) {
    crc = crc16_ccitt_false_update(crc, data[i]);
  }
  return crc;
}

static inline uint16_t crc16_ccitt_false_append(uint16_t crc, const uint8_t *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    crc = crc16_ccitt_false_update(crc, data[i]);
  }
  return crc;
}

static inline void write_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xFFu);
  dst[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static inline void write_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xFFu);
  dst[1] = (uint8_t)((v >> 8) & 0xFFu);
  dst[2] = (uint8_t)((v >> 16) & 0xFFu);
  dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline uint16_t read_u16_le(const uint8_t *src) {
  return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

static inline uint32_t read_u32_le(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
         ((uint32_t)src[3] << 24);
}

static inline void write_i32_le(uint8_t *dst, int32_t v) {
  write_u32_le(dst, (uint32_t)v);
}

static inline int32_t read_i32_le(const uint8_t *src) {
  return (int32_t)read_u32_le(src);
}

static inline void write_f32_le(uint8_t *dst, float v) {
  uint32_t bits;
  memcpy(&bits, &v, sizeof(bits));
  write_u32_le(dst, bits);
}

static inline float read_f32_le(const uint8_t *src) {
  uint32_t bits = read_u32_le(src);
  float v;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static inline void write_header(uint8_t *dst, const ClusterPacketHeader &header) {
  write_u32_le(dst + 0, header.magic);
  dst[4] = header.version;
  dst[5] = header.msg_type;
  dst[6] = header.src_board;
  dst[7] = header.dst_board;
  write_u32_le(dst + 8, header.seq);
  write_u16_le(dst + 12, header.payload_len);
  write_u16_le(dst + 14, header.crc16);
}

static inline ClusterPacketHeader read_header(const uint8_t *src) {
  ClusterPacketHeader header;
  header.magic = read_u32_le(src + 0);
  header.version = src[4];
  header.msg_type = src[5];
  header.src_board = src[6];
  header.dst_board = src[7];
  header.seq = read_u32_le(src + 8);
  header.payload_len = read_u16_le(src + 12);
  header.crc16 = read_u16_le(src + 14);
  return header;
}

static inline uint16_t packet_crc16(const uint8_t *header_bytes, const uint8_t *payload,
                                    uint16_t payload_len) {
  uint8_t header_for_crc[CLUSTER_PACKET_HEADER_SIZE];
  memcpy(header_for_crc, header_bytes, CLUSTER_PACKET_HEADER_SIZE);
  header_for_crc[14] = 0;
  header_for_crc[15] = 0;

  uint16_t crc = 0xFFFFu;
  crc = crc16_ccitt_false_append(crc, header_for_crc, CLUSTER_PACKET_HEADER_SIZE);
  if (payload_len > 0 && payload != nullptr) {
    crc = crc16_ccitt_false_append(crc, payload, payload_len);
  }
  return crc;
}

static inline bool encode_packet(uint8_t msg_type, uint8_t src_board, uint8_t dst_board, uint32_t seq,
                                 const uint8_t *payload, uint16_t payload_len, uint8_t *out,
                                 size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr) return false;
  if (payload_len > 0 && payload == nullptr) return false;

  const size_t total_len = CLUSTER_PACKET_HEADER_SIZE + (size_t)payload_len;
  if (out_capacity < total_len) return false;

  ClusterPacketHeader header;
  header.magic = CLUSTER_PACKET_MAGIC;
  header.version = CLUSTER_PACKET_VERSION;
  header.msg_type = msg_type;
  header.src_board = src_board;
  header.dst_board = dst_board;
  header.seq = seq;
  header.payload_len = payload_len;
  header.crc16 = 0;

  write_header(out, header);
  if (payload_len > 0) {
    memcpy(out + CLUSTER_PACKET_HEADER_SIZE, payload, payload_len);
  }

  header.crc16 = packet_crc16(out, payload, payload_len);
  write_u16_le(out + 14, header.crc16);
  *out_len = total_len;
  return true;
}

static inline ClusterDecodeStatus decode_packet(const uint8_t *packet, size_t packet_len,
                                                ClusterPacketHeader *header_out,
                                                const uint8_t **payload_out,
                                                size_t *payload_len_out) {
  if (packet == nullptr || header_out == nullptr || payload_out == nullptr || payload_len_out == nullptr) {
    return CLUSTER_DECODE_BAD_ARGUMENT;
  }
  if (packet_len < CLUSTER_PACKET_HEADER_SIZE) return CLUSTER_DECODE_TRUNCATED;

  ClusterPacketHeader header = read_header(packet);
  if (header.magic != CLUSTER_PACKET_MAGIC) return CLUSTER_DECODE_BAD_MAGIC;
  if (header.version != CLUSTER_PACKET_VERSION) return CLUSTER_DECODE_BAD_VERSION;

  const size_t expected_len = CLUSTER_PACKET_HEADER_SIZE + (size_t)header.payload_len;
  if (packet_len < expected_len) return CLUSTER_DECODE_TRUNCATED;
  if (packet_len != expected_len) return CLUSTER_DECODE_BAD_LENGTH;

  const uint8_t *payload = packet + CLUSTER_PACKET_HEADER_SIZE;
  const uint16_t crc = packet_crc16(packet, payload, header.payload_len);
  if (crc != header.crc16) return CLUSTER_DECODE_BAD_CRC;

  *header_out = header;
  *payload_out = payload;
  *payload_len_out = header.payload_len;
  return CLUSTER_DECODE_OK;
}

static inline void fill_matmul_fixture_vector(int8_t *vector_out) {
  if (vector_out == nullptr) return;
  for (size_t i = 0; i < CLUSTER_MATMUL_VECTOR_LEN; i++) {
    vector_out[i] = (int8_t)((int)i - 8);
  }
}

static inline int8_t matmul_fixture_weight(uint8_t worker_board, size_t i) {
  if (worker_board == 1) return (int8_t)(i + 1);
  if (worker_board == 2) return (int8_t)(CLUSTER_MATMUL_VECTOR_LEN - i);
  return 0;
}

static inline int8_t clamp_i4(int value) {
  if (value < -8) return -8;
  if (value > 7) return 7;
  return (int8_t)value;
}

static inline uint8_t pack_i4_pair(int8_t low, int8_t high) {
  return (uint8_t)(((uint8_t)low & 0x0Fu) | (((uint8_t)high & 0x0Fu) << 4));
}

static inline int8_t unpack_i4_low(uint8_t packed) {
  int8_t value = (int8_t)(packed & 0x0Fu);
  return value >= 8 ? (int8_t)(value - 16) : value;
}

static inline int8_t unpack_i4_high(uint8_t packed) {
  int8_t value = (int8_t)((packed >> 4) & 0x0Fu);
  return value >= 8 ? (int8_t)(value - 16) : value;
}

static inline int8_t matmul_fixture_int4_weight_unpacked(uint8_t worker_board, size_t i) {
  if (worker_board == 1) return clamp_i4((int)(i % 8) - 4);
  if (worker_board == 2) return clamp_i4(3 - (int)(i % 8));
  return 0;
}

static inline uint8_t matmul_fixture_int4_weight_packed_byte(uint8_t worker_board, size_t byte_index) {
  const size_t low_i = byte_index * 2;
  const size_t high_i = low_i + 1;
  return pack_i4_pair(matmul_fixture_int4_weight_unpacked(worker_board, low_i),
                      matmul_fixture_int4_weight_unpacked(worker_board, high_i));
}

static inline int8_t matmul_fixture_int4_weight(uint8_t worker_board, size_t i) {
  const uint8_t packed = matmul_fixture_int4_weight_packed_byte(worker_board, i >> 1);
  return (i & 1) ? unpack_i4_high(packed) : unpack_i4_low(packed);
}

static inline int8_t matmul_fixture_weight(uint8_t fixture_id, uint8_t worker_board, size_t i) {
  if (fixture_id == CLUSTER_MATMUL_FIXTURE_INT8_ID) return matmul_fixture_weight(worker_board, i);
  if (fixture_id == CLUSTER_MATMUL_FIXTURE_INT4_ID) return matmul_fixture_int4_weight(worker_board, i);
  return 0;
}

static inline bool matmul_fixture_is_supported(uint8_t fixture_id) {
  return fixture_id == CLUSTER_MATMUL_FIXTURE_INT8_ID || fixture_id == CLUSTER_MATMUL_FIXTURE_INT4_ID;
}

static inline int32_t matmul_fixture_expected_dot(uint8_t worker_board,
                                                  uint8_t fixture_id = CLUSTER_MATMUL_FIXTURE_ID) {
  int8_t vector[CLUSTER_MATMUL_VECTOR_LEN];
  fill_matmul_fixture_vector(vector);
  int32_t dot = 0;
  for (size_t i = 0; i < CLUSTER_MATMUL_VECTOR_LEN; i++) {
    dot += (int32_t)vector[i] * (int32_t)matmul_fixture_weight(fixture_id, worker_board, i);
  }
  return dot;
}

static inline int32_t matmul_fixture_expected_gather(uint8_t fixture_id = CLUSTER_MATMUL_FIXTURE_ID) {
  return matmul_fixture_expected_dot(1, fixture_id) + matmul_fixture_expected_dot(2, fixture_id);
}

static inline bool encode_matmul_request_payload(uint8_t fixture_id, uint8_t *out,
                                                 size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr) return false;
  if (out_capacity < CLUSTER_MATMUL_REQUEST_PAYLOAD_SIZE) return false;
  out[0] = fixture_id;
  int8_t vector[CLUSTER_MATMUL_VECTOR_LEN];
  fill_matmul_fixture_vector(vector);
  memcpy(out + 1, vector, CLUSTER_MATMUL_VECTOR_LEN);
  *out_len = CLUSTER_MATMUL_REQUEST_PAYLOAD_SIZE;
  return true;
}

static inline bool decode_matmul_request_payload(const uint8_t *payload, size_t payload_len,
                                                 uint8_t *fixture_id_out, int8_t *vector_out) {
  if (payload == nullptr || fixture_id_out == nullptr || vector_out == nullptr) return false;
  if (payload_len != CLUSTER_MATMUL_REQUEST_PAYLOAD_SIZE) return false;
  *fixture_id_out = payload[0];
  memcpy(vector_out, payload + 1, CLUSTER_MATMUL_VECTOR_LEN);
  return true;
}

static inline bool encode_matmul_result_payload(uint8_t fixture_id, int32_t dot, uint8_t *out,
                                                size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr) return false;
  if (out_capacity < CLUSTER_MATMUL_RESULT_PAYLOAD_SIZE) return false;
  out[0] = fixture_id;
  write_i32_le(out + 1, dot);
  *out_len = CLUSTER_MATMUL_RESULT_PAYLOAD_SIZE;
  return true;
}

static inline bool decode_matmul_result_payload(const uint8_t *payload, size_t payload_len,
                                                uint8_t *fixture_id_out, int32_t *dot_out) {
  if (payload == nullptr || fixture_id_out == nullptr || dot_out == nullptr) return false;
  if (payload_len != CLUSTER_MATMUL_RESULT_PAYLOAD_SIZE) return false;
  *fixture_id_out = payload[0];
  *dot_out = read_i32_le(payload + 1);
  return true;
}

static constexpr size_t CLUSTER_FC_HIDDEN = 256;
static constexpr size_t CLUSTER_FC_REQUEST_PAYLOAD_SIZE = 1 + 4 + CLUSTER_FC_HIDDEN;
static constexpr size_t CLUSTER_FC_RESULT_PAYLOAD_SIZE = 1 + 1 + 4 + 1 + 1;

static constexpr size_t CLUSTER_LSTM_HIDDEN = 512;
static constexpr size_t CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE = 1 + 4 + 4 + CLUSTER_LSTM_HIDDEN + CLUSTER_LSTM_HIDDEN;
static constexpr size_t CLUSTER_LSTM_GATE_RESULT_HEADER_SIZE = 1 + 2 + 2;
static constexpr size_t CLUSTER_LSTM_GATE_RESULT_MAX_VALUES = 240;
static constexpr size_t CLUSTER_LSTM_GATE_RESULT_MAX_PAYLOAD_SIZE = CLUSTER_LSTM_GATE_RESULT_HEADER_SIZE + (CLUSTER_LSTM_GATE_RESULT_MAX_VALUES * 2);

static inline bool encode_lstm_gate_request_payload(uint8_t layer, float input_scale, float h_scale,
                                                    const int8_t *qx, const int8_t *qh, uint8_t *out,
                                                    size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr || qx == nullptr || qh == nullptr) return false;
  if (out_capacity < CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE) return false;
  out[0] = layer;
  write_f32_le(out + 1, input_scale);
  write_f32_le(out + 5, h_scale);
  memcpy(out + 9, qx, CLUSTER_LSTM_HIDDEN);
  memcpy(out + 9 + CLUSTER_LSTM_HIDDEN, qh, CLUSTER_LSTM_HIDDEN);
  *out_len = CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE;
  return true;
}

static inline bool decode_lstm_gate_request_payload(const uint8_t *payload, size_t payload_len,
                                                    uint8_t *layer_out, float *input_scale_out,
                                                    float *h_scale_out, int8_t *qx_out, int8_t *qh_out) {
  if (payload == nullptr || layer_out == nullptr || input_scale_out == nullptr ||
      h_scale_out == nullptr || qx_out == nullptr || qh_out == nullptr) return false;
  if (payload_len != CLUSTER_LSTM_GATE_REQUEST_PAYLOAD_SIZE) return false;
  *layer_out = payload[0];
  *input_scale_out = read_f32_le(payload + 1);
  *h_scale_out = read_f32_le(payload + 5);
  memcpy(qx_out, payload + 9, CLUSTER_LSTM_HIDDEN);
  memcpy(qh_out, payload + 9 + CLUSTER_LSTM_HIDDEN, CLUSTER_LSTM_HIDDEN);
  return true;
}

static inline bool encode_lstm_gate_result_payload(uint8_t layer, uint16_t row_start,
                                                   const int16_t *values, uint16_t count,
                                                   uint8_t *out, size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr || values == nullptr) return false;
  if (count > CLUSTER_LSTM_GATE_RESULT_MAX_VALUES) return false;
  const size_t needed = CLUSTER_LSTM_GATE_RESULT_HEADER_SIZE + (size_t)count * 2;
  if (out_capacity < needed) return false;
  out[0] = layer;
  write_u16_le(out + 1, row_start);
  write_u16_le(out + 3, count);
  for (uint16_t i = 0; i < count; i++) write_u16_le(out + 5 + (size_t)i * 2, (uint16_t)values[i]);
  *out_len = needed;
  return true;
}

static inline bool decode_lstm_gate_result_payload(const uint8_t *payload, size_t payload_len,
                                                   uint8_t *layer_out, uint16_t *row_start_out,
                                                   int16_t *values_out, uint16_t *count_out) {
  if (payload == nullptr || layer_out == nullptr || row_start_out == nullptr ||
      values_out == nullptr || count_out == nullptr) return false;
  if (payload_len < CLUSTER_LSTM_GATE_RESULT_HEADER_SIZE) return false;
  uint16_t count = read_u16_le(payload + 3);
  if (count > CLUSTER_LSTM_GATE_RESULT_MAX_VALUES) return false;
  if (payload_len != CLUSTER_LSTM_GATE_RESULT_HEADER_SIZE + (size_t)count * 2) return false;
  *layer_out = payload[0];
  *row_start_out = read_u16_le(payload + 1);
  *count_out = count;
  for (uint16_t i = 0; i < count; i++) values_out[i] = (int16_t)read_u16_le(payload + 5 + (size_t)i * 2);
  return true;
}

static inline bool encode_fc_shard_request_payload(uint8_t prompt_id, float hidden_scale,
                                                   const int8_t *hidden_q8, uint8_t *out,
                                                   size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr || hidden_q8 == nullptr) return false;
  if (out_capacity < CLUSTER_FC_REQUEST_PAYLOAD_SIZE) return false;
  out[0] = prompt_id;
  write_f32_le(out + 1, hidden_scale);
  memcpy(out + 5, hidden_q8, CLUSTER_FC_HIDDEN);
  *out_len = CLUSTER_FC_REQUEST_PAYLOAD_SIZE;
  return true;
}

static inline bool decode_fc_shard_request_payload(const uint8_t *payload, size_t payload_len,
                                                   uint8_t *prompt_id_out, float *hidden_scale_out,
                                                   int8_t *hidden_q8_out) {
  if (payload == nullptr || prompt_id_out == nullptr || hidden_scale_out == nullptr || hidden_q8_out == nullptr) return false;
  if (payload_len != CLUSTER_FC_REQUEST_PAYLOAD_SIZE) return false;
  *prompt_id_out = payload[0];
  *hidden_scale_out = read_f32_le(payload + 1);
  memcpy(hidden_q8_out, payload + 5, CLUSTER_FC_HIDDEN);
  return true;
}

static inline bool fc_shard_range_for_worker(uint8_t worker_board, uint8_t *start_out, uint8_t *end_out) {
  if (start_out == nullptr || end_out == nullptr) return false;
  if (worker_board == 1) { *start_out = 0; *end_out = 16; return true; }
  if (worker_board == 2) { *start_out = 17; *end_out = 32; return true; }
  return false;
}

static inline bool encode_fc_shard_result_payload(uint8_t prompt_id, uint8_t best_token,
                                                  float best_logit, uint8_t shard_start,
                                                  uint8_t shard_end, uint8_t *out,
                                                  size_t out_capacity, size_t *out_len) {
  if (out == nullptr || out_len == nullptr) return false;
  if (out_capacity < CLUSTER_FC_RESULT_PAYLOAD_SIZE) return false;
  out[0] = prompt_id;
  out[1] = best_token;
  write_f32_le(out + 2, best_logit);
  out[6] = shard_start;
  out[7] = shard_end;
  *out_len = CLUSTER_FC_RESULT_PAYLOAD_SIZE;
  return true;
}

static inline bool decode_fc_shard_result_payload(const uint8_t *payload, size_t payload_len,
                                                  uint8_t *prompt_id_out, uint8_t *best_token_out,
                                                  float *best_logit_out, uint8_t *shard_start_out,
                                                  uint8_t *shard_end_out) {
  if (payload == nullptr || prompt_id_out == nullptr || best_token_out == nullptr ||
      best_logit_out == nullptr || shard_start_out == nullptr || shard_end_out == nullptr) return false;
  if (payload_len != CLUSTER_FC_RESULT_PAYLOAD_SIZE) return false;
  *prompt_id_out = payload[0];
  *best_token_out = payload[1];
  *best_logit_out = read_f32_le(payload + 2);
  *shard_start_out = payload[6];
  *shard_end_out = payload[7];
  return true;
}

}  // namespace cluster_protocol
