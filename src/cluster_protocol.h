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

enum ClusterMsgType : uint8_t {
  CLUSTER_MSG_PING = 1,
  CLUSTER_MSG_PONG = 2,
  CLUSTER_MSG_BROADCAST_VECTOR = 3,
  CLUSTER_MSG_MATMUL_REQUEST = 4,
  CLUSTER_MSG_MATMUL_RESULT = 5,
  CLUSTER_MSG_ERROR = 6,
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

}  // namespace cluster_protocol
