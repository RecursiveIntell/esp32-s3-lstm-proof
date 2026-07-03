#!/usr/bin/env python3

import struct


MAGIC = 0x43454952
VERSION = 1
HEADER_FORMAT = "<IBBBBIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

PING = 1
PONG = 2
BROADCAST_VECTOR = 3
MATMUL_REQUEST = 4
MATMUL_RESULT = 5
ERROR = 6
FC_SHARD_REQUEST = 7
FC_SHARD_RESULT = 8
LSTM_GATE_REQUEST = 9
LSTM_GATE_RESULT = 10
MATMUL_FIXTURE_ID = 1
MATMUL_FIXTURE_INT8_ID = 1
MATMUL_FIXTURE_INT4_ID = 2
MATMUL_VECTOR_LEN = 16
LSTM_HIDDEN = 512
LSTM_GATE_RESULT_MAX_VALUES = 240


class ProtocolError(ValueError):
    pass


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def packet_crc(header: bytes, payload: bytes) -> int:
    header_for_crc = bytearray(header)
    header_for_crc[14] = 0
    header_for_crc[15] = 0
    return crc16_ccitt_false(bytes(header_for_crc) + payload)


def encode_packet(msg_type: int, src_board: int, dst_board: int, seq: int, payload: bytes = b"") -> bytes:
    if len(payload) > 0xFFFF:
        raise ValueError("payload too large")

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        msg_type,
        src_board,
        dst_board,
        seq,
        len(payload),
        0,
    )
    crc = packet_crc(header, payload)
    return header[:14] + struct.pack("<H", crc) + payload


def decode_packet(packet: bytes) -> tuple[dict[str, int], bytes]:
    if len(packet) < HEADER_SIZE:
        raise ProtocolError("truncated packet")

    magic, version, msg_type, src_board, dst_board, seq, payload_len, crc = struct.unpack(
        HEADER_FORMAT, packet[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != VERSION:
        raise ProtocolError("bad version")

    expected_len = HEADER_SIZE + payload_len
    if len(packet) < expected_len:
        raise ProtocolError("truncated packet")
    if len(packet) != expected_len:
        raise ProtocolError("bad length")

    payload = packet[HEADER_SIZE:]
    actual_crc = packet_crc(packet[:HEADER_SIZE], payload)
    if actual_crc != crc:
        raise ProtocolError("bad crc")

    return (
        {
            "magic": magic,
            "version": version,
            "msg_type": msg_type,
            "src_board": src_board,
            "dst_board": dst_board,
            "seq": seq,
            "payload_len": payload_len,
            "crc16": crc,
        },
        payload,
    )


def matmul_vector() -> list[int]:
    return [i - 8 for i in range(MATMUL_VECTOR_LEN)]


def matmul_weight(worker: int, i: int) -> int:
    if worker == 1:
        return i + 1
    if worker == 2:
        return MATMUL_VECTOR_LEN - i
    return 0


def clamp_i4(value: int) -> int:
    return min(7, max(-8, value))


def pack_i4_pair(low: int, high: int) -> int:
    return (low & 0x0F) | ((high & 0x0F) << 4)


def unpack_i4_low(packed: int) -> int:
    value = packed & 0x0F
    return value - 16 if value >= 8 else value


def unpack_i4_high(packed: int) -> int:
    value = (packed >> 4) & 0x0F
    return value - 16 if value >= 8 else value


def matmul_int4_weight_unpacked(worker: int, i: int) -> int:
    if worker == 1:
        return clamp_i4((i % 8) - 4)
    if worker == 2:
        return clamp_i4(3 - (i % 8))
    return 0


def matmul_int4_weight_packed_byte(worker: int, byte_index: int) -> int:
    low_i = byte_index * 2
    return pack_i4_pair(
        matmul_int4_weight_unpacked(worker, low_i),
        matmul_int4_weight_unpacked(worker, low_i + 1),
    )


def matmul_int4_weight(worker: int, i: int) -> int:
    packed = matmul_int4_weight_packed_byte(worker, i >> 1)
    return unpack_i4_high(packed) if i & 1 else unpack_i4_low(packed)


def fixture_weight(fixture: int, worker: int, i: int) -> int:
    if fixture == MATMUL_FIXTURE_INT8_ID:
        return matmul_weight(worker, i)
    if fixture == MATMUL_FIXTURE_INT4_ID:
        return matmul_int4_weight(worker, i)
    return 0


def matmul_expected_dot(worker: int, fixture: int = MATMUL_FIXTURE_ID) -> int:
    return sum(value * fixture_weight(fixture, worker, i) for i, value in enumerate(matmul_vector()))


def encode_matmul_request_payload(fixture: int = MATMUL_FIXTURE_ID) -> bytes:
    return struct.pack("<B16b", fixture, *matmul_vector())


def decode_matmul_result_payload(payload: bytes) -> tuple[int, int]:
    if len(payload) != 5:
        raise ProtocolError("bad matmul result length")
    return struct.unpack("<Bi", payload)



def encode_lstm_gate_request_payload(layer: int, input_scale: float, h_scale: float, qx: bytes, qh: bytes) -> bytes:
    if len(qx) != LSTM_HIDDEN or len(qh) != LSTM_HIDDEN:
        raise ValueError("bad q vector length")
    return struct.pack("<Bff", layer, input_scale, h_scale) + qx + qh


def decode_lstm_gate_request_payload(payload: bytes) -> tuple[int, float, float, bytes, bytes]:
    if len(payload) != 1 + 4 + 4 + LSTM_HIDDEN + LSTM_HIDDEN:
        raise ProtocolError("bad lstm gate request length")
    layer, input_scale, h_scale = struct.unpack("<Bff", payload[:9])
    qx = payload[9:9 + LSTM_HIDDEN]
    qh = payload[9 + LSTM_HIDDEN:]
    return layer, input_scale, h_scale, qx, qh


def encode_lstm_gate_result_payload(layer: int, row_start: int, values: list[int]) -> bytes:
    if len(values) > LSTM_GATE_RESULT_MAX_VALUES:
        raise ValueError("too many gate values")
    return struct.pack("<BHH", layer, row_start, len(values)) + struct.pack("<" + "h" * len(values), *values)


def decode_lstm_gate_result_payload(payload: bytes) -> tuple[int, int, list[int]]:
    if len(payload) < 5:
        raise ProtocolError("bad lstm gate result length")
    layer, row_start, count = struct.unpack("<BHH", payload[:5])
    if count > LSTM_GATE_RESULT_MAX_VALUES or len(payload) != 5 + 2 * count:
        raise ProtocolError("bad lstm gate result length")
    values = list(struct.unpack("<" + "h" * count, payload[5:])) if count else []
    return layer, row_start, values

def expect_reject(packet: bytes, expected: str) -> None:
    try:
        decode_packet(packet)
    except ProtocolError as exc:
        if expected not in str(exc):
            raise AssertionError(f"expected {expected!r}, got {exc!s}") from exc
        return
    raise AssertionError(f"expected rejection for {expected!r}")


def test_ping_empty_payload() -> None:
    packet = encode_packet(PING, src_board=0, dst_board=1, seq=42)
    header, payload = decode_packet(packet)
    assert header["magic"] == MAGIC
    assert header["version"] == VERSION
    assert header["msg_type"] == PING
    assert header["src_board"] == 0
    assert header["dst_board"] == 1
    assert header["seq"] == 42
    assert header["payload_len"] == 0
    assert payload == b""


def test_payload_roundtrip() -> None:
    payload = bytes(range(256))
    packet = encode_packet(BROADCAST_VECTOR, src_board=0, dst_board=2, seq=7, payload=payload)
    header, decoded_payload = decode_packet(packet)
    assert header["msg_type"] == BROADCAST_VECTOR
    assert header["payload_len"] == len(payload)
    assert decoded_payload == payload


def test_bad_magic() -> None:
    packet = bytearray(encode_packet(PONG, src_board=1, dst_board=0, seq=9))
    packet[0] ^= 0xFF
    expect_reject(bytes(packet), "bad magic")


def test_corrupted_crc() -> None:
    packet = bytearray(encode_packet(MATMUL_RESULT, src_board=2, dst_board=0, seq=11, payload=b"abc"))
    packet[-1] ^= 0x01
    expect_reject(bytes(packet), "bad crc")


def test_truncated_packet() -> None:
    packet = encode_packet(MATMUL_REQUEST, src_board=0, dst_board=1, seq=12, payload=b"abcdef")
    expect_reject(packet[:-1], "truncated")
    expect_reject(packet[: HEADER_SIZE - 1], "truncated")


def test_matmul_fixture_payloads() -> None:
    assert matmul_expected_dot(1) == 272
    assert matmul_expected_dot(2) == -408
    request_payload = encode_matmul_request_payload()
    packet = encode_packet(MATMUL_REQUEST, src_board=0, dst_board=1, seq=13, payload=request_payload)
    header, payload = decode_packet(packet)
    fixture_id, *vector = struct.unpack("<B16b", payload)
    assert header["msg_type"] == MATMUL_REQUEST
    assert fixture_id == MATMUL_FIXTURE_ID
    assert vector == matmul_vector()

    result_payload = struct.pack("<Bi", MATMUL_FIXTURE_ID, matmul_expected_dot(1))
    fixture_id, dot = decode_matmul_result_payload(result_payload)
    assert fixture_id == MATMUL_FIXTURE_ID
    assert dot == 272


def test_matmul_int4_fixture_math() -> None:
    assert [matmul_int4_weight(1, i) for i in range(MATMUL_VECTOR_LEN)] == [
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
    ]
    assert [matmul_int4_weight(2, i) for i in range(MATMUL_VECTOR_LEN)] == [
        3,
        2,
        1,
        0,
        -1,
        -2,
        -3,
        -4,
        3,
        2,
        1,
        0,
        -1,
        -2,
        -3,
        -4,
    ]
    assert [matmul_int4_weight_packed_byte(1, i) for i in range(MATMUL_VECTOR_LEN // 2)] == [
        pack_i4_pair(-4, -3),
        pack_i4_pair(-2, -1),
        pack_i4_pair(0, 1),
        pack_i4_pair(2, 3),
        pack_i4_pair(-4, -3),
        pack_i4_pair(-2, -1),
        pack_i4_pair(0, 1),
        pack_i4_pair(2, 3),
    ]
    assert matmul_expected_dot(1, MATMUL_FIXTURE_INT4_ID) == 88
    assert matmul_expected_dot(2, MATMUL_FIXTURE_INT4_ID) == -80
    assert matmul_expected_dot(1, MATMUL_FIXTURE_INT4_ID) + matmul_expected_dot(2, MATMUL_FIXTURE_INT4_ID) == 8

    request_payload = encode_matmul_request_payload(MATMUL_FIXTURE_INT4_ID)
    fixture_id, *vector = struct.unpack("<B16b", request_payload)
    assert fixture_id == MATMUL_FIXTURE_INT4_ID
    assert vector == matmul_vector()



def test_fc_shard_payloads() -> None:
    hidden = [max(-127, min(127, i - 128)) for i in range(256)]
    scale = 0.00390625
    payload = struct.pack("<Bf256b", 1, scale, *hidden)
    packet = encode_packet(FC_SHARD_REQUEST, src_board=0, dst_board=1, seq=21, payload=payload)
    header, decoded = decode_packet(packet)
    assert header["msg_type"] == FC_SHARD_REQUEST
    prompt_id, decoded_scale, *decoded_hidden = struct.unpack("<Bf256b", decoded)
    assert prompt_id == 1
    assert abs(decoded_scale - scale) < 1e-9
    assert decoded_hidden == hidden

    result_payload = struct.pack("<BBfBB", 1, 12, -0.25, 0, 16)
    packet = encode_packet(FC_SHARD_RESULT, src_board=1, dst_board=0, seq=21, payload=result_payload)
    header, decoded = decode_packet(packet)
    assert header["msg_type"] == FC_SHARD_RESULT
    assert struct.unpack("<BBfBB", decoded) == (1, 12, -0.25, 0, 16)

def main() -> None:
    test_ping_empty_payload()
    test_payload_roundtrip()
    test_bad_magic()
    test_corrupted_crc()
    test_truncated_packet()
    test_matmul_fixture_payloads()
    test_matmul_int4_fixture_math()
    test_fc_shard_payloads()
    print("PASS packet encode/decode/crc")


if __name__ == "__main__":
    main()


def test_lstm_gate_payloads() -> None:
    qx = bytes((i % 256 for i in range(LSTM_HIDDEN)))
    qh = bytes(((255 - i) % 256 for i in range(LSTM_HIDDEN)))
    payload = encode_lstm_gate_request_payload(2, 0.25, 0.5, qx, qh)
    packet = encode_packet(LSTM_GATE_REQUEST, src_board=0, dst_board=1, seq=44, payload=payload)
    header, decoded = decode_packet(packet)
    layer, input_scale, h_scale, got_qx, got_qh = decode_lstm_gate_request_payload(decoded)
    assert header["msg_type"] == LSTM_GATE_REQUEST
    assert layer == 2
    assert abs(input_scale - 0.25) < 1e-6
    assert abs(h_scale - 0.5) < 1e-6
    assert got_qx == qx
    assert got_qh == qh

    values = [i - 120 for i in range(LSTM_GATE_RESULT_MAX_VALUES)]
    result = encode_lstm_gate_result_payload(1, 480, values)
    packet = encode_packet(LSTM_GATE_RESULT, src_board=1, dst_board=0, seq=45, payload=result)
    header, decoded = decode_packet(packet)
    layer, row_start, got_values = decode_lstm_gate_result_payload(decoded)
    assert header["msg_type"] == LSTM_GATE_RESULT
    assert layer == 1
    assert row_start == 480
    assert got_values == values
