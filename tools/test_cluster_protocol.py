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
MATMUL_FIXTURE_ID = 1
MATMUL_VECTOR_LEN = 16


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


def matmul_expected_dot(worker: int) -> int:
    return sum(value * matmul_weight(worker, i) for i, value in enumerate(matmul_vector()))


def encode_matmul_request_payload() -> bytes:
    return struct.pack("<B16b", MATMUL_FIXTURE_ID, *matmul_vector())


def decode_matmul_result_payload(payload: bytes) -> tuple[int, int]:
    if len(payload) != 5:
        raise ProtocolError("bad matmul result length")
    return struct.unpack("<Bi", payload)


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


def main() -> None:
    test_ping_empty_payload()
    test_payload_roundtrip()
    test_bad_magic()
    test_corrupted_crc()
    test_truncated_packet()
    test_matmul_fixture_payloads()
    print("PASS packet encode/decode/crc")


if __name__ == "__main__":
    main()
