#!/usr/bin/env python3
"""Export H512 LSTM gate-row shards for ESP32-S3 worker partitions.

This is the build-time artifact generator for recurrent/gate sharding. It does
not flash. It reads a RILM v1 weight pack and writes one compact binary shard per
worker containing the LSTM gate rows that worker owns.

Shard format RIWS v1 (little endian):
  magic u32 = 0x53574952  # 'RIWS'
  version u16 = 1
  worker u16
  hidden u32
  layers u32
  shard_start u32  # gate row start in [0, 4*hidden)
  shard_end u32    # inclusive gate row end
  tensor_count u32
  repeated tensors:
    name_len u16, name bytes
    dtype u8       # copied RILM dtype: 0=f32,1=i8,2=i4
    ndim u8
    dims u32[ndim] # rows are sliced for weight/bias tensors
    scale f32
    payload_len u32
    payload bytes

For mixed TinyStories H512, input matrices are already int4 and recurrent
matrices are int8. Row slicing preserves dtype/scale. The firmware side can map
or clone this shard later without requiring the full 4.8MB weight pack on every
worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC_RILM = 0x4D4C4952
MAGIC_RIWS = 0x53574952

@dataclass
class Tensor:
    name: str
    dtype: int
    dims: tuple[int, ...]
    scale: float
    payload: bytes


def read_rilm(path: Path) -> tuple[int, list[Tensor]]:
    data = path.read_bytes()
    off = 0
    magic, version, _reserved, count = struct.unpack_from('<IHHI', data, off)
    off += 12
    if magic != MAGIC_RILM:
        raise ValueError(f'bad RILM magic 0x{magic:08x}')
    tensors: list[Tensor] = []
    for _ in range(count):
        (name_len,) = struct.unpack_from('<H', data, off); off += 2
        name = data[off:off+name_len].decode(); off += name_len
        dtype = data[off]; ndim = data[off+1]; off += 2
        dims = struct.unpack_from('<' + 'I'*ndim, data, off); off += 4*ndim
        (scale,) = struct.unpack_from('<f', data, off); off += 4
        (payload_len,) = struct.unpack_from('<I', data, off); off += 4
        payload = data[off:off+payload_len]; off += payload_len
        tensors.append(Tensor(name, dtype, tuple(int(d) for d in dims), float(scale), payload))
    return version, tensors


def row_slice_payload(t: Tensor, row_start: int, row_end_excl: int) -> tuple[tuple[int, ...], bytes]:
    if len(t.dims) == 1:
        elem_size = 4 if t.dtype == 0 else 1
        return (row_end_excl - row_start,), t.payload[row_start*elem_size:row_end_excl*elem_size]
    if len(t.dims) != 2:
        raise ValueError(f'unsupported dims for {t.name}: {t.dims}')
    rows, cols = t.dims
    if row_end_excl > rows:
        raise ValueError(f'shard rows exceed {t.name}: {row_end_excl}>{rows}')
    if t.dtype == 2:
        # int4 rows. Current H512 cols=512 is even, so rows are byte-aligned.
        row_bytes = (cols + 1) // 2
    elif t.dtype == 1:
        row_bytes = cols
    elif t.dtype == 0:
        row_bytes = cols * 4
    else:
        raise ValueError(f'bad dtype {t.dtype}')
    return (row_end_excl - row_start, cols), t.payload[row_start*row_bytes:row_end_excl*row_bytes]


def write_shard(path: Path, worker: int, hidden: int, layers: int, start: int, end_excl: int, tensors: list[Tensor]) -> dict:
    selected: list[Tensor] = []
    for layer in range(layers):
        for stem in ('lstm.weight_ih_l', 'lstm.weight_hh_l', 'lstm.bias_ih_l', 'lstm.bias_hh_l'):
            name = f'{stem}{layer}'
            src = next(t for t in tensors if t.name == name)
            dims, payload = row_slice_payload(src, start, end_excl)
            selected.append(Tensor(name, src.dtype, dims, src.scale, payload))
    out = bytearray()
    out += struct.pack('<IHHIIIII', MAGIC_RIWS, 1, worker, hidden, layers, start, end_excl - 1, len(selected))
    for t in selected:
        nb = t.name.encode()
        out += struct.pack('<H', len(nb)) + nb
        out += struct.pack('<BB', t.dtype, len(t.dims))
        out += struct.pack('<' + 'I'*len(t.dims), *t.dims)
        out += struct.pack('<fI', t.scale, len(t.payload))
        out += t.payload
    path.write_bytes(out)
    return {
        'worker': worker,
        'path': str(path),
        'bytes': len(out),
        'sha256': hashlib.sha256(out).hexdigest(),
        'hidden': hidden,
        'layers': layers,
        'gate_row_start': start,
        'gate_row_end': end_excl - 1,
        'tensor_count': len(selected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights_h512_p7_backup_709ff8.bin')
    ap.add_argument('--out-dir', default='shards/tinystories-h512-lstm')
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    weights = Path(args.weights)
    if not weights.is_absolute(): weights = root / weights
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute(): out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _version, tensors = read_rilm(weights)
    total_gate_rows = 4 * args.hidden
    rows_per = total_gate_rows // args.workers
    receipts = []
    for worker in range(1, args.workers + 1):
        start = (worker - 1) * rows_per
        end = total_gate_rows if worker == args.workers else worker * rows_per
        receipts.append(write_shard(out_dir / f'worker{worker}_lstm_gate_shard.riws', worker, args.hidden, args.layers, start, end, tensors))
    receipt = {
        'schema': 'ri_esp32_lstm_gate_shards_v1',
        'source_weights': str(weights),
        'source_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'source_bytes': weights.stat().st_size,
        'workers': args.workers,
        'shards': receipts,
        'claim_boundary': 'Build artifact only until flashed/OTA-updated and live hardware verifier passes.',
    }
    (out_dir / 'shard_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
