# TinyStories H512 OTA shard update path

Status: build-ready, not live-flashed in this pass.

Purpose:

Add an over-the-air path for model/shard data, not only app firmware.
This keeps future worker updates USB-free once a coordinator relay-capable firmware is installed.

Implemented pieces:

1. H512 LSTM shard firmware envs

```bash
pio run -e cluster_coord_ap_lstm_shard
pio run -e cluster_worker1_ap_lstm_shard
pio run -e cluster_worker2_ap_lstm_shard
```

All three build as of this report.

2. Gate-row shard exporter

```bash
python3 tools/export_lstm_gate_shards.py \
  --weights weights_h512_p7_backup_709ff8.bin \
  --out-dir shards/tinystories-h512-lstm
```

Output format: RIWS v1 (`0x53574952`, ASCII `RIWS`), one shard per worker.

Generated receipt:

```text
source_sha256=709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61
worker1 shard: 2384308 bytes sha256=62e3a3b287e8da7c24f5da3f21feb9589fa890009fbb7beaf5913233139f3d9b rows=0-1023 tensors=12
worker2 shard: 2384308 bytes sha256=2af693549024b717110f6ca1a2369d6644228da4bdd9decfdc9439872a00682f rows=1024-2047 tensors=12
```

3. Firmware data OTA endpoint

Each cluster firmware with `CLUSTER_ENABLE_HTTP_UPDATE=1` now exposes:

```text
/update          app firmware OTA
/update_weights  data/weights partition update
```

`/update_weights` erases and writes the `weights` data partition (`data, subtype 0x40, label=weights`). It accepts multipart upload and writes the received bytes directly to flash.

4. Coordinator serial relay supports data target

The coordinator accepts:

```text
CLUSTER_RELAY_UPDATE board=<1|2> size=<bytes> target=weights
```

and forwards the byte stream to the selected worker as multipart HTTP POST `/update_weights`.

Host dry-run examples:

```bash
python3 tools/relay_worker_update.py \
  --role worker1 --mode lstm_shard --target weights \
  --artifact shards/tinystories-h512-lstm/worker1_lstm_gate_shard.riws \
  --port /dev/ttyACM0

python3 tools/relay_worker_update.py \
  --role worker2 --mode lstm_shard --target weights \
  --artifact shards/tinystories-h512-lstm/worker2_lstm_gate_shard.riws \
  --port /dev/ttyACM0
```

Execution form after coordinator/worker LSTM shard app firmware is installed:

```bash
python3 tools/relay_worker_update.py \
  --role worker1 --mode lstm_shard --target weights \
  --artifact shards/tinystories-h512-lstm/worker1_lstm_gate_shard.riws \
  --port /dev/ttyACM0 --wait-worker --relay-timeout 420 --execute

python3 tools/relay_worker_update.py \
  --role worker2 --mode lstm_shard --target weights \
  --artifact shards/tinystories-h512-lstm/worker2_lstm_gate_shard.riws \
  --port /dev/ttyACM0 --wait-worker --relay-timeout 420 --execute
```

5. Worker shard validation on boot

`cluster_worker*_ap_lstm_shard` firmware maps the `weights` partition and validates RIWS header fields:

- magic/version
- worker id matches board id
- hidden/layer dimensions match firmware
- tensor count is sane
- gate row range is printed

Expected receipt after a successful shard data OTA and reboot:

```text
CLUSTER_MODEL_WORKER_LSTM_SHARD_READY board_id=1 source=weights_partition format=RIWSv1 hidden=512 layers=3 rows=0-1023 tensors=12 ...
CLUSTER_MODEL_WORKER_LSTM_SHARD_READY board_id=2 source=weights_partition format=RIWSv1 hidden=512 layers=3 rows=1024-2047 tensors=12 ...
```

Verification run from this pass:

```text
python3 -m py_compile tools/*.py: PASS
python3 tools/test_cluster_protocol.py: PASS packet encode/decode/crc
pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard: SUCCESS
python3 tools/export_lstm_gate_shards.py --weights weights_h512_p7_backup_709ff8.bin --out-dir shards/tinystories-h512-lstm: SUCCESS
python3 tools/relay_worker_update.py --role worker1 --mode lstm_shard --target weights --artifact shards/tinystories-h512-lstm/worker1_lstm_gate_shard.riws --port /dev/ttyACM0: DRY_RUN OK
```

Boundary:

This commit adds the app+data OTA plumbing and build-ready H512 LSTM gate-shard artifacts. It does not yet claim live distributed recurrent H512 generation or a live worker shard-data OTA receipt. The current live TinyStories proof remains single-board H512. The current live cluster inference proof remains H256 output-head sharding.
