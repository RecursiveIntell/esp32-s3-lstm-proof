# Cluster WiFi Ping Flash Manifest — 2026-07-03

Status: historical bootstrap manifest, superseded by later live receipts.

Purpose: exact images and commands for the first WiFi-normal-mode 3-board transport proof. This file is retained as the initial ping-flash manifest only; later commits proved matmul, H256 sharded FC inference, TinyStories H512 single-board generation, and build-ready H512 data-OTA shard plumbing.

## Runtime shape

- Coordinator runs as a WiFi access point/gateway.
- Workers join the coordinator AP.
- No internet is required.
- USB is only needed for flashing/debug/host-control.
- UDP port: `42100`.
- Packet framing: `src/cluster_protocol.h` CRC/seq packets.

Default public AP credentials for proof builds:

- SSID: `RI-ESP-CLUSTER`
- Passphrase: `localfirstai`

To override locally, copy:

```bash
cp src/wifi_secrets.local.h.example src/wifi_secrets.local.h
```

Then edit `src/wifi_secrets.local.h`. The real local file is gitignored.

## Flash order

Use one USB port and flash one board at a time.

### Board 0 — coordinator AP/gateway

Environment:

```text
cluster_coord_ap_ping
```

Firmware artifact:

```text
/home/sikmindz/projects/esp32-s3-lstm-proof/.pio/build/cluster_coord_ap_ping/firmware.bin
```

SHA256:

```text
f227c3c1a0c0d100e305f5c25475b0a808933287beb836ef7a2dac4a6b124614
```

Dry-run command:

```bash
python3 tools/flash_cluster_wifi.py --role coord --port /dev/ttyACM0
```

Actual flash command when Josh confirms:

```bash
python3 tools/flash_cluster_wifi.py --role coord --port /dev/ttyACM0 --execute
```

Expected serial signs after boot:

```text
CLUSTER_WIFI_BOOT board_id=0 role=coord ping_only=1
CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100
CLUSTER_WIFI_UDP_READY board_id=0 port=42100
CLUSTER_WIFI_PING_BROADCAST seq=...
```

### Board 1 — worker 1

Environment:

```text
cluster_worker1_ap_ping
```

Firmware artifact:

```text
/home/sikmindz/projects/esp32-s3-lstm-proof/.pio/build/cluster_worker1_ap_ping/firmware.bin
```

SHA256:

```text
e4337471168f3069fd3aa9b7e5cd8ba55b9a1f6896c8212e2a4e5f3017801699
```

Dry-run command:

```bash
python3 tools/flash_cluster_wifi.py --role worker1 --port /dev/ttyACM0
```

Actual flash command when Josh confirms:

```bash
python3 tools/flash_cluster_wifi.py --role worker1 --port /dev/ttyACM0 --execute
```

Expected serial signs after boot:

```text
CLUSTER_WIFI_BOOT board_id=1 role=worker ping_only=1
CLUSTER_WIFI_STA_CONNECTING board_id=1 ssid=RI-ESP-CLUSTER
CLUSTER_WIFI_WORKER_READY board_id=1 ip=192.168.4.x rssi=... port=42100
CLUSTER_WIFI_UDP_READY board_id=1 port=42100
CLUSTER_WIFI_PONG_SENT board_id=1 seq=...
```

### Board 2 — worker 2

Environment:

```text
cluster_worker2_ap_ping
```

Firmware artifact:

```text
/home/sikmindz/projects/esp32-s3-lstm-proof/.pio/build/cluster_worker2_ap_ping/firmware.bin
```

SHA256:

```text
48fa10a27ddd75ddecbd59725db7bc554a987374cbfee03a0fbde218d1f34216
```

Dry-run command:

```bash
python3 tools/flash_cluster_wifi.py --role worker2 --port /dev/ttyACM0
```

Actual flash command when Josh confirms:

```bash
python3 tools/flash_cluster_wifi.py --role worker2 --port /dev/ttyACM0 --execute
```

Expected serial signs after boot:

```text
CLUSTER_WIFI_BOOT board_id=2 role=worker ping_only=1
CLUSTER_WIFI_STA_CONNECTING board_id=2 ssid=RI-ESP-CLUSTER
CLUSTER_WIFI_WORKER_READY board_id=2 ip=192.168.4.x rssi=... port=42100
CLUSTER_WIFI_UDP_READY board_id=2 port=42100
CLUSTER_WIFI_PONG_SENT board_id=2 seq=...
```

## Controller verification already performed

```text
python3 -m py_compile tools/*.py: SUCCESS
python3 tools/flash_cluster_wifi.py --role coord --port /dev/ttyACM0: dry-run OK
python3 tools/flash_cluster_wifi.py --role worker1 --port /dev/ttyACM0: dry-run OK
python3 tools/flash_cluster_wifi.py --role worker2 --port /dev/ttyACM0: dry-run OK
pio run -e cluster_coord_ap_ping: SUCCESS
pio run -e cluster_worker1_ap_ping: SUCCESS
pio run -e cluster_worker2_ap_ping: SUCCESS
pio run -e esp32s3_lstm: SUCCESS
```

## What this proves after flashing

This first firmware set proves only WiFi transport readiness:

- coordinator AP without internet
- worker station join
- UDP broadcast PING
- CRC/seq validated PONG replies
- two workers can run from wall/battery after initial flash

Historical boundary for this first ping manifest: by itself it did not prove sharded matmul, H256 cluster inference, TinyStories, or performance scaling. Later receipts supersede that boundary:

- sharded matmul: proven in `THREE_BOARD_CLUSTER_RECEIPT_2026-07-02.md`
- H256 sharded FC inference: commit `d588f26`
- TinyStories H512 single-board run: commit `d0fa117`
- H512 app/data OTA shard path: commit `c82498e`
