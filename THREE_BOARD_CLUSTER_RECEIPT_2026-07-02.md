# Three-Board ESP32-S3 Cluster Receipt (2026-07-02)

Status: stub

## Purpose

Track the live implementation plan for a 3-board ESP32-S3 tensor-parallel cluster, including claims, hardware inventory, and phase evidence.
This deployment plan is intentionally targeted at users with minimal local infrastructure.

## Claim Boundary

- Do not claim TinyStories model behavior (or performance) until a validated TinyStories hardware run is recorded.
- Do not claim the 3-board cluster hardware path is production-ready until phase-level evidence confirms transport, sharding, and generation behavior across all boards.
- Do not claim internet or cloud access is required for local inference.
- Do claim that coordinator AP/gateway mode operates without upstream internet after provisioning.

## Phase 0 Evidence Completed

- Routing design doc: `docs/plans/hybrid-codex-fable-cluster-routing.md`
- Cleanup commit completed: `0200e1b`
- Parser fix commit completed: `61adcf0`
- p22 baseline commit: `2b4da43`
- p22 baseline mean: `39.51703333333333 char-token/s`, `25.30333333333333 ms/token` (3 runs)
- p22 baseline benchmark summary: `benchmarks/p22_i4_wih_whh_simd_h256/p22_i4_wih_whh_simd_h256-summary.json`

## Hardware Inventory (placeholders)

### Boards

- Board IDs:
  - `BOARD_0_ID: <TODO>`
  - `BOARD_1_ID: <TODO>`
  - `BOARD_2_ID: <TODO>`
- Serial ports:
  - `BOARD_0_PORT: <TODO>`
  - `BOARD_1_PORT: <TODO>`
  - `BOARD_2_PORT: <TODO>`
- Transport mode:
  - `TRANSPORT_MODE: WiFi`
  - `WIFI_MODE_LAN_OR_AP: <TODO> # Existing LAN or Coordinator AP`
  - `COORDINATOR_WIFI_SSID: <TODO>`
  - `COORDINATOR_WIFI_IP: <TODO>`
  - `WORKER_1_WIFI_IP: <TODO>`
  - `WORKER_2_WIFI_IP: <TODO>`
- USB usage:
  - `DEVELOPMENT_USB_TO_COORDINATOR_ONLY: true`
  - `WORKER_USB_REQUIRED_IN_RUN: false`
- WiFi mode claims:
  - `WIFI_UDP_PREFERRED: true`
  - `WIFI_TCP_FALLBACK_ALLOWED: true`
  - `CLUSTER_MODE_AP_WITHOUT_INTERNET: true`
- Model shard hashes:
  - `shard_0_hash: <TODO>`
  - `shard_1_hash: <TODO>`
  - `shard_2_hash: <TODO>`

## Phase 1: Multi-board transport proof

- [x] Task 1.1 board-role firmware variants
  - 2026-07-02: `pio run -e cluster_coord` — SUCCESS (`00:00:05.871`), role macros: `CLUSTER_ROLE_COORD=1`, `CLUSTER_BOARD_ID=0`
  - 2026-07-02: `pio run -e cluster_worker1` — SUCCESS (`00:00:08.123`), role macros: `CLUSTER_ROLE_WORKER=1`, `CLUSTER_BOARD_ID=1`
  - 2026-07-02: `pio run -e cluster_worker2` — SUCCESS (`00:00:08.173`), role macros: `CLUSTER_ROLE_WORKER=1`, `CLUSTER_BOARD_ID=2`
- [x] Task 1.2 packet framing and CRC format
  - 2026-07-03: `python3 tools/test_cluster_protocol.py` — SUCCESS (`PASS packet encode/decode/crc`)
  - 2026-07-03: `python3 -m py_compile tools/*.py` — SUCCESS
  - 2026-07-03: `pio run -e cluster_coord` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker1` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2` — SUCCESS
- [x] Task 1.3 WiFi worker PING/PONG server proof (UDP)
  - 2026-07-03: Phase 1.3 firmware artifacts prepared, not flashed.
  - Build envs:
    - `cluster_coord_ap_ping`: coordinator board `0`, SoftAP mode, UDP PING broadcast on port `42100`
    - `cluster_worker1_ap_ping`: worker board `1`, station mode, UDP PING/PONG on port `42100`
    - `cluster_worker2_ap_ping`: worker board `2`, station mode, UDP PING/PONG on port `42100`
  - Flash helper dry-run: `python3 tools/flash_cluster_wifi.py --role coord|worker1|worker2 --port <serial-port>`
  - No-flash status: hardware flashing intentionally not performed during artifact preparation.
  - 2026-07-03 coordinator flash: `python3 tools/flash_cluster_wifi.py --role coord --port /dev/ttyACM0 --execute` — SUCCESS.
  - Coordinator hardware: ESP32-S3 MAC `94:a9:90:d2:41:f4`.
  - Coordinator boot receipt: `CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100`.
  - Coordinator UDP receipt: `CLUSTER_WIFI_PING_BROADCAST seq=1 dst=192.168.4.255 port=42100 sent=true`.
  - 2026-07-03 worker1 flash: `python3 tools/flash_cluster_wifi.py --role worker1 --port /dev/ttyACM1 --execute` — SUCCESS.
  - Worker1 hardware: ESP32-S3 MAC `a4:cb:8f:d9:24:ec`.
  - Worker1 join/status receipt: `CLUSTER_WIFI_WORKER_STATUS board_id=1 ip=192.168.4.2 rssi=-55 port=42100`.
  - Worker1 reply receipt: `CLUSTER_WIFI_PING board=1 seq=129 from_board=0 from=192.168.4.1:42100 reply=sent rssi=-56`.
  - Coordinator receive receipt: `CLUSTER_WIFI_PONG src_board=1 seq=136 from=192.168.4.2:42100 rssi=0`.
  - 2026-07-03 worker2 flash: `python3 tools/flash_cluster_wifi.py --role worker2 --port /dev/ttyACM1 --execute` — SUCCESS.
  - Worker2 hardware: ESP32-S3 MAC `94:a9:90:d2:40:b0`.
  - Worker2 reply receipt: `CLUSTER_WIFI_PING board=2 seq=228 from_board=0 from=192.168.4.1:42100 reply=sent rssi=-43`.
  - Coordinator receive receipt: `CLUSTER_WIFI_PONG src_board=2 seq=229 from=192.168.4.3:42100 rssi=0`.
- [x] Task 1.4 two-worker WiFi barrier sync proof (UDP with fallback notes)
  - 2026-07-03 live AP-mode receipt: both workers powered from non-laptop power and coordinator on `/dev/ttyACM0`.
  - Coordinator broadcast: `CLUSTER_WIFI_PING_BROADCAST seq=431 dst=192.168.4.255 port=42100 sent=true`.
  - Worker1 receive: `CLUSTER_WIFI_PONG src_board=1 seq=431 from=192.168.4.2:42100 rssi=0`.
  - Worker2 receive: `CLUSTER_WIFI_PONG src_board=2 seq=431 from=192.168.4.3:42100 rssi=0`.
  - Result: coordinator observed both workers replying to the same UDP broadcast sequence in one live window.
  - Note: RGB LEDs are not driven by the current WiFi worker firmware; power LED on + coordinator PONG receipt is the verification signal.
- [x] Task 1.5 Coordinator AP and OTA provisioning plan
  - 2026-07-03: Provisioning plan completed: `docs/CLUSTER_AP_OTA_PROVISIONING_PLAN_2026-07-03.md`
  - Note: original phase scope was design/provisioning only.
- [x] Task 1.6 ArduinoOTA firmware path and host helper
  - 2026-07-03: OTA code implemented for WiFi cluster modes only (`CLUSTER_ENABLE_OTA=1` on AP ping and AP matmul envs).
  - Deterministic OTA hostnames:
    - coordinator: `ri-esp-cluster-coord`
    - worker1: `ri-esp-cluster-worker1`
    - worker2: `ri-esp-cluster-worker2`
  - Host helper added: `tools/ota_cluster_wifi.py`.
  - Expected dry-run examples:
    - `python3 tools/ota_cluster_wifi.py --role coord --mode matmul --ip 192.168.4.1`
    - `python3 tools/ota_cluster_wifi.py --role worker1 --mode matmul --ip 192.168.4.2`
    - `python3 tools/ota_cluster_wifi.py --role worker2 --mode matmul --ip 192.168.4.3`
  - 2026-07-03: `python3 -m py_compile tools/*.py` — SUCCESS
  - 2026-07-03: `python3 tools/test_cluster_protocol.py` — SUCCESS (`PASS packet encode/decode/crc`)
  - 2026-07-03: OTA helper dry-runs for coordinator, worker1, and worker2 matmul IPs — SUCCESS
  - 2026-07-03: `pio run -e cluster_coord_ap_ping` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker1_ap_ping` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2_ap_ping` — SUCCESS
  - 2026-07-03: `pio run -e cluster_coord_ap_matmul` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker1_ap_matmul` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2_ap_matmul` — SUCCESS
  - No live OTA upload receipt yet. Controller must first USB-flash OTA-capable firmware, then push an update over WiFi before live OTA proof can be marked complete.

## Phase 2: Sharded matmul proof with synthetic data

- [x] Task 2.1 deterministic matmul fixture
  - 2026-07-03: Firmware fixture artifacts prepared, not flashed.
  - Payloads:
    - request: `fixture_id:uint8`, `vector:int8[16]`
    - result: `fixture_id:uint8`, `dot:int32_le`
  - Fixture math is computed in code:
    - `vector[i] = i - 8`
    - worker1 weights: `i + 1`, expected dot `272`
    - worker2 weights: `16 - i`, expected dot `-408`
    - gathered expected total `-136`
  - Protocol check: `python3 tools/test_cluster_protocol.py` covers packet framing plus matmul fixture payload shape.
  - 2026-07-03: `python3 tools/test_cluster_protocol.py` — SUCCESS (`PASS packet encode/decode/crc`)
  - 2026-07-03: `python3 -m py_compile tools/*.py` — SUCCESS
- [x] Task 2.2 worker matmul command
  - 2026-07-03: `cluster_worker1_ap_matmul` and `cluster_worker2_ap_matmul` build envs added.
  - Workers decode `CLUSTER_MSG_MATMUL_REQUEST`, compute deterministic int8 dot shards, and reply with `CLUSTER_MSG_MATMUL_RESULT`.
  - Worker serial receipt format: `CLUSTER_MATMUL_WORKER board=... seq=... fixture=... dot=... expected=... ok=... reply=...`.
  - 2026-07-03: `pio run -e cluster_worker1_ap_matmul` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2_ap_matmul` — SUCCESS
- [x] Task 2.3 coordinator gather for 3 shards
  - 2026-07-03: `cluster_coord_ap_matmul` build env added.
  - Coordinator AP broadcasts deterministic per-worker `CLUSTER_MSG_MATMUL_REQUEST` packets and gathers worker1 + worker2 results.
  - Coordinator receipt formats:
    - `CLUSTER_MATMUL_REQUEST seq=... fixture=... dst=... sent=...`
    - `CLUSTER_MATMUL_RESULT src_board=... seq=... fixture=... dot=... expected=... ok=...`
    - `CLUSTER_MATMUL_GATHER seq=... worker1=... worker2=... total=... expected=... ok=...`
  - Host verifier added: `python3 tools/verify_cluster_matmul.py --port <coordinator-serial-port>`.
  - 2026-07-03: `pio run -e cluster_coord_ap_matmul` — SUCCESS
  - No-flash status: live Phase 2 hardware proof intentionally not performed during artifact preparation; controller must run coordinator serial verification before marking hardware proof complete.
- [x] Task 2.4 int4 sharded matmul fixture
  - 2026-07-03: Firmware/protocol artifacts prepared, not flashed.
  - Added fixture `2` for packed signed int4 weights while preserving fixture `1` int8 behavior.
  - Fixture `2` keeps request vector `int8[16]`; worker weights are packed two signed int4 values per byte in firmware/protocol helper code.
  - Fixture `2` math is computed in code:
    - `vector[i] = i - 8`
    - worker1 int4 weights: `clamp_i4((i % 8) - 4)`, expected dot `88`
    - worker2 int4 weights: `clamp_i4(3 - (i % 8))`, expected dot `-80`
    - gathered expected total `8`
  - Coordinator alternates fixture `1` and fixture `2` by matmul sequence.
  - Receipt formats now include fixture IDs on result and gather lines:
    - `CLUSTER_MATMUL_RESULT src_board=... seq=... fixture=... dot=... expected=... ok=...`
    - `CLUSTER_MATMUL_GATHER seq=... fixture=... worker1=... worker2=... total=... expected=... ok=...`
  - Host verifier supports `--fixture 1`, `--fixture 2`, and `--fixture both` with `both` as the default.
  - 2026-07-03: `python3 tools/test_cluster_protocol.py` — SUCCESS (`PASS packet encode/decode/crc`)
  - 2026-07-03: `python3 -m py_compile tools/*.py` — SUCCESS
  - 2026-07-03: `python3 tools/verify_cluster_matmul.py --help` — SUCCESS
  - 2026-07-03: `pio run -e cluster_coord_ap_matmul` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker1_ap_matmul` — SUCCESS
  - 2026-07-03: `pio run -e cluster_worker2_ap_matmul` — SUCCESS
  - Build-only boundary: live int4 hardware proof is pending until coordinator and both workers receive this updated firmware.
  - Current live hardware proof remains fixture `1` only.

## Phase 2 live hardware proof

- 2026-07-03 coordinator USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role coord --mode matmul --port /dev/ttyACM0 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:41:f4`
  - coordinator env: `cluster_coord_ap_matmul`
- 2026-07-03 live three-board synthetic sharded matmul proof:
  - verifier command: `python3 tools/verify_cluster_matmul.py --port /dev/ttyACM0 --timeout 90`
  - verifier result: `PASS cluster matmul seq=3 worker1=272 worker2=-408 total=-136`
  - coordinator receipt: `CLUSTER_MATMUL_RESULT src_board=1 seq=6 fixture=1 dot=272 expected=272 ok=true`
  - coordinator receipt: `CLUSTER_MATMUL_RESULT src_board=2 seq=6 fixture=1 dot=-408 expected=-408 ok=true`
  - coordinator gather: `CLUSTER_MATMUL_GATHER seq=6 worker1=272 worker2=-408 total=-136 expected=-136 ok=true`
  - result: coordinator AP sent deterministic matmul requests to both workers, gathered both shard results, and matched expected total.
- Fixture boundary: this live proof is fixture `1` int8 only. Live fixture `2` int4 proof is pending until all boards receive the updated alternating-fixture firmware.
- OTA upload boundary:
  - OTA servers are installed on worker1, worker2, and coordinator firmware after USB flashes.
  - Current laptop network is normal WiFi `192.168.50.181/24`; it cannot route to coordinator AP `192.168.4.x` without joining `RI-ESP-CLUSTER`.
  - Live OTA upload is not attempted from this session because switching the only WiFi interface to the no-internet cluster AP would likely sever agent connectivity.
  - OTA host helper remains build/dry-run verified: `tools/ota_cluster_wifi.py`.

## OTA installation receipts

- 2026-07-03 coordinator USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role coord --mode matmul --port /dev/ttyACM0 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:41:f4`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=0 role=coord mode=matmul`
  - WiFi/AP receipt: `CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=0 hostname=ri-esp-cluster-coord ip=192.168.4.1 port=3232`
- 2026-07-03 worker1 USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker1 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `a4:cb:8f:d9:24:ec`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=1 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=1 ip=192.168.4.2 rssi=-41 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=1 hostname=ri-esp-cluster-worker1 ip=192.168.4.2 port=3232`
  - note: coordinator is still running old ping firmware, so worker1 matmul firmware correctly continues to answer PING packets until coordinator is upgraded.
- 2026-07-03 worker2 USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker2 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:40:b0`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=2 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=2 ip=192.168.4.3 rssi=-46 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=2 hostname=ri-esp-cluster-worker2 ip=192.168.4.3 port=3232`

## Phase 3: Shard the existing H256 LSTM model

- [ ] Task 3.1 model shard exporter
- [ ] Task 3.2 per-board shard flash flow
- [ ] Task 3.3 one-token H256 cluster inference check
- [ ] Task 3.4 full H256 utility suite on cluster

## Phase 4: Transformer micro-model before TinyStories

- [ ] Task 4.x pending

## Phase 5: TinyStories reduced model

- [ ] Task 5.x pending

## Phase 6: TinyStories-33M attempt

- [ ] Task 6.x pending

## Phase 7: Sensor-grounded demo integration

- [ ] Task 7.x pending
