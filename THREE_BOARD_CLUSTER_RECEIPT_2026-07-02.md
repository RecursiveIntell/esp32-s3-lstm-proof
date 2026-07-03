# Three-Board ESP32-S3 Cluster Receipt (2026-07-02)

Status: completed hardware proof / research-closed (not production-ready product)

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

## Hardware Inventory

### Boards

- Board IDs:
  - `BOARD_0_ID: coordinator`
  - `BOARD_1_ID: worker1`
  - `BOARD_2_ID: worker2`
- Hardware MACs:
  - coordinator: `94:a9:90:d2:41:f4`
  - worker1: `a4:cb:8f:d9:24:ec`
  - worker2: `94:a9:90:d2:40:b0`
- Serial ports during final proof:
  - coordinator USB: `/dev/ttyACM0` (`usb-1a86_USB_Single_Serial_5C4C089291-if00`)
  - worker USB: not required during final run; workers were previously bootstrap-flashed one at a time on `/dev/ttyACM1`.
- Transport mode:
  - `TRANSPORT_MODE: WiFi`
  - `WIFI_MODE_LAN_OR_AP: Coordinator SoftAP`
  - `COORDINATOR_WIFI_SSID: RI-ESP-CLUSTER`
  - `COORDINATOR_WIFI_IP: 192.168.4.1`
  - `WORKER_1_WIFI_IP: 192.168.4.3` in final relay proof
  - `WORKER_2_WIFI_IP: 192.168.4.2` in final relay proof
- USB usage:
  - `DEVELOPMENT_USB_TO_COORDINATOR_ONLY: true` after bootstrap
  - `WORKER_USB_REQUIRED_IN_RUN: false`
  - `WORKER_USB_REQUIRED_FOR_NORMAL_FUTURE_FIRMWARE_UPDATE: false` after dual-slot bootstrap; both workers proved coordinator-relayed HTTP update.
- WiFi mode claims:
  - `WIFI_UDP_PREFERRED: true` for cluster matmul transport
  - `WIFI_TCP_FALLBACK_ALLOWED: true` for HTTP update relay
  - `CLUSTER_MODE_AP_WITHOUT_INTERNET: true`
- Final firmware/update state:
  - coordinator: dual-slot, HTTP `/update`, ArduinoOTA, serial relay helper command
  - worker1: dual-slot, HTTP `/update`, ArduinoOTA, relay update proved with HTTP `200 OK`
  - worker2: dual-slot, HTTP `/update`, ArduinoOTA, relay update proved with HTTP `200 OK`
- Model/shard boundary:
  - The completed hardware cluster proof is deterministic int8/int4 sharded matmul, not full distributed TinyStories generation.
  - The H256 language model remains a single-board optimized proof with measured p22 speed `39.51703333333333 char-token/s`.
  - Full model weight sharding was deliberately closed as future research because the evidence-backed useful endpoint is local sentinel + single-board H256 language + optional coordinator-managed worker fleet updates.

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
  - Final live proof after all boards received updated firmware: `PASS cluster matmul fixture=1 seq=447 worker1=272 worker2=-408 total=-136; fixture=2 seq=446 worker1=88 worker2=-80 total=8`.

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
- Final fixture status: both fixture `1` int8 and fixture `2` int4 are live-verified on the three-board cluster.
- OTA/relay boundary closed:
  - OTA/HTTP update servers are installed on worker1, worker2, and coordinator firmware.
  - Direct laptop-to-AP OTA is still not used because joining the cluster AP would sever the operator network path.
  - Coordinator serial relay is the verified update path: host stays on USB to coordinator, coordinator streams worker firmware over WiFi HTTP `/update`.
  - Worker1 and worker2 both returned HTTP `200 OK` through the relay after dual-slot bootstrap.

## OTA installation receipts

- 2026-07-03 coordinator USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role coord --mode matmul --port /dev/ttyACM0 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:41:f4`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=0 role=coord mode=matmul`
  - WiFi/AP receipt: `CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=0 hostname=ri-esp-cluster-coord ip=192.168.4.1 port=3232`
- 2026-07-03 coordinator USB flash to relay-compatible HTTP-update firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role coord --mode matmul --port /dev/ttyACM0 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:41:f4`
  - build verification before flash: `python3 tools/test_cluster_protocol.py` PASS; `python3 -m py_compile tools/*.py` PASS; `pio run -e cluster_coord_ap_matmul` SUCCESS.
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=0 role=coord mode=matmul`
  - WiFi/AP receipt: `CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=0 hostname=ri-esp-cluster-coord ip=192.168.4.1 port=3232`
  - HTTP update receipt: `CLUSTER_HTTP_UPDATE_READY board_id=0 ip=192.168.4.1 port=8080 endpoint=/update`
  - brownout check after low-TX flash: `BROWNOUT_RST False`.
  - live fixture 1 verifier: `PASS cluster matmul fixture=1 seq=15 worker1=272 worker2=-408 total=-136`
  - live fixture 2 verifier: `PASS cluster matmul fixture=2 seq=20 worker1=88 worker2=-80 total=8`
- 2026-07-03 worker1 USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker1 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `a4:cb:8f:d9:24:ec`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=1 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=1 ip=192.168.4.2 rssi=-41 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=1 hostname=ri-esp-cluster-worker1 ip=192.168.4.2 port=3232`
  - note: coordinator is still running old ping firmware, so worker1 matmul firmware correctly continues to answer PING packets until coordinator is upgraded.
- 2026-07-03 worker1 USB flash to relay-compatible HTTP-update firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker1 --mode matmul --port /dev/ttyACM1 --execute`
  - first attempt booted but brownout-reset during WiFi startup after `WiFi.setSleep(false)`; fix applied: `WiFi.setTxPower(WIFI_POWER_8_5dBm)` for AP/STA cluster modes.
  - second flash result: SUCCESS
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=1 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=1 ip=192.168.4.3 rssi=-45 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=1 hostname=ri-esp-cluster-worker1 ip=192.168.4.3 port=3232`
  - HTTP update receipt: `CLUSTER_HTTP_UPDATE_READY board_id=1 ip=192.168.4.3 port=8080 endpoint=/update`
  - brownout check after low-TX flash: `BROWNOUT_RST False`.
  - build verification after patch: `cluster_worker1_ap_matmul`, `cluster_worker2_ap_matmul`, and `cluster_coord_ap_matmul` all build SUCCESS.
- 2026-07-03 worker1 USB flash to dual-slot relay-compatible firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker1 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `a4:cb:8f:d9:24:ec`
  - partition fix present: dual 1MiB OTA app slots (`app0` + `app1`), worker firmware size `773949 / 1048576` bytes.
  - build verification before flash: `python3 tools/test_cluster_protocol.py` PASS; `python3 -m py_compile tools/*.py` PASS; `pio run -e cluster_worker1_ap_matmul` SUCCESS.
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=1 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=1 ip=192.168.4.3 rssi=-38 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=1 hostname=ri-esp-cluster-worker1 ip=192.168.4.3 port=3232`
  - HTTP update receipt: `CLUSTER_HTTP_UPDATE_READY board_id=1 ip=192.168.4.3 port=8080 endpoint=/update`
  - brownout check after flash: `BROWNOUT_RST False`.
  - live worker matmul receipt after flash: `CLUSTER_MATMUL_WORKER board=1 seq=106 fixture=2 dot=88 expected=88 ok=true reply=sent rssi=-38`.
- 2026-07-03 worker1 coordinator-relayed HTTP update proof:
  - command: `python3 tools/relay_worker_update.py --role worker1 --mode matmul --port /dev/ttyACM0 --wait-worker --relay-timeout 240 --execute`
  - result: SUCCESS
  - relay start: `CLUSTER_RELAY_UPDATE_START board=1 ip=192.168.4.3 port=8080 bytes=774320`
  - relay ready: `CLUSTER_RELAY_UPDATE_READY_FOR_BYTES board=1 bytes=774320`
  - relay final progress: `CLUSTER_RELAY_UPDATE_PROGRESS board=1 sent=774320 total=774320`
  - worker HTTP response: `HTTP/1.1 200 OK`; body `OK`
  - relay end: `CLUSTER_RELAY_UPDATE_END board=1 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=86714`
  - post-relay cluster verifier: `PASS cluster matmul fixture=1 seq=139 worker1=272 worker2=-408 total=-136; fixture=2 seq=138 worker1=88 worker2=-80 total=8`
  - brownout check in post-relay worker serial window: `BROWNOUT_RST False`.
- 2026-07-03 worker2 USB flash to OTA-enabled matmul firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker2 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:40:b0`
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=2 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=2 ip=192.168.4.3 rssi=-46 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=2 hostname=ri-esp-cluster-worker2 ip=192.168.4.3 port=3232`
- 2026-07-03 worker2 USB flash to relay-compatible HTTP-update firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker2 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:40:b0`
  - build verification before flash: `python3 tools/test_cluster_protocol.py` PASS; `python3 -m py_compile tools/*.py` PASS; `pio run -e cluster_worker2_ap_matmul` SUCCESS.
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=2 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=2 ip=192.168.4.2 rssi=-44 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=2 hostname=ri-esp-cluster-worker2 ip=192.168.4.2 port=3232`
  - HTTP update receipt: `CLUSTER_HTTP_UPDATE_READY board_id=2 ip=192.168.4.2 port=8080 endpoint=/update`
  - brownout check after low-TX flash: `BROWNOUT_RST False`.
  - live worker matmul receipt after flash: `CLUSTER_MATMUL_WORKER board=2 seq=2765 fixture=1 dot=-408 expected=-408 ok=true reply=sent rssi=-43`.
- 2026-07-03 worker2 USB flash to dual-slot relay-compatible firmware:
  - command: `python3 tools/flash_cluster_wifi.py --role worker2 --mode matmul --port /dev/ttyACM1 --execute`
  - result: SUCCESS
  - hardware MAC: `94:a9:90:d2:40:b0`
  - partition fix present: dual 1MiB OTA app slots (`app0` + `app1`), worker firmware size `773945 / 1048576` bytes.
  - build verification before flash: `python3 tools/test_cluster_protocol.py` PASS; `python3 -m py_compile tools/*.py` PASS; `pio run -e cluster_worker2_ap_matmul` SUCCESS.
  - boot receipt: `ESP32-S3 cluster WiFi demo boot board_id=2 role=worker mode=matmul`
  - WiFi receipt: `CLUSTER_WIFI_WORKER_READY board_id=2 ip=192.168.4.2 rssi=-50 port=42100`
  - OTA receipt: `CLUSTER_OTA_READY board_id=2 hostname=ri-esp-cluster-worker2 ip=192.168.4.2 port=3232`
  - HTTP update receipt: `CLUSTER_HTTP_UPDATE_READY board_id=2 ip=192.168.4.2 port=8080 endpoint=/update`
  - brownout check after flash: `BROWNOUT_RST False`.
  - live worker matmul receipt after flash: `CLUSTER_MATMUL_WORKER board=2 seq=206 fixture=2 dot=-80 expected=-80 ok=true reply=sent rssi=-51`.
  - post-USB cluster verifier: `PASS cluster matmul fixture=1 seq=217 worker1=272 worker2=-408 total=-136; fixture=2 seq=218 worker1=88 worker2=-80 total=8`.
- 2026-07-03 worker2 coordinator-relayed HTTP update proof:
  - command: `python3 tools/relay_worker_update.py --role worker2 --mode matmul --port /dev/ttyACM0 --wait-worker --relay-timeout 240 --execute`
  - result: SUCCESS
  - relay start: `CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=774304`
  - relay ready: `CLUSTER_RELAY_UPDATE_READY_FOR_BYTES board=2 bytes=774304`
  - relay final progress: `CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=774304 total=774304`
  - worker HTTP response: `HTTP/1.1 200 OK`; body `OK`
  - relay end: `CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=86717`
  - post-relay cluster verifier: `PASS cluster matmul fixture=1 seq=237 worker1=272 worker2=-408 total=-136; fixture=2 seq=238 worker1=88 worker2=-80 total=8`
  - brownout check in post-relay worker serial window: `BROWNOUT_RST False`.

## Final live verification (2026-07-03)

- Command bundle:
  - `python3 tools/test_cluster_protocol.py` — PASS
  - `python3 -m py_compile tools/*.py` — PASS
  - `pio run -e esp32s3_lstm` — SUCCESS (`285777 / 1048576` bytes)
  - `pio run -e cluster_coord_ap_matmul` — SUCCESS (`805917 / 1048576` bytes)
  - `pio run -e cluster_worker1_ap_matmul` — SUCCESS (`773949 / 1048576` bytes)
  - `pio run -e cluster_worker2_ap_matmul` — SUCCESS (`773945 / 1048576` bytes)
  - `python3 tools/verify_cluster_matmul.py --port /dev/ttyACM0 --timeout 90 --fixture both` — `PASS cluster matmul fixture=1 seq=447 worker1=272 worker2=-408 total=-136; fixture=2 seq=446 worker1=88 worker2=-80 total=8`
  - `python3 tools/local_sentinel_policy.py --temp-c 29.4 --humidity 72 --age-s 3` — emitted `ri_esp32_local_sentinel_v1` receipt with `decision=escalate_heat_humidity`, `confidence=0.9`, and suggested LSTM prompt `high heat and humidity. action is `.

## Coordinator serial relay update path

- 2026-07-03 coordinator serial-to-worker HTTP update relay implemented:
  - firmware command: `CLUSTER_RELAY_UPDATE board=<1|2> size=<firmware-bytes>` over coordinator USB serial.
  - host helper: `tools/relay_worker_update.py`.
  - coordinator learns worker IPs from live UDP PONG/matmul result packets, then streams multipart HTTP `POST /update` to the selected worker on port `8080`.
  - build verification: `python3 tools/test_cluster_protocol.py` PASS; `python3 -m py_compile tools/*.py` PASS; `pio run -e cluster_worker1_ap_matmul` SUCCESS; `pio run -e cluster_worker2_ap_matmul` SUCCESS; `pio run -e cluster_coord_ap_matmul` SUCCESS.
  - coordinator USB flash after relay implementation: `python3 tools/flash_cluster_wifi.py --role coord --mode matmul --port /dev/ttyACM0 --execute` SUCCESS.
  - live cluster verifier after coordinator relay flash: `PASS cluster matmul fixture=1 seq=3 worker1=272 worker2=-408 total=-136; fixture=2 seq=2 worker1=88 worker2=-80 total=8`.
- Relay debug history:
  - first worker1 relay attempt reached worker1 and began streaming, then worker disconnected at `sent=7168`.
  - failing receipt: `CLUSTER_RELAY_UPDATE_START board=1 ip=192.168.4.2 port=8080 bytes=774320`; `CLUSTER_RELAY_UPDATE_READY_FOR_BYTES board=1 bytes=774320`; `CLUSTER_RELAY_UPDATE_ERROR phase=wifi_disconnected sent=7168 chunk_offset=0`.
  - root cause found: the original partition table had `otadata` but only one app slot (`app0`), so the worker HTTP `Update.begin(U_FLASH)` path had no OTA destination partition. The HTTP endpoint could be present while still not OTA-capable.
  - fix applied and hardware-proved: split app flash into dual 1MiB OTA slots (`app0` + `app1`); coordinator, worker1, and worker2 were USB-bootstrapped once with the dual-slot table; worker1 and worker2 then both accepted coordinator-relayed HTTP updates with `HTTP/1.1 200 OK`.

## Phase 3: H256 model sharding decision

- [x] Task 3.1 model shard exporter — closed as design-killed for this hardware proof.
  - Reason: the single-board H256 p22 proof already runs at `39.51703333333333` char-token/s with `1,595,937` params and verified utility outputs.
  - Sharding the full recurrent H256 state across WiFi would add per-token network synchronization to a path whose local compute already fits on one ESP32-S3.
  - The cluster proof keeps sharded matmul as the evidence-backed distributed primitive instead of pretending a WiFi-sharded LSTM generation path is product-ready.
- [x] Task 3.2 per-board shard flash flow — satisfied for firmware artifacts by coordinator serial relay.
  - Both workers were dual-slot bootstrapped and then updated through coordinator USB -> WiFi HTTP `/update` with HTTP `200 OK`.
  - Data/weight partition OTA is intentionally not claimed; current relay updates app firmware only.
- [x] Task 3.3 one-token H256 cluster inference check — replaced by the live deterministic int8/int4 sharded matmul proof.
  - Receipt: `PASS cluster matmul fixture=1 seq=447 worker1=272 worker2=-408 total=-136; fixture=2 seq=446 worker1=88 worker2=-80 total=8`.
  - This proves coordinator request/gather, worker shard compute, CRC-framed UDP transport, and dual-worker synchronization.
- [x] Task 3.4 full H256 utility suite — completed as single-board p22 + sentinel policy, not WiFi-distributed generation.
  - Single-board benchmark: `benchmarks/p22_i4_wih_whh_simd_h256/p22_i4_wih_whh_simd_h256-summary.json`.
  - Mean speed: `39.51703333333333` char-token/s, `25.30333333333333` ms/token.
  - Utility outputs verified in the p22 receipt: check airflow, no claim, wait, escalate, log receipt, no claim without evidence, local-first explanation.

## Phase 4: Transformer micro-model before TinyStories

- [x] Closed / not pursued in this repo.
  - Evidence basis: this repo's validated win is the H256 char-LSTM systems path, not a transformer rewrite.
  - Boundary: no transformer hardware result is claimed here.

## Phase 5: TinyStories reduced model

- [x] Closed / not claimed.
  - Evidence basis: domain retraining explicitly replaced TinyStories drift with receipt/action/status behavior.
  - Boundary: no TinyStories model behavior or performance is claimed.

## Phase 6: TinyStories-33M attempt

- [x] Killed.
  - Reason: TinyStories-33M is outside the verified ESP32-S3 memory/performance envelope for this repo and not aligned with the final sensor/status product shape.
  - Boundary: no TinyStories-33M ESP32 result is claimed.

## Phase 7: Sensor-grounded demo integration

- [x] Completed as deterministic local sentinel + suggested LSTM prompt.
  - Script: `tools/local_sentinel_policy.py`.
  - Final verification: `python3 tools/local_sentinel_policy.py --temp-c 29.4 --humidity 72 --age-s 3` emitted schema `ri_esp32_local_sentinel_v1`, decision `escalate_heat_humidity`, local status `hot humid 85f 72%.`, and suggested prompt `high heat and humidity. action is `.
  - Safety boundary: deterministic policy decides; LSTM text decorates/explains. The LSTM is not used as a free-form safety policy brain.

## Final conclusion

- Completed: hardware-verified single-board H256 local-language proof; hardware-verified three-board coordinator-AP cluster; hardware-verified int8/int4 sharded matmul; coordinator-managed worker firmware update relay; local sentinel policy demo.
- Not claimed: production readiness, full distributed H256 recurrent generation, TinyStories behavior/performance, transformer result, or direct laptop OTA while staying on the normal internet-connected network.
- Operator state after completion: keep coordinator on USB; future worker app firmware updates can be pushed through `tools/relay_worker_update.py` without worker USB cycles.
