# Coordinator AP + OTA Provisioning Plan (2026-07-03)

## Purpose and claim boundary

This document defines the provisioning model for the 3-board ESP32-S3 cluster in environments with no external infrastructure.

- Transport runtime is WiFi-first in normal operation.
- Provisioning is split into **first-flash USB bootstrap** and **post-first-flash OTA update flow**.
- ArduinoOTA support now exists for WiFi cluster firmware modes after the implementation commit.
- A first USB flash is still required to install OTA-capable firmware before any WiFi update can be pushed.
- Live OTA upload proof is not claimed until a controller flashes OTA-enabled firmware by USB and records a successful WiFi OTA receipt.

## 1) No-infrastructure deployment model

- Primary mode is **coordinator AP/gateway**. Coordinator creates a private SSID and workers join it as stations.
- No internet uplink is required after flashing.
- Coordinator remains a local control plane for:
  - model/firmware manifest distribution (future OTA),
  - in-band status/health collection,
  - worker role coordination.
- USB is used only for bootstrap flashing and deep recovery.
- Hardware requirements:
  - Coordinator + Worker 1 + Worker 2 (ESP32-S3).
  - One reliable serial path for first flash (or per-board one-at-a-time flash).
- Expected AP proving values (already validated in Phase 1.4):
  - Coordinator AP IP: `192.168.4.1`
  - Worker 1 IP: `192.168.4.2`
  - Worker 2 IP: `192.168.4.3`

## 2) First-flash USB provisioning flow

Goal: bring all boards to a known-good baseline role/transport state.

1. **Prepare host secrets**
   - Use default `wifi_secrets.local.h` if local-only mode is intended, or create a local secret override before flashing.
2. **Flash coordinator image via USB**
   - Flash board with `cluster_coord` equivalent environment for AP role.
   - Validate boot logs for AP and UDP-ready state at `192.168.4.1`.
3. **Enroll worker images via USB (one at a time)**
   - Flash worker 1 with its worker image.
   - Flash worker 2 with its worker image.
   - Validate that each station associates and binds expected local IP (`.2`, `.3`).
4. **Record post-flash state**
   - Confirm both workers report worker-ready and UDP-ready.
   - Confirm coordinator can observe worker packets (PING/PONG or equivalent transport receipt).

Expected safety outcome: once first-flash finishes, USB can be unplugged and cluster runs without host tethering in coordinator AP mode.

## 3) Post-first-flash OTA flow (partially implemented)

The immediate implemented path uses ArduinoOTA directly from the host to each board over the coordinator AP network. The broader coordinator-managed, manifest-verified rollout remains planned.

Current host-driven commands:

- `python3 tools/ota_cluster_wifi.py --role coord --mode matmul --ip 192.168.4.1 --execute`
- `python3 tools/ota_cluster_wifi.py --role worker1 --mode matmul --ip 192.168.4.2 --execute`
- `python3 tools/ota_cluster_wifi.py --role worker2 --mode matmul --ip 192.168.4.3 --execute`

Expected firmware OTA receipts:

- `CLUSTER_OTA_READY board_id=... hostname=... ip=... port=3232`
- `CLUSTER_OTA_START type=sketch`
- `CLUSTER_OTA_PROGRESS percent=...`
- `CLUSTER_OTA_END ok=1`
- `CLUSTER_OTA_ERROR code=...`

### 3.1 Flow overview

Implemented now:

1. Host builds the role/mode-specific PlatformIO environment.
2. Host runs `espota.py` against the target board IP on port `3232`.
3. Board accepts ArduinoOTA sketch updates using `CLUSTER_OTA_PASSWORD`.
4. Board emits OTA serial receipts during the update.

Planned coordinator-managed flow:

1. Coordinator downloads/receives OTA bundle metadata (host-provided artifact + manifest).
2. Coordinator verifies:
   - manifest signature/hash,
   - target firmware compatibility (board role + minimum hardware/flash assumptions,
   - monotonic version number.
3. Coordinator pushes update package to workers over local WiFi control plane.
4. Workers stage update into inactive slot and report staging checksum.
5. On successful checksum + manifest match, worker marks staged image as bootable.
6. Worker reboots into staged image and emits update-success receipt.

### 3.2 Worker enrollment flow

- Enrollment is role-based by board identity (pre-provisioned board ID + expected role mapping).
- On first successful network join after provisioning window, worker enters `ENROLLED_AWAITING_POLICY` state.
- Coordinator confirms:
  - board identity,
  - role consistency,
  - shard/variant compatibility,
  - transport heartbeat.
- Worker transitions to `RUN` only after coordinator-signed enrollment completion.

## 4) Coordinator AP/gateway responsibilities

- SSID/passphrase ownership and DHCP handoff.
- Static-address intent/announcement for worker discovery.
- Enrollment ledger (board ID, role, version, last-good firmware hash).
- OTA staging status tracking:
  - pending,
  - staged,
  - validated,
  - active.
- Health policy:
  - heartbeat timeout and stale-node handling.
- Fail-safe policy owner for downgrade/rollback trigger.

## 5) Worker responsibilities

- Station join, SSID/pass validation, and heartbeat publishing.
- Report board ID, free memory, role hash, and current firmware hash at boot and on state change.
- Reject unknown update payload unless it passes manifest precheck.
- Perform checksum + image header checks before activation.
- Persist last-known-good state and rollback trigger flag.

## 6) Safety and fallback policy if OTA fails

- **Two-slot model (planned):** active slot + staged/backup slot.
- **Never overwrite active slot in-place** during OTA.
- If worker fails integrity checks, load fails, or heartbeat does not return within timeout:
  - rollback to previous image automatically.
- If two consecutive failed rollouts occur:
  - block further OTA for that worker,
  - require manual USB rescue on that board,
  - keep cluster operational on remaining healthy nodes where possible.
- Keep an explicit “OTA-disabled / freeze” mode until manual approval to prevent repeated bad updates.

## 7) Security notes

### 7.1 Default passphrase behavior

- `localfirstai` is acceptable only as a development/default bootstrap credential.
- For any deployment-like run:
  - replace with a local custom passphrase,
  - avoid reuse across test domains,
  - keep credentials outside source control (`wifi_secrets.local.h` style override).
- Treat default-passphrase environments as “lab-only” and annotate that they are not production-hardened.

### 7.2 User-configured passphrase behavior

- Use per-site passphrases and rotate before field-use.
- Restrict AP mode visibility and keep channel/power settings conservative for close-range coverage.
- Prefer WPA2-PSK minimum; avoid open APs.
- Log whether custom credentials are in effect for auditability.

## 8) Explicit claim boundaries (Phase 1.5)

- This is a **provisioning design and operational plan** for no-infrastructure deployments, now with a basic ArduinoOTA implementation path for WiFi cluster firmware modes.
- Firmware and host helper implementation exists, but no live OTA upload proof is claimed yet.
- Phase 1.5 evidence remains:
  - coordinator AP proof,
  - worker station joins,
  - worker enrollment behavior by design only,
  - OTA build/dry-run verification only.
- OTA live acceptance remains deferred until OTA-capable firmware is first installed by USB and then updated over WiFi.
