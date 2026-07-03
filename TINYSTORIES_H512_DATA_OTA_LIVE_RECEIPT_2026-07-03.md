# TinyStories H512 data OTA live receipt — 2026-07-03

Status: partial live hardware proof.

Completed live:

- Coordinator on `/dev/ttyACM0` was USB-flashed to `cluster_coord_ap_lstm_shard`.
- Coordinator booted AP mode and loaded the TinyStories H512 full weights from its `weights` data partition.
- Worker2 was reachable on the coordinator AP at `192.168.4.2`.
- Worker2 app firmware was relayed over coordinator USB -> worker HTTP `/update`.
- Worker2 RIWS gate-row shard was relayed over coordinator USB -> worker HTTP `/update_weights`.
- Worker2 rebooted and reported `model_ready=1` through coordinator-visible PONG payload.

Coordinator boot receipt:

```text
ESP32-S3 cluster WiFi demo boot board_id=0 role=coord mode=lstm_shard
CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100
CLUSTER_HTTP_UPDATE_READY board_id=0 ip=192.168.4.1 port=8080 endpoint=/update data_endpoint=/update_weights
weights partition addr=0x210000 size=5242880
RILM version=1 tensors=15
payloads cloned bytes=4784772 free_heap=206864 free_psram=3635859
int4 recurrent converted=3 saved=1572864 free_psram=2062947
CLUSTER_MODEL_READY board_id=0 ok=1 role=coord
```

Worker2 app OTA receipt:

```text
python3 tools/relay_worker_update.py --role worker2 --mode lstm_shard --port /dev/ttyACM0 --wait-worker --wait-timeout 90 --relay-timeout 360 --execute

CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=775792 target=app
CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=775792 total=775792
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=HTTP/1.1 200 OK
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=OK
CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=109343
```

Worker2 data OTA receipt:

```text
python3 tools/relay_worker_update.py --role worker2 --mode lstm_shard --target weights --artifact shards/tinystories-h512-lstm/worker2_lstm_gate_shard.riws --port /dev/ttyACM0 --wait-worker --wait-timeout 120 --relay-timeout 540 --execute

CLUSTER_WIFI_PONG src_board=2 seq=7 from=192.168.4.2:42100 rssi=0 model_ready=0
CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=2384308 target=weights
CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=2384308 total=2384308
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=HTTP/1.1 200 OK
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=OK
CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=294556
```

Worker2 post-data reboot readiness receipt:

```text
CLUSTER_WIFI_PONG src_board=2 seq=17 from=192.168.4.2:42100 rssi=0 model_ready=1
```

Worker2 shard artifact:

```text
path=shards/tinystories-h512-lstm/worker2_lstm_gate_shard.riws
bytes=2384308
sha256=2af693549024b717110f6ca1a2369d6644228da4bdd9decfdc9439872a00682f
rows=1024-2047
tensors=12
```

Worker1 blocker:

Worker1 did not appear on the coordinator AP during repeated 90-second discovery windows. Coordinator only observed worker2 PONGs.

Fallback direct relay attempt to worker1's last-known SoftAP DHCP address also failed:

```text
python3 tools/relay_worker_update.py --role worker1 --mode lstm_shard --port /dev/ttyACM0 --relay-timeout 180 --execute

CLUSTER_RELAY_UPDATE_WARN board=1 reason=worker_ip_unknown using_fallback_ip=192.168.4.3
CLUSTER_RELAY_UPDATE_START board=1 ip=192.168.4.3 port=8080 bytes=775776 target=app
CLUSTER_RELAY_UPDATE_ERROR phase=connect board=1 ip=192.168.4.3
```

Conclusion:

- Data/model OTA is live-proven on worker2.
- Coordinator relay hardening is live-proven for app and data OTA on worker2.
- Worker1 remains physically/network unreachable from the coordinator AP in the current hardware state, so worker1 cannot be completed over the air until it is powered/connected or attached by USB.

Claim boundary:

Do claim worker2 H512 RIWS data OTA proof. Do not claim both workers have live data OTA proof yet.
