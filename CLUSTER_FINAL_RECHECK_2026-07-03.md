# ESP32-S3 cluster final recheck — 2026-07-03

Status: PASS live hardware recheck after prior one-token distributed generation closure.

Timestamp:

```text
2026-07-03T07:17:51-05:00
```

Scope:

- Re-ran local build/protocol gates from the repository.
- Reopened coordinator serial on `/dev/ttyACM0`.
- Observed both workers connected over coordinator AP WiFi.
- Captured a fresh live distributed TinyStories H512 one-token generation pass.

Build/protocol receipts:

```text
python3 -m py_compile tools/*.py
PASS

python3 tools/test_cluster_protocol.py
PASS packet encode/decode/crc

pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
cluster_coord_ap_lstm_shard    SUCCESS
cluster_worker1_ap_lstm_shard  SUCCESS
cluster_worker2_ap_lstm_shard  SUCCESS
3 succeeded in 00:00:07.691
```

Live hardware receipts:

```text
/dev/ttyACM0 coordinator boot:
ESP32-S3 cluster WiFi demo boot board_id=0 role=coord mode=lstm_shard
CLUSTER_WIFI_AP_READY ok=1 ssid=RI-ESP-CLUSTER ip=192.168.4.1 port=42100
CLUSTER_HTTP_UPDATE_READY board_id=0 ip=192.168.4.1 port=8080 endpoint=/update data_endpoint=/update_weights
CLUSTER_MODEL_READY board_id=0 ok=1 role=coord

Worker discovery during run:
CLUSTER_WIFI_PONG src_board=1 seq=18 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=21 from=192.168.4.3:42100 rssi=0 model_ready=1

Fresh final pass line:
CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t dist_token=19 dist_char=t logit=8.154635 elapsed_ms=5971 status=PASS note=worker_int8_recurrent_vs_local_int4_reference
```

Interpretation:

- Coordinator AP was live at `192.168.4.1`.
- Worker1 and worker2 both answered WiFi pings with `model_ready=1`.
- The full recurrent gate-row distributed path completed again.
- The fresh generated distributed token was `t` and matched the local p22 reference token.
- Fresh elapsed time was `5971 ms`, slightly faster than the prior recorded `6009 ms` pass.

Claim boundary:

Safe claim:

- Live hardware-proven one-token distributed TinyStories H512 recurrent generation step over a two-worker ESP32-S3 cluster, reproduced after final recheck.

Do not claim:

- optimized throughput
- multi-token streaming benchmark
- production-ready distributed inference

Those still require separate repeated-token benchmark and transport/chunk scheduling optimization receipts.
