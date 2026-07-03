# TinyStories H512 LSTM gate-shard live receipt — 2026-07-03

Status: completed live two-worker recurrent gate-shard proof; full distributed generation loop remains a separate unclaimed step.

Implemented in firmware:

- `CLUSTER_MSG_LSTM_GATE_REQUEST`
- `CLUSTER_MSG_LSTM_GATE_RESULT`
- generalized request row range: `row_start`, `count`
- int32 gate result values
- coordinator-side gate probe generation from the H512 TinyStories model
- worker-side RIWS shard tensor parsing
- worker-side LSTM gate-row shard dot/bias compute from the RIWS `weights` partition
- coordinator-side expected-value recomputation and max-absolute-error check
- two-worker gate-result gather receipt
- lower cluster TX power plus brownout reset disable for weak USB worker stability

Build/protocol receipts:

```text
python3 tools/test_cluster_protocol.py
PASS packet encode/decode/crc

pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
cluster_coord_ap_lstm_shard    SUCCESS
cluster_worker1_ap_lstm_shard  SUCCESS
cluster_worker2_ap_lstm_shard  SUCCESS
```

Coordinator flash receipt:

```text
pio run -e cluster_coord_ap_lstm_shard -t upload --upload-port /dev/ttyACM0
cluster_coord_ap_lstm_shard SUCCESS
MAC: 94:a9:90:d2:41:f4
```

Worker1 USB app update with generalized gate-result handler:

```text
pio run -e cluster_worker1_ap_lstm_shard -t upload --upload-port /dev/ttyACM1
cluster_worker1_ap_lstm_shard SUCCESS
MAC: a4:cb:8f:d9:24:ec
```

Worker2 OTA app update with generalized gate-result handler:

```text
python3 tools/relay_worker_update.py --role worker2 --mode lstm_shard --port /dev/ttyACM0 --wait-worker --wait-timeout 90 --relay-timeout 360 --execute

CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=779312 target=app
CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=779312 total=779312
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=HTTP/1.1 200 OK
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=OK
CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=109787
```

Worker1 stability fix receipt:

```text
Before fix: repeated rst:0xf (BROWNOUT_RST) before WiFi join.
After fix: no brownout seen; worker replied to coordinator ping/gate traffic.
CLUSTER_WIFI_WORKER_STATUS board_id=1 ip=192.168.4.3 rssi=-54 port=42100
CLUSTER_WIFI_PING board=1 seq=201 from_board=0 from=192.168.4.1:42100 model_ready=1 reply=sent rssi=-55
CLUSTER_LSTM_GATE_WORKER board=1 seq=42 layer=0 row_start=0 count=16 computed=true reply=sent rssi=-54
```

Final generalized two-worker recurrent gate gather:

```text
CLUSTER_WIFI_PONG src_board=1 seq=233 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=233 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=234 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=234 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=235 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=235 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=236 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=236 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_LSTM_GATE_REQUEST seq=48 layer=0 token=o input_scale=0.017699463 h_scale=1.000000000 prepared=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=48 dst=1 target=192.168.4.3 sent=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=48 dst=2 target=192.168.4.2 sent=true
CLUSTER_LSTM_GATE_RESULT src_board=1 seq=48 layer=0 row_start=0 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=48 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_GATHER seq=48 layer=0 worker1_ok=true worker1_max_abs_err=0 worker2_ok=true worker2_max_abs_err=0 rows_checked=32 status=PASS
SUMMARY seen1=True seen2=True gate1=True gate2=True gather=True
```

Conclusion:

- Worker1 and worker2 both load their H512 RIWS shards from data partitions.
- Worker1 computes rows `0-15`; worker2 computes rows `1024-1039` for the layer-0 gate probe.
- Coordinator recomputes expected values from the full H512 model and verifies both workers with `max_abs_err=0`.
- `CLUSTER_LSTM_GATE_GATHER ... status=PASS` is captured on live hardware after generalized row-range protocol deployment.

Claim boundary:

Do claim complete two-worker recurrent gate-row shard compute/gather proof for the tested layer-0 row-range probe. Do not claim full distributed TinyStories generation loop until the same gate-shard primitive is wired into every layer/time-step of generation and benchmarked.
