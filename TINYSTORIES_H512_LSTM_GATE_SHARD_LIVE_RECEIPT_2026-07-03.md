# TinyStories H512 LSTM gate-shard live receipt — 2026-07-03

Status: completed live two-worker recurrent gate-shard proof.

Implemented in firmware:

- `CLUSTER_MSG_LSTM_GATE_REQUEST`
- `CLUSTER_MSG_LSTM_GATE_RESULT`
- coordinator-side gate probe generation from the H512 TinyStories model
- worker-side RIWS shard tensor parsing
- worker-side LSTM gate-row shard dot/bias compute from the RIWS `weights` partition
- coordinator-side expected-value recomputation and max-absolute-error check
- two-worker gate-result gather receipt

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

Worker1 USB app update with gate-result handler:

```text
pio run -e cluster_worker1_ap_lstm_shard -t upload --upload-port /dev/ttyACM1
cluster_worker1_ap_lstm_shard SUCCESS
MAC: a4:cb:8f:d9:24:ec
```

Worker2 app update with gate-result handler:

```text
python3 tools/relay_worker_update.py --role worker2 --mode lstm_shard --port /dev/ttyACM0 --wait-worker --wait-timeout 60 --relay-timeout 360 --execute

CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=779664 target=app
CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=779664 total=779664
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=HTTP/1.1 200 OK
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=OK
CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=109712
```

Two-worker final readiness and recurrent gate gather:

```text
CLUSTER_WIFI_PONG src_board=1 seq=900 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=901 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=901 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=902 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=902 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=1 seq=903 from=192.168.4.3:42100 rssi=0 model_ready=1
CLUSTER_WIFI_PONG src_board=2 seq=903 from=192.168.4.2:42100 rssi=0 model_ready=1
CLUSTER_LSTM_GATE_REQUEST seq=182 layer=0 token=o input_scale=0.017699463 h_scale=1.000000000 encoded=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=182 dst=1 target=192.168.4.3 sent=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=182 dst=2 target=192.168.4.2 sent=true
CLUSTER_LSTM_GATE_RESULT src_board=1 seq=182 layer=0 row_start=0 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=182 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_GATHER seq=182 layer=0 worker1_ok=true worker1_max_abs_err=0 worker2_ok=true worker2_max_abs_err=0 rows_checked=32 status=PASS
SUMMARY seen1=True seen2=True gate1=True gate2=True gather=True
```

Conclusion:

- The recurrent gate-shard compute verifier is implemented and build-verified for coordinator + both workers.
- Worker1 and worker2 both load their H512 RIWS shards from data partitions.
- Worker1 computes rows `0-15`; worker2 computes rows `1024-1039` for the layer-0 gate probe.
- Coordinator recomputes expected values from the full H512 model and verifies both workers with `max_abs_err=0`.
- `CLUSTER_LSTM_GATE_GATHER ... status=PASS` is captured on live hardware.

Claim boundary:

Do claim complete two-worker recurrent gate-row shard compute/gather proof for the tested layer-0 probe. Do not claim full distributed TinyStories generation loop until the same gate-shard primitive is wired into every layer/time-step of generation and benchmarked.
