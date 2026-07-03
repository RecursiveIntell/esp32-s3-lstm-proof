# TinyStories H512 LSTM gate-shard live receipt — 2026-07-03

Status: partial live hardware proof; worker2 recurrent gate-row compute verified, worker1 unreachable in current hardware state.

Implemented in firmware:

- `CLUSTER_MSG_LSTM_GATE_REQUEST`
- `CLUSTER_MSG_LSTM_GATE_RESULT`
- coordinator-side gate probe generation from the H512 TinyStories model
- worker-side RIWS shard tensor parsing
- worker-side LSTM gate-row shard dot/bias compute from the RIWS `weights` partition
- coordinator-side expected-value recomputation and max-absolute-error check

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

Worker2 app update with gate-result handler:

```text
python3 tools/relay_worker_update.py --role worker2 --mode lstm_shard --port /dev/ttyACM0 --wait-worker --wait-timeout 60 --relay-timeout 360 --execute

CLUSTER_RELAY_UPDATE_START board=2 ip=192.168.4.2 port=8080 bytes=779664 target=app
CLUSTER_RELAY_UPDATE_PROGRESS board=2 sent=779664 total=779664
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=HTTP/1.1 200 OK
CLUSTER_RELAY_UPDATE_RESPONSE board=2 line=OK
CLUSTER_RELAY_UPDATE_END board=2 ok=1 status="HTTP/1.1 200 OK" elapsed_ms=109712
```

Worker2 recurrent gate-shard compute receipts:

```text
CLUSTER_LSTM_GATE_REQUEST seq=21 layer=0 token=o input_scale=0.017699463 h_scale=1.000000000 encoded=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=21 dst=1 target=192.168.4.255 sent=true
CLUSTER_LSTM_GATE_REQUEST_SEND seq=21 dst=2 target=192.168.4.2 sent=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=21 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true

CLUSTER_LSTM_GATE_RESULT src_board=2 seq=22 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=23 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=24 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=25 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=26 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=27 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=28 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=29 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=30 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=31 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=32 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=33 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=34 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=35 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=36 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=37 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
CLUSTER_LSTM_GATE_RESULT src_board=2 seq=38 layer=0 row_start=1024 count=16 max_abs_err=0 ok=true
```

Worker1 blocker in this pass:

- Before coordinator reflashing, both workers had reported `model_ready=1` after data OTA.
- After coordinator was flashed to the new gate-shard verifier firmware, worker2 rejoined and responded.
- Worker1 did not appear in a 180-second AP watch.
- Direct fallback app relay to worker1 at `192.168.4.3:8080` failed with connect error.
- No worker1 USB serial device was present; only `/dev/ttyACM0` coordinator was visible.

Worker1 failed relay receipt:

```text
python3 tools/relay_worker_update.py --role worker1 --mode lstm_shard --port /dev/ttyACM0 --relay-timeout 60 --execute

CLUSTER_RELAY_UPDATE_WARN board=1 reason=worker_ip_unknown using_fallback_ip=192.168.4.3
CLUSTER_RELAY_UPDATE_START board=1 ip=192.168.4.3 port=8080 bytes=779680 target=app
CLUSTER_RELAY_UPDATE_ERROR phase=connect board=1 ip=192.168.4.3
```

Long watch summary:

```text
SUMMARY seen1=False seen2=True gate2=True gather=False
```

Conclusion:

- The recurrent gate-shard compute verifier is implemented and build-verified for coordinator + both workers.
- Worker2 proves live RIWS shard compute against coordinator expected values with `max_abs_err=0`.
- Full two-worker `CLUSTER_LSTM_GATE_GATHER ... status=PASS` is not yet captured because worker1 is currently not reachable on WiFi or USB.

Claim boundary:

Do claim worker2 live recurrent gate-shard compute proof. Do not claim complete two-worker recurrent gate-shard gather until worker1 is reachable and emits a matching `CLUSTER_LSTM_GATE_RESULT`.
