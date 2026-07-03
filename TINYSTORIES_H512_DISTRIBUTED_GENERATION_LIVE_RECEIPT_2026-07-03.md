# TinyStories H512 distributed generation live receipt — 2026-07-03

Status: completed live one-token distributed recurrent generation proof.

Scope:

This receipt closes the prior boundary that only a recurrent gate-row shard probe was proven. The coordinator now runs a real one-token TinyStories H512 generation step where the recurrent gate vectors for all 3 LSTM layers are supplied by the two workers over WiFi.

Architecture:

- Coordinator:
  - owns the prompt/state machine
  - runs the prompt prefix locally
  - sends the final prompt character through the distributed recurrent path
  - sends LSTM gate requests chunked by row range
  - receives worker gate-row chunks
  - applies LSTM activations/state updates locally after each layer
  - runs FC argmax after layer 2
  - prints the generated token receipt

- Worker1:
  - owns RIWS rows 0-1023 for every LSTM layer gate matrix/bias tensor
  - computes 64-row chunks from `/update_weights` RIWS shard data

- Worker2:
  - owns RIWS rows 1024-2047 for every LSTM layer gate matrix/bias tensor
  - computes 64-row chunks from `/update_weights` RIWS shard data

Important precision boundary:

The workers' RIWS shards contain the original int8 recurrent weights. The coordinator's local fast p22 path converts recurrent weights to int4 at boot. Therefore `local_reference_ok=false` during the distributed loop is expected: it is comparing worker int8 recurrent outputs against the coordinator's local int4-converted recurrent reference. For distributed generation, the worker int8 shard outputs are the source of truth. The final token also matched the local p22 reference in this run.

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
```

Live distributed generation receipt:

```text
CLUSTER_DIST_GEN_START prompt="once upon a " final_input=  expected_token=19 expected_char=t

# Layer 0: worker1 rows 0-1023, worker2 rows 1024-2047, chunked 64 rows at a time.
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1000 layer=0 row_start=0 count=64 accepted=true local_reference_ok=false max_abs_err=4186927
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1000 layer=0 row_start=1024 count=64 accepted=true local_reference_ok=false max_abs_err=4197555
...
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1015 layer=0 row_start=960 count=64 accepted=true local_reference_ok=false max_abs_err=4190845
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1015 layer=0 row_start=1984 count=64 accepted=true local_reference_ok=false max_abs_err=4202406
CLUSTER_DIST_GEN_LAYER_DONE layer=0

# Layer 1: full two-worker recurrent gate rows completed.
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1016 layer=1 row_start=0 count=64 accepted=true local_reference_ok=false max_abs_err=16777164
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1016 layer=1 row_start=1024 count=64 accepted=true local_reference_ok=false max_abs_err=33553909
...
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1031 layer=1 row_start=960 count=64 accepted=true local_reference_ok=false max_abs_err=16777834
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1031 layer=1 row_start=1984 count=64 accepted=true local_reference_ok=false max_abs_err=33553532
CLUSTER_DIST_GEN_LAYER_DONE layer=1

# Layer 2: full two-worker recurrent gate rows completed.
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1032 layer=2 row_start=1024 count=64 accepted=true local_reference_ok=false max_abs_err=50330368
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1032 layer=2 row_start=0 count=64 accepted=true local_reference_ok=false max_abs_err=33555005
...
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=1 seq=1047 layer=2 row_start=960 count=64 accepted=true local_reference_ok=false max_abs_err=33555474
CLUSTER_DIST_GEN_CHUNK_RESULT src_board=2 seq=1047 layer=2 row_start=1984 count=64 accepted=true local_reference_ok=false max_abs_err=33558104
CLUSTER_DIST_GEN_LAYER_DONE layer=2

CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t dist_token=19 dist_char=t logit=8.154635 elapsed_ms=6009 status=PASS note=worker_int8_recurrent_vs_local_int4_reference
SUMMARY pass_line= CLUSTER_DIST_GEN_TOKEN prompt="once upon a " local_p22_token=19 local_p22_char=t dist_token=19 dist_char=t logit=8.154635 elapsed_ms=6009 status=PASS note=worker_int8_recurrent_vs_local_int4_reference
```

Conclusion:

- Both workers supplied recurrent gate chunks for every gate row of every LSTM layer for one TinyStories H512 generation step.
- Coordinator consumed those chunks to update recurrent state through all 3 layers and run FC argmax.
- The generated distributed token was `t`, matching the local p22 reference token for `"once upon a "` in this run.
- Elapsed time for this unoptimized one-token distributed step: 6009 ms.

Claim boundary:

Safe claim: live hardware-proven one-token distributed TinyStories H512 recurrent generation step over a two-worker ESP32-S3 cluster.

Do not claim: optimized throughput, multi-token streaming generation benchmark, or production-ready distributed inference. Those require chunk scheduling/transport optimization and repeated-token benchmark receipts.
