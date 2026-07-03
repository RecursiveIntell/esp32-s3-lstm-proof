# H256 LSTM Cluster Phase 3 Design Review (Fable, 2026-07-03)

## Recommendation

Do the clustered H256 proof, but treat it as mechanism scaffolding, not a speedup target.

At the current single-board p22 baseline (~28.83 ms/token), H256 already fits and runs well on one ESP32-S3. A three-board LSTM proof will likely be equal or slower because it adds per-layer WiFi latency. Its ROI is the distributed inference machinery needed later for a model that does not fit or is bandwidth-bound on one board.

Do not optimize H256 cluster throughput. Get byte-exact parity once, record timing honestly, then move on.

Skip an output-projection-only shard proof. The existing synthetic matmul proof already proves remote dot + gather. The first real Phase 3 proof should be one LSTM layer / one token / real weights, then the full utility suite.

## Proposed architecture

Use hidden-unit row sharding of the LSTM gate matrices.

- worker1 owns hidden units `[0, 128)` for all 3 layers.
- worker2 owns hidden units `[128, 256)` for all 3 layers.
- coordinator owns orchestration, embedding, final FC/argmax, full float hidden state assembly, utility receipts, and retries.
- workers own their local cell-state slice `c[layer][128]`; do not transmit it each step.

Per layer/token:

1. Coordinator computes input vector `x` for this layer.
2. Coordinator quantizes full `x[256]` and full previous `h[layer][256]` exactly once.
3. Coordinator broadcasts one `LSTM_STEP_REQUEST` containing:
   - layer
   - step sequence
   - `x_scale`
   - `h_scale`
   - `qx[256]`
   - `qh[256]`
4. Each worker computes its own 128-unit gate slice from real weights and local `c` slice.
5. Worker replies with `h_slice[128]` as float32.
6. Coordinator assembles full `h[layer][256]`, continues to next layer.
7. After layer 2, coordinator runs FC + argmax locally.

Workers should return float h slices, not quantized h slices, because per-slice quantization would break parity.

## Required protocol additions

Add message types in `src/cluster_protocol.h`:

- `CLUSTER_MSG_LSTM_HELLO` worker -> coordinator
- `CLUSTER_MSG_LSTM_RESET` coordinator -> broadcast
- `CLUSTER_MSG_LSTM_RESET_ACK` worker -> coordinator
- `CLUSTER_MSG_LSTM_STEP_REQUEST` coordinator -> broadcast
- `CLUSTER_MSG_LSTM_STEP_RESULT` worker -> coordinator
- `CLUSTER_MSG_LSTM_STEP_ERROR` worker -> coordinator

Payload sizes require increasing cluster packet buffers beyond 256 bytes. `LSTM_STEP_REQUEST` is ~522 bytes payload plus header. Use at least 640 bytes for cluster packet buffers.

Use a separate `step_seq` in the payload, not just header seq, so retries can be idempotent.

## Reliability requirement

UDP retry must be idempotent.

A worker recomputing a repeated step would update its local `c` twice and corrupt the run. Each worker must cache per-layer:

- last `step_seq`
- last h-slice result

On duplicate `(layer, step_seq)`, re-send cached result without recomputing or mutating `c`.

## Export / shard artifacts

Do not build a shard exporter for Phase 3.

All three boards can keep the same full RILM model image because H256 fits. Workers simply compute row ranges by board ID. A per-board RILM shard exporter is for later models that do not fit on one board.

Useful host artifact:

- `tools/lstm_step_golden.py`: host-side golden for one-step/layer debugging against firmware arithmetic.

## Firmware changes

Immediate prerequisites:

- Increase UDP packet buffers from 256 to >=640.
- Disable WiFi modem sleep on coordinator and workers via `WiFi.setSleep(false)`.

Phase 3 firmware gate:

- Add `CLUSTER_LSTM_PROOF=1` envs:
  - `cluster_coord_ap_lstm`
  - `cluster_worker1_ap_lstm`
  - `cluster_worker2_ap_lstm`

Worker boot path:

- load model partition
- clone tensors to PSRAM
- convert LSTM weights to int4
- resolve model tensors
- init activation LUT
- allocate worker slice state/cache

Coordinator boot path:

- load model partition enough for embedding/fc
- run cluster model loop
- broadcast step requests
- wait/retry for both slices
- assemble h
- print parity receipt

Refactor recommended:

- Extract a shared gate-range compute helper so single-board and worker arithmetic cannot diverge by copy/paste.

## Host tools

- Extend `tools/test_cluster_protocol.py` for new LSTM payloads.
- Add `tools/verify_cluster_lstm.py` to compare coordinator `CLUSTER_LSTM_RECEIPT` against p22 baseline output.
- Add `tools/lstm_step_golden.py` for layer/token debugging.

## Verification gates

1. Host protocol tests pass.
2. Single-layer parity: one token, layer 0, zero state. h CRC must match golden.
3. Full-token parity: one `model_step` through all 3 layers and FC, argmax matches p22.
4. Utility-suite parity: all 8 utility outputs byte-identical to p22 receipt.
5. Fault injection: worker loss produces clear `passed:false`, not a hang.
6. Timing receipt: record cluster ms/token honestly versus p22.

## Risks / kill criteria

- WiFi power-save latency: disable it first. If RTT remains high, parity proof still matters but speedup claims die.
- Non-idempotent retries: hard correctness risk; dedup cache is mandatory.
- Float parity: likely okay on identical S3s; if it fails, suspect row/order differences first.
- Kill if byte-exact parity is not achieved within two focused sessions after single-layer parity works.
- Kill H256 speed optimization attempts. H256 is mechanism proof; TinyStories / larger models are the real cluster target.
