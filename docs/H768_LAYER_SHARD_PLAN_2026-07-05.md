# ESP32-S3 H768 Layer-Shard 3-Board Implementation Plan

## Architecture

Board 0 (coordinator): embed + layer0 + fc
  - Loads 7 tensors from board0 shard (4.8MB)
  - Runs embedding lookup for input token
  - Runs LSTM layer 0 locally
  - Sends hidden+cell state to board 1
  - Receives updated state from board 2
  - Runs FC head to produce output token

Board 1 (worker): layer1 only
  - Loads 4 tensors from board1 shard (4.7MB)
  - Receives hidden+cell state from coordinator
  - Runs LSTM layer 1
  - Sends updated hidden+cell state back

Board 2 (worker): layer2 only
  - Loads 4 tensors from board2 shard (4.7MB)
  - Receives hidden+cell state from coordinator
  - Runs LSTM layer 2
  - Sends updated hidden+cell state back

## Data Flow (per token)

1. coord: embed lookup(token) -> st.x[HIDDEN]
2. coord: lstm_layer(0, st.x, HIDDEN) -> st.h[0], st.c[0]
3. coord: quantize_q8(st.h[0], qx, HIDDEN) -> hidden_scale
         quantize_q8(st.c[0], qc, HIDDEN) -> cell_scale
4. coord: send STATE_FORWARD_REQUEST(board=1, qx, qc, scales) -> board1
5. board1: dequantize qx,qc -> h, c
6. board1: lstm_layer(1, h, HIDDEN) -> st.h[1], st.c[1]
7. board1: quantize st.h[1], st.c[1] -> qx', qc'
8. board1: send STATE_FORWARD_RESULT(qx', qc', scales') -> coord
9. coord: dequantize result -> st.h[1], st.c[1]
10. coord: send STATE_FORWARD_REQUEST(board=2, qx', qc') -> board2
11. board2: dequantize -> h, c
12. board2: lstm_layer(2, h, HIDDEN) -> st.h[2], st.c[2]
13. board2: quantize -> qx'', qc''
14. board2: send STATE_FORWARD_RESULT -> coord
15. coord: dequantize -> st.h[2], st.c[2]
16. coord: st.x = st.h[2]
17. coord: model_finish_fc() -> token

## Protocol Packets

Existing (from commit 41c1e28):
- CLUSTER_MSG_LSTM_STATE_FORWARD_REQUEST = 14
- CLUSTER_MSG_LSTM_STATE_FORWARD_RESULT = 15
- Payload: token_id(1) + layer_start(1) + layer_count(1) + hidden_scale(4) + cell_scale(4) + qx[HIDDEN] + qc[HIDDEN]

Change needed: CLUSTER_LSTM_STATE_HIDDEN must be configurable (512 -> 768 via RI_HIDDEN)

## Firmware Changes

1. cluster_protocol.h: Make CLUSTER_LSTM_STATE_HIDDEN = RI_HIDDEN
2. main.cpp: Add cluster_model_init_layer_shard() that:
   - Loads whatever tensors are in the partition
   - Resolves only the layers present (not all 15)
   - Allocates state for assigned layers
3. main.cpp: Replace smoke trigger with real generation tick:
   - Coordinator runs embed+layer0, sends to workers, receives results, runs FC
   - Worker receives state, runs its layer, sends back
4. main.cpp: Worker handler for STATE_FORWARD_REQUEST runs real LSTM
5. platformio.ini: Add 3 H768 layer-shard envs

## Verification

1. Build all 3 envs: pio run -e cluster_coord_ap_layer_shard_h768 ...
2. Flash layer shards to all 3 boards
3. Flash firmware to all 3 boards
4. Capture serial receipt with CLUSTER_LAYER_SHARD_TOKEN line
5. Validate token correctness