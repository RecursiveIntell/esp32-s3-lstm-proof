# ESP32-S3 Distributed H512 Wireless Transport Recovery Plan

Date: 2026-07-04
Repo: /home/sikmindz/projects/esp32-s3-lstm-proof

## Current receipt-backed state

Stable path:
- Transport: UDP over ESP32-S3 SoftAP
- Chunk size: 256 gate rows per worker response
- Round trips per token: 24
- Latest live receipt: `receipts/dist_gen_after_user_reset_20260704.log`
- Result: `elapsed_ms=2839`, token PASS (`dist_token=19`, `dist_char=t`)

Unstable path:
- Transport: Arduino `WiFiClient` TCP
- Intended chunk size: 1024 gate rows per worker response
- Round trips per token: 6
- Best receipt seen: ~2229 ms/token, but coordinator crash is intermittent and not release-safe
- Status: TCP code preserved behind `CLUSTER_WIFI_TCP_DIST`, but disabled for stable work

Target:
- Order-of-magnitude cut from ~2839 ms/token to ~250-300 ms/token or better.
- Anything above ~1000 ms/token is not strategically worth polishing except as research evidence.

## Blunt verdict

TCP alone will not deliver 10x.

Even stable 1024-row TCP only reduces round trips from 24 to 6. The measured/proven range is around 2229 ms/token, not 300 ms/token. This means most remaining time is worker compute + coordination overhead, not just UDP MTU.

To get an order-of-magnitude cut, transport fixes must be combined with worker compute changes and pipelining:

1. Stable large-payload transport: raw lwIP TCP sockets or segmented UDP
2. Worker-side int4 recurrent compute: avoid int8 shard compute being slower than coordinator path
3. Pipeline chunks/layers where legal: overlap network and compute; avoid stop-and-wait
4. Then consider ESP-NOW / raw 802.11 only if IP transports plateau

## Why Arduino WiFiClient TCP is the wrong next layer

Evidence from installed Arduino framework:
- `WiFiClient` uses `shared_ptr<WiFiClientSocketHandle>` and `shared_ptr<WiFiClientRxBuffer>` internally.
- `WiFiServer::available()` returns `WiFiClient` by value.
- Assignment calls `stop()`, then shares socket/rx-buffer handles.
- `connect()` toggles sockets between nonblocking and blocking internally.
- `setNoDelay()` calls `setsockopt(fd(), TCP_NODELAY, ...)` through the abstraction.

This lifecycle is fragile under repeated accept/connect/reconnect while the ESP32 is also running a large PSRAM-backed model. We already saw Guru Meditation / LoadProhibited failures in the coordinator TCP I/O path.

Conclusion:
- Stop trying to fix TCP through more `WiFiClient` guards.
- Replace it with raw lwIP/BSD sockets if TCP remains the transport.

## Best TCP recovery path: raw lwIP sockets

Use direct sockets, not `WiFiClient` / `WiFiServer`.

Coordinator:
- Maintain `int worker_fd[3] = {-1, -1, -1}`.
- `socket(AF_INET, SOCK_STREAM, IPPROTO_IP)` after model load and worker PONG discovery.
- `setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, ...)`.
- `setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, ...)` optional.
- `fcntl(fd, F_SETFL, O_NONBLOCK)`.
- Use explicit `send_all(fd, buf, len, timeout_ms)` and `recv_exact(fd, buf, len, timeout_ms)`.
- Never use temporary C++ client objects.
- Never call `.connected()`, `.stop()`, `.available()`, `.remoteIP()`, or copy client wrappers.

Worker:
- `listen_fd = socket(...)`, `bind`, `listen`, `fcntl(O_NONBLOCK)` after WiFi connected and model loaded.
- Accept one coordinator fd.
- `setsockopt(TCP_NODELAY)` on accepted fd.
- Read request header + payload with `recv_exact`.
- Compute gate rows.
- Write response with `send_all`.

Acceptance gate:
- 10 consecutive one-token runs with no Guru Meditation.
- Every run produces `CLUSTER_DIST_GEN_TOKEN ... status=PASS`.
- Median TCP elapsed must be < UDP 256 elapsed by at least 2x.
- If median > 1400 ms/token, raw TCP is not enough; move to compute/pipeline.

Expected improvement:
- Stability improvement: high.
- Latency improvement vs broken Arduino TCP: unknown.
- Latency vs stable UDP: likely 1.2x-2x unless combined with pipelining/int4.

## Best non-TCP path: segmented UDP superchunks

Current UDP failure mode:
- ESP32 WiFi MTU silently drops UDP payloads over ~1472 bytes.
- 256-row result: `5 + 256*4 = 1029` bytes; works.
- 384-row result: `5 + 384*4 = 1541` bytes; fails.
- 1024-row result: `4101` bytes; impossible in one UDP datagram.

Fix:
- Keep UDP but add application-level segmentation.
- Worker computes 1024 rows, then sends 4 UDP fragments of 256 rows each for the same logical chunk.
- Coordinator reassembles fragments before advancing layer.
- This avoids TCP/WiFiClient instability and keeps the known-good UDP stack.

Important limitation:
- Segmented UDP does not reduce worker->coordinator packet count below current 24 result packets/token.
- It can reduce request count and coordinator scheduling overhead, but not the dominant result bytes.
- It may improve latency modestly, not 10x by itself.

Acceptance gate:
- One logical chunk per worker per layer.
- Four result fragments per worker per layer.
- Token PASS for 10 consecutive runs.
- Median < 2000 ms/token or kill as insufficient.

## ESP-NOW feasibility

Primary source facts:
- ESP-NOW v2 max payload: 1470 bytes.
- ESP-NOW v1 max payload: 250 bytes.
- Installed Arduino/ESP-IDF header only exposes `ESP_NOW_MAX_DATA_LEN 250`; no `ESP_NOW_MAX_DATA_LEN_V2`.
- ESP-NOW callbacks run in a high-priority WiFi task; heavy compute must be queued to a lower-priority task.

Current framework:
- ESP-NOW v1 only.
- Cannot send 1037-byte request or 1029/4101-byte result directly.
- Would require 5+ fragments for request and 17+ fragments for 1024-row result.

Upgrade path:
- Move to ESP-IDF/newer Arduino framework with ESP-NOW v2 support.
- Use v2 1470-byte unicast for request/result fragments.
- Still limited by 1470 bytes, so 1024-row int32 result needs 3 fragments.

Expected value:
- v1: poor fit; likely worse than UDP for this payload size.
- v2: plausible replacement for UDP/TCP if raw TCP still fails; lower overhead and no association/TCP state, but not a guaranteed 10x.

Acceptance gate for ESP-NOW v2:
- Same token PASS receipt.
- Median < 1400 ms/token before further work.
- If not, abandon ESP-NOW as transport-only fix.

## Raw 802.11 / esp_wifi_80211_tx

Primary source facts:
- API exists: `esp_wifi_80211_tx()`.
- Supports beacon/probe/action/non-QoS data frames.
- Max frame still ~1500 bytes.
- No convenient receive path equivalent to UDP/TCP sockets.

Verdict:
- Do not do this now.
- ESP-NOW is the managed version of this idea.
- Raw 802.11 is only justified if building a paper/demo about custom MAC transport, not for fastest path to working inference.

## Compute bottleneck reality

The measured UDP 256 path:
- 24 request/response pairs per token.
- 2839 ms/token.
- Average pair cost ≈ 118 ms, but this includes worker compute on 256 rows and coordination.

TCP 1024 reducing to 6 round trips did not reach ~700 ms; best observed was ~2229 ms. That implies worker compute/result construction dominates more than earlier projection assumed.

Therefore the order-of-magnitude path is not transport-only.

## Highest-ROI implementation order

### Phase 0 — freeze the stable baseline

Do not disturb the stable UDP 256 receipt.

Gates:
- `python3 tools/test_cluster_protocol.py`
- `pio run` for all 9 envs
- live UDP one-token receipt PASS

### Phase 1 — raw lwIP TCP, no Arduino WiFiClient

Goal:
- Stable 1024-row TCP without coordinator crash.

Why first:
- It directly tests whether the previous TCP crash was `WiFiClient` lifecycle, not the transport itself.

Kill gate:
- If raw TCP median > 1400 ms/token after stable implementation, TCP is not enough; move on.

### Phase 2 — worker int4 recurrent shards

Goal:
- Make workers compute the same int4 recurrent path as coordinator, not int8 shard path.

Why:
- Transport-only cannot reach 10x; worker compute needs ~2x+ cut.

Gate:
- Per-chunk worker compute time drops materially.
- Token remains PASS or bounded-drift with same argmax.

### Phase 3 — pipeline

Goal:
- Stop waiting for chunk N before preparing/sending chunk N+1 where state dependencies allow it.

Safe scope:
- Within a layer, chunks at different row offsets are independent once qx/qh are fixed.
- Send all chunks for a layer to both workers, gather responses, then update LSTM state.

Expected:
- This is the biggest remaining architecture win.

### Phase 4 — ESP-NOW v2 / framework upgrade

Only after raw TCP + int4 + pipelining plateau.

Why:
- ESP-NOW v1 is payload-hostile.
- ESP-NOW v2 requires framework/IDF upgrade and still fragments 1024-row results.

## Deep Research prompt

Use this if feeding a stronger external research agent:

```
We have a 3-board ESP32-S3 cluster running a 6.34M-param char-LSTM with HIDDEN=512 and 3 layers. Coordinator runs LSTM state updates and FC argmax; two workers compute recurrent gate row shards. Each token requires, per layer, 2048 gate rows split across two workers (1024 rows per worker). Current stable transport is UDP over ESP32 SoftAP with 256-row result chunks. It produces correct one-token distributed inference in ~2839 ms/token. UDP result payload limit is ~1472 bytes, so 384-row and 512-row result datagrams are dropped. A previous Arduino WiFiClient TCP implementation with 1024-row chunks reached ~2229 ms/token but crashed the coordinator intermittently (LoadProhibited/Guru Meditation) due to WiFiClient lifecycle/null pointer/shared_ptr/socket handling. Installed Arduino ESP32 framework exposes ESP-NOW v1 only (`ESP_NOW_MAX_DATA_LEN=250`), no v2 macro. Official ESP-IDF docs say ESP-NOW v2 supports 1470-byte payloads and callbacks run in the high-priority WiFi task.

Research the best path to reduce single-stream distributed inference latency by ~10x, targeting <=300 ms/token. Focus on:
1. Raw lwIP/BSD sockets on ESP32-S3 vs Arduino WiFiClient for persistent TCP, including nonblocking connect/accept/read/write, TCP_NODELAY, keepalive, send/recv buffer sizes, and known crash patterns.
2. ESP-NOW v2 feasibility on ESP32-S3: required ESP-IDF/Arduino versions, payload size, rate config, latency vs UDP/TCP, unicast reliability, callback queue design, and fragmentation strategy for 1037-byte requests and 4101-byte int32 results.
3. UDP application-level segmentation and pipelining strategies for 1024-row logical chunks using <=1472-byte datagrams.
4. FreeRTOS dual-core design to overlap WiFi I/O and worker compute, keeping callbacks short and compute on a separate task.
5. Worker-side int4 recurrent matrix compute feasibility to reduce compute time vs current int8 shard workers.
6. Whether raw 802.11 `esp_wifi_80211_tx` offers any practical receive path or latency benefit over ESP-NOW.

Return a ranked implementation plan with expected speedups, exact ESP-IDF APIs/Kconfig options, likely pitfalls on Arduino-ESP32/PlatformIO, and binary kill criteria. Do not suggest TX power increases; this Freenove ESP32-S3 WROOM setup crashes at 8.5dBm and 19.5dBm and is stable only at 2dBm.
```

## Bottom line

If the goal is <=300 ms/token, do not spend another session tweaking Arduino `WiFiClient`.

Best shot:
1. raw lwIP TCP to prove stable large chunks;
2. int4 worker compute;
3. layer-level pipelining;
4. only then ESP-NOW v2/framework upgrade if IP transport remains the limiter.

Kill rule:
- If raw TCP + pipelined 1024-row chunks + int4 workers cannot get below ~500 ms/token, the three-board one-stream path is not worth pursuing as a performance demo. Keep it as a correctness/research proof and use H512 3-board aggregate local generation as the public utility demo.
