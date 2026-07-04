# WiFi Optimization Research for ESP32-S3 Distributed Inference Cluster

Date: 2026-07-04
Context: 3-board ESP32-S3 cluster, distributed LSTM inference, currently TCP over WiFi SoftAP
Target: 20-25 BPE tok/s single-stream (~100-125 ms/token)
Current: UDP 256-row chunks = 2853 ms/token; TCP projected ~376 ms/token

## Executive Summary

Ranked 15 WiFi optimization techniques by feasibility, estimated speedup, and implementation difficulty. Top 5 actionable findings:

1. TX power is set to 2dBm (minimum) - increasing to 19.5dBm could dramatically improve range/reliability
2. TCP_NODELAY is not set - Nagle algorithm adds ~40-200ms latency per send
3. ESP-NOW v2.0 supports 1470-byte payloads - but installed Arduino-ESP32 framework only supports v1.0 (250 bytes)
4. Dual-core task pinning already done (compute on core 0, WiFi/loop on core 1) but no async I/O overlap
5. AMPDU/AMSDU aggregation is enabled by default in ESP-IDF WiFi config - can tune BA window

## Ranked Techniques

### Rank 1: Increase WiFi TX Power from 2dBm to 19.5dBm
- Feasibility: Trivial one-line change
- Current state: WiFi.setTxPower(WIFI_POWER_2dBm) at lines 1219, 1232 in main.cpp
- Finding: TX power is set to 2dBm (the minimum available). Available range: -1dBm to 19.5dBm
- ESP-IDF mapping: {8->2dBm, 20->5dBm, 28->7dBm, 34->8.5dBm, 44->11dBm, 52->13dBm, 56->14dBm, 60->15dBm, 68->17dBm, 74->18.5dBm, 76->19dBm, 78->19.5dBm}
- Estimated speedup: 1.2x-2x latency reduction (fewer retransmits, better SNR)
- Implementation: Change WIFI_POWER_2dBm to WIFI_POWER_19_5dBm
- Risk: Higher power consumption; negligible for AC-powered cluster
- Difficulty: 1/5 (1 line)

### Rank 2: Enable TCP_NODELAY (Disable Nagle Algorithm)
- Feasibility: Trivial one-line per connection
- Current state: NOT set anywhere in firmware. WiFiClient::setNoDelay(bool) is available.
- Finding: Nagle algorithm batches small TCP writes, adding 40-200ms latency. Catastrophic for request-response patterns.
- ESP32 TCP stack: Uses lwIP with Nagle enabled by default.
- Estimated speedup: 1.3x-2x latency reduction for TCP distributed inference
- Implementation: Call worker_tcp[board_id].setNoDelay(true) after connect, and cluster_tcp_worker_client.setNoDelay(true) after accept
- Risk: None for this use case
- Difficulty: 1/5 (2 lines)

### Rank 3: ESP-NOW v2.0 as Transport (Bypass TCP/IP Stack)
- Feasibility: Requires ESP-IDF upgrade or raw API calls
- Finding: ESP-NOW is Espressif connectionless WiFi protocol using vendor-specific action frames:
  - No AP association needed - direct radio-to-radio communication
  - No TCP/IP overhead - eliminates IP header, TCP handshake, ACK, window management
  - ESP-NOW v2.0: max payload = 1470 bytes (ESP_NOW_MAX_DATA_LEN_V2)
  - ESP-NOW v1.0: max payload = 250 bytes (ESP_NOW_MAX_DATA_LEN)
  - Default bit rate: 1 Mbps (configurable via esp_now_set_peer_rate_config)
  - MAC-layer ACK: built-in for unicast
  - Broadcast supported: for coordinator-to-all-workers dispatch
  - Max peers: 20 (enough for small clusters)
- Current Arduino framework limitation: Installed Arduino-ESP32 (3.20017.241212) has v1.0 header only (250 bytes). ESP-NOW v2.0 requires newer ESP-IDF. Can access via direct ESP-IDF API calls.
- For 1024-row chunks: Each row = 512 bytes int8 + metadata. Would need multiple ESP-NOW packets. Better suited for sending the input/hidden vector (~512 bytes fits in one v1.0 packet) and receiving partial gate results in chunks.
- Estimated speedup: 2x-3x over TCP for small payloads. Eliminates TCP handshake (~100ms), ACK overhead, IP stack processing.
- Implementation: Include esp_now.h, use ESP-IDF API. For v2.0 support, need ESP-IDF upgrade.
- Risk: ESP-NOW v1.0 limited to 250 bytes. Broadcast is unreliable (no ACK).
- Difficulty: 3/5 (moderate)

### Rank 4: Async/Overlapped I/O - Compute While Receiving
- Feasibility: Already have dual-core infrastructure
- Current state: Compute pinned to core 0, WiFi/loop on core 1. But I/O is blocking - cluster_tcp_read_exact() blocks until all data arrives, then compute starts.
- Finding: Current pattern is sequential: read, compute, wait, send. Proposed async pattern:
  1. Start reading chunks as they arrive (core 1, interrupt-driven)
  2. Begin computing on first chunk while later chunks still arriving (core 0)
  3. Pipeline: receive chunk N+1 while computing chunk N
  4. Send response as soon as compute finishes (core 1)
- Estimated speedup: 1.5x-2x (overlaps network I/O with compute)
- Implementation: Use FreeRTOS queues between WiFi receive task and compute task
- Difficulty: 3/5 (moderate)

### Rank 5: Persistent TCP Connections (Already Partially Done)
- Feasibility: Already implemented via cluster_tcp_ensure_worker_connected()
- Current state: Code checks if connection is alive, reconnects only if needed (1s cooldown)
- Finding: Connection reuse already implemented. TCP handshake costs ~50-100ms over WiFi.
- Improvement: Add TCP keepalive to detect dead connections faster
- Estimated speedup: Already captured; keepalive could save ~100ms on first dead-connection detection
- Difficulty: 1/5 (minor enhancement)

### Rank 6: WiFi Sleep Mode Disable (Already Done)
- Feasibility: Confirmed implemented
- Current state: WiFi.setSleep(false) at lines 1218, 1231 in main.cpp
- Finding: WiFi modem sleep is disabled. ESP-IDF default is WIFI_PS_MIN_MODEM.
- Action: No change needed. Confirmed working.
- Difficulty: Already done

### Rank 7: ESP32-S3 Dual-Core - WiFi on Core 1, Compute on Core 0
- Feasibility: Already partially implemented
- Current state: Compute task pinned to core 0, Arduino loop on core 1. WiFi task configurable via ESP_WIFI_TASK_CORE_ID Kconfig.
- Finding: WiFi task defaults to core 0 in ESP-IDF. Compute also on core 0. Potential conflict.
- ESP-IDF Kconfig: CONFIG_ESP_WIFI_TASK_PINNED_TO_CORE_1
- Estimated speedup: 1.1x-1.3x (reduces context-switch overhead)
- Implementation: Set CONFIG_ESP_WIFI_TASK_CORE_ID to core 1 in build config
- Difficulty: 2/5 (build config change)

### Rank 8: WiFi MTU and Packet Aggregation (AMPDU/AMSDU)
- Feasibility: Limited - ESP32 WiFi MTU is 1500 bytes (standard Ethernet)
- Finding: 
  - AMPDU TX: Enabled by default, BA window = 6
  - AMPDU RX: Enabled by default, BA window = 6
  - AMSDU TX: Disabled by default - can enable
  - Increasing BA window from 6 to 32 could improve aggregation efficiency
- Estimated speedup: 1.1x-1.2x with larger BA window and AMSDU enabled
- Implementation: May require custom sdkconfig (Kconfig options, not Arduino build flags)
- Difficulty: 3/5 (may require custom sdkconfig)

### Rank 9: UDP vs TCP Latency on ESP32 SoftAP
- Feasibility: Both protocols available
- Finding: 
  - UDP: ~28 bytes header, no handshake, no ACK, no retransmit
  - TCP: ~40+ bytes header, 3-way handshake (~100ms), ACK per packet
  - TCP improvement is from larger chunk sizes (1024 vs 256 rows), not TCP being faster
  - UDP with application-level segmentation could match TCP chunk sizes without TCP overhead
- Estimated speedup: UDP with manual segmentation could be 1.2x-1.5x faster than TCP for same chunk size
- Difficulty: 2/5 (application-level framing + retry logic)

### Rank 10: ESP-IDF WiFi API vs Arduino WiFi
- Feasibility: Possible but complex migration
- Finding: Arduino WiFi wraps ESP-IDF with additional abstraction. Overhead difference is small (<5%). ESP-IDF allows event-driven callback I/O instead of polling client.available().
- Estimated speedup: 1.05x-1.1x (marginal)
- Difficulty: 4/5 (major refactor)

### Rank 11: Raw 802.11 Frame Injection
- Feasibility: API exists but restrictive
- Finding: esp_wifi_80211_tx() sends raw 802.11 frames. Max 1500 bytes. Only non-QoS data frames. No standard receive path. ESP-NOW is a managed version of this - prefer ESP-NOW.
- Estimated speedup: Marginal over ESP-NOW (~5-10%)
- Difficulty: 5/5 (full custom MAC layer)

### Rank 12: PSRAM Bandwidth - Weight Loading Bottleneck
- Finding: Weights are flash-mapped (~40MB/s), not PSRAM. PSRAM used for state buffers. Compute is ~94% of token time. Weight loading is not the primary bottleneck.
- Estimated speedup: N/A
- Difficulty: 2/5 (analysis only)

### Rank 13: ESP-NOW Broadcast vs Unicast
- Finding: Broadcast is slightly faster (no ACK wait) but unreliable. Use broadcast for input vector dispatch (same data to all workers), unicast for result return.
- Estimated speedup: 1.1x-1.2x for broadcast dispatch
- Difficulty: 2/5 (moderate)

### Rank 14: Real-Time WiFi / QoS / WMM
- Finding: ESP-IDF raw 802.11 TX only supports non-QoS data frames. Not useful for dedicated SoftAP with no traffic contention.
- Estimated speedup: Negligible
- Difficulty: N/A

### Rank 15: DMA for WiFi Packets
- Finding: ESP32-S3 WiFi MAC internally uses GDMA for packet TX/RX. Already handled by WiFi driver. No application action needed.
- Estimated speedup: N/A (already done)
- Difficulty: N/A

## Published Papers

No published papers found specifically on distributed inference over WiFi on ESP32 or similar microcontrollers. Related work exists in broader edge AI (ARM/Linux, not MCUs). This cluster project appears to be novel work.

## Implementation Priority Roadmap

### Phase 1: Quick Wins (minutes, highest ROI)
1. Change TX power: WIFI_POWER_2dBm to WIFI_POWER_19_5dBm (2 lines)
2. Enable TCP_NODELAY: worker_tcp[i].setNoDelay(true) after connect (2 lines)
3. Confirm sleep disabled: Already done
- Estimated combined speedup: 1.5x-3x latency reduction
- Time: 10 minutes

### Phase 2: Architecture Improvements (hours)
4. Async I/O overlap: Pipeline WiFi receive with compute using FreeRTOS queues
5. Verify WiFi task core: Ensure WiFi task on core 1, compute on core 0
6. Tune AMPDU BA window: Increase from 6 to 32 if possible
- Estimated combined speedup: 1.5x-2x additional
- Time: 4-8 hours

### Phase 3: Transport Migration (days)
7. ESP-NOW v2.0 transport: For input vector broadcast and partial result return
8. UDP with application-level segmentation: Split large chunks across multiple UDP packets
- Estimated combined speedup: 2x-3x over current TCP
- Time: 1-3 days

### Phase 4: Advanced (research, lower ROI)
9. Raw 802.11 frame injection (very high effort, marginal gain)
10. ESP-IDF direct API migration (high effort, marginal gain)
11. QoS/WMM (not useful for dedicated SoftAP)

## Current Firmware WiFi Configuration (from main.cpp)
- Mode: WIFI_AP (coordinator) / WIFI_STA (workers)
- Sleep: WiFi.setSleep(false) - confirmed disabled
- TX power: WIFI_POWER_2dBm - WARNING: minimum!
- TCP port: 42101, UDP port: 42100
- TCP persistent connections: Yes (cluster_tcp_ensure_worker_connected)
- TCP_NODELAY: NOT SET
- Async I/O: NOT IMPLEMENTED
- Dual-core: Compute on core 0, loop/WiFi on core 1
- WiFi task core: Default (likely core 0 - potential conflict)

## Sources
- ESP-IDF ESP-NOW Documentation
- ESP-IDF WiFi API
- ESP-IDF WiFi Kconfig (github.com/espressif/esp-idf)
- Installed SDK headers (esp_now.h, esp_wifi.h, WiFiGeneric.h)
- Project firmware: src/main.cpp
- Project config: platformio.ini
