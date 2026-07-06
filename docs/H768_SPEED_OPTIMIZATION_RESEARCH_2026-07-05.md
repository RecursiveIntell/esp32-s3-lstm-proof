# ESP32-S3 Cluster H768 Speed Optimization Research

## Current performance baseline

H512 single-board local: 11.62 chars/s (86ms/token)
H512 3-board aggregate local: 34.78 chars/s
H512 distributed one-stream UDP: 3143ms/token
H256 single-board (p22 SRAM+dual-core): 32.59 tok/s (30.69ms/token)

H768 projected single-board: ~5 chars/s (compute ~200ms/token)
H768 layer-shard 3-board: ~300ms/token = ~3.3 chars/s (compute + WiFi)

## Bottleneck analysis from p16/p22 receipts

H256 p16 op breakdown (30.69ms/token):
  lstm_wih: 11.06ms (36%) — input weight dot products, int4 x int8
  sram_copy: 13.34ms (43%) — copying weights from PSRAM to SRAM
  lstm_whh: 1.98ms (6%) — recurrent weight dots, int4 x int8, SRAM-resident
  core1_wait: 16.21ms — dual-core synchronization overhead
  activation: 0.65ms — sigmoid/tanh LUT
  fc: 0.19ms — output head
  embed: 0.04ms — embedding lookup

H512 op breakdown (86ms/token, ~2.8x H256):
  lstm_wih: ~31ms — dominant, PSRAM bandwidth limited
  lstm_whh: ~5.6ms — SRAM tiled, faster per-element
  sram_copy: ~37ms — larger weights to copy
  Total compute: ~86ms

H768 projected (3.5x H512 per-layer compute):
  lstm_wih: ~109ms — dominant
  lstm_whh: ~20ms
  sram_copy: ~130ms
  Total per-layer: ~260ms
  3 layers: ~780ms if all on one board
  Layer-shard: ~260ms compute + ~100ms WiFi = ~360ms/token = ~2.8 chars/s

## Optimization directions (ranked by ROI)

### 1. ESP-NN SIMD for int4 x int8 dot products (HIGHEST ROI)

Current: int4 x int8 dot uses custom assembly (dot_i4_i8_fast_esp32s3)
Proven: ESP-NN esp_nn_dot_s8_aligned/unaligned gives 4-8x speedup for int8 x int8
Problem: ESP-NN has NO int4 x int8 kernel — only int8 x int8

Approach A: Dequantize int4 weights to int8 at boot, use ESP-NN int8 x int8
  - Doubles weight memory (int4 -> int8)
  - H768 all-int4 = 7.1MB, dequantized to int8 = 14.2MB — won't fit
  - But layer-shard: board0 4.8MB int4 -> 9.6MB int8 — won't fit in 5MB partition
  - Only feasible for H256/H512 where weights are smaller

Approach B: Write custom Xtensa SIMD int4 x int8 kernel
  - ESP32-S3 has Xtensa LX7 with 8-wide int8 SIMD
  - int4 weights can be unpacked to int8 in registers (zero overhead)
  - Custom assembly can do: load int4 pair -> sign-extend to 2x int8 -> dot with int8 input
  - Expected: 2-4x over current scalar int4 path
  - This is the real win for H768

Approach C: Use int2/ternary weights (2-bit)
  - From provekv/compressed-scorer research: PerDimScorer achieves cosine >0.99 at 8-bit
  - Ternary weights (-1, 0, +1) need only 2 bits per weight
  - H768 ternary: 14.2M * 2/8 = 3.55MB — fits on ONE board!
  - But requires retraining with ternary quantization awareness
  - Quality impact unknown for char-LSTM

### 2. SRAM weight tiling for H768 (HIGH ROI, PROVEN)

H768 layer weights: 4 * 768 * 768 = 2.36M params per layer
  int4: 1.18MB per layer — fits in SRAM? No, SRAM is 384KB usable
  int8: 2.36MB per layer — definitely doesn't fit

Partial tiling: tile only the recurrent (whh) weights in SRAM
  whh int4: 1.18MB — still too large for SRAM
  whh int8: 2.36MB — way too large

For H768, SRAM tiling is NOT feasible per-layer. The weights are too large.

Alternative: tile weight BLOCKS in SRAM
  Copy 128 rows of whh at a time (128 * 768 * 0.5 = 49KB) to SRAM
  Compute dot products against the SRAM-resident block
  Move to next 128-row block
  This gives SRAM-speed access for the hot block while streaming from PSRAM

### 3. Dual-core parallelism (PROVEN, MEDIUM ROI)

Already implemented in p16/p22:
  Core 0: gates [0, 2*HIDDEN)
  Core 1: gates [2*HIDDEN, 4*HIDDEN)
  Synchronized via FreeRTOS semaphores

For H768 layer-shard:
  Each board runs one layer
  Within that layer, split gates across cores
  This works and is already in the firmware

p16 shows core1_wait = 16.21ms — the synchronization overhead is significant
  For H768: compute is ~260ms, wait should be proportionally larger
  Optimization: reduce sync granularity (split into 4 sub-batches instead of 2)

### 4. Compressed hidden state transfer (provekv-inspired, NOVEL)

Current: hidden state = 768 floats, quantized to 768 int8 + 1 float scale = 769 bytes per transfer
  Per token: 2 transfers (coord->board1, coord->board2) = 1538 bytes

From provekv/poly-kv research:
  PerDimScorer: 8-bit per-dim uniform quantization, cosine >0.99
  TurboQuant: deterministic from dim+bits+seed, instant indexing
  FibQuant: Lloyd-Max refinement, spherical-Beta source

Approach: Apply compressed-domain scoring to hidden state transfer
  Instead of sending full 768-dim quantized state, send top-K dimensions + compressed remainder
  K=128 most significant dims at int8 = 128 + 4 (scale) = 132 bytes
  Remainder at int4 = 320 bytes (640 dims / 2)
  Total: 452 bytes vs 769 bytes = 41% reduction
  WiFi transfer time: proportional reduction

But: the quality impact of compressed hidden state on LSTM generation is unknown.
  LSTM hidden states are NOT the same as attention K/V — they're recurrent state
  Aggressive compression may destroy temporal information
  Need to test empirically

Safer approach: int4 hidden state transfer
  768 int4 values = 384 bytes + 2 scales = 386 bytes (50% reduction)
  Quality: int4 quantization of float hidden state — similar to weight quantization
  This is low-risk and directly testable

### 5. Pipeline overlapping (MEDIUM ROI)

Current: coordinator sends to board1, waits, receives, sends to board2, waits, receives
  Sequential: compute_layer0 -> send1 -> compute1 -> recv1 -> send2 -> compute2 -> recv2 -> fc

Pipeline: while board2 computes layer2 for token N, coordinator starts embed+layer0 for token N+1
  Requires: coordinator has the input for N+1 (which is the output of N)
  Cannot pipeline autoregressive generation — each token depends on the previous

Alternative: batch generation
  Generate multiple tokens from the same prompt prefix in parallel
  Not possible with standard autoregressive LSTM

Alternative: speculative decoding
  Coordinator predicts token N+1 greedily, sends to workers speculatively
  If prediction correct, skip the full pipeline
  If wrong, redo with correct token
  Expected speedup: depends on prediction accuracy
  For char-level LSTM with small vocab, greedy prediction accuracy ~40-60%
  Net speedup: 1.3-1.5x

### 6. WiFi transport optimization (LOW ROI, already optimized)

Current: UDP 256-row chunks, 24 round-trips per token for row-shard
Layer-shard: only 2 round-trips per token (much better)
  Each transfer: ~769 bytes payload + 16 byte header = 785 bytes
  Well within UDP MTU (1472 bytes)
  WiFi latency: ~50ms per round-trip on ESP32 SoftAP

Optimization: ESP-NOW v2 (1470 byte payload) — eliminates IP/TCP/UDP overhead
  But installed Arduino stack only has ESP-NOW v1 (250 bytes) — would need framework upgrade
  For 785-byte payload: still needs 4 fragments with v1, 1 with v2
  Expected improvement: 10-20ms per round-trip with v2

### 7. ProveKV validation on ESP32-S3 (RESEARCH, NOVEL)

ProveKV is Josh's Rust crate for proof-backed KV validation.
Could test on ESP32-S3 via:
  - Cross-compile provekv to Xtensa (no_std)
  - Use as hidden state integrity validator
  - Or: use compressed-scorer PerDimScorer for hidden state compression
  - Test: does compressed hidden state preserve generation quality?

This is the most novel research direction — proving that compression techniques
from the KV-cache domain transfer to recurrent state compression on MCU hardware.

## Recommended implementation order

1. int4 hidden state transfer (386 bytes vs 769 bytes) — easy, testable, 50% WiFi reduction
2. Custom Xtensa SIMD int4 x int8 kernel — 2-4x compute speedup
3. SRAM block tiling for H768 weights — partial SRAM access for hot blocks
4. Dual-core gate split refinement — reduce sync overhead
5. ProveKV/compressed-scorer hidden state validation — research test
6. ESP-NOW v2 framework upgrade — if WiFi becomes the bottleneck

## Expected combined speedup

Current H768 layer-shard: ~360ms/token = ~2.8 chars/s
With optimizations 1-4: ~150ms/token = ~6.7 chars/s (2.4x)
With all optimizations: ~100ms/token = ~10 chars/s (3.6x)

That would put H768 3-board at similar speed to H512 single-board,
but with 2.25x more parameters and layer-shard proof.