# ESP32-S3 LSTM Speed Optimization and Benchmark Plan

> For Hermes: this is a plan/spec only. When executing, use TDD where host-testable and hardware receipts for every firmware optimization.

Goal: turn the current ESP32-S3 compressed LSTM proof from a slow scalar demo into a benchmarked, repeatable AI-performance experiment that quantifies each optimization's effect on local inference.

Architecture: keep the current proof path intact, add a benchmark harness first, then optimize one bottleneck at a time. Every speed claim must compare against the captured baseline: same board, same weights, same seed set, same token count, same serial/JSON receipt format.

Tech stack: PlatformIO Arduino ESP32-S3 firmware, custom RILM mixed int4/int8 binary weights, PSRAM state buffers, flash-mapped weights partition, host Python receipt parser, optional later ESP-NN/hand-SIMD C kernels.

Repo: /home/sikmindz/projects/esp32-s3-lstm-proof
Date: 2026-06-30

---

## Evidence-backed current state

Verified hardware proof already exists.

Board:
- ESP32-S3 on /dev/ttyACM0
- 8MB PSRAM detected
- weights partition: 0x210000, size 0x500000

Current model:
- 6,337,569 params
- profile: mixed_lstm_safe
- compressed binary: /home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin
- compressed bytes: 4,785,276
- SHA256: 709ff8a921612f0d1a075ce586d3355c895b16a100d12d8f74bb3367cdf83c61
- measured host quantized PPL: 2.7831
- delta vs fp32: +0.1864

Current runtime receipt:
- seed: `the esp32 sees `
- generated: `arm air `
- observed latency: about 1636 ms/token
- throughput: about 0.61 tok/sec

Current bottlenecks visible in /home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp:
- tensor lookup by string during every token/layer/path (`find_tensor()` inside hot path)
- generic `tensor_get()` called per multiply with dtype branch per access
- int4 values decoded one scalar nibble at a time per multiply
- int8/int4 weights dequantized to float per multiply
- all matmul accumulates float
- recurrent state is stored in PSRAM by default, likely causing slow repeated reads/writes
- per-layer debug `Serial.printf(" layer %d done\n", l)` inside model_step
- activation uses libm/stdlib `expf()` and `tanhf()` per hidden unit
- build flags are `-O2`, not tuned for speed/flash tradeoff
- benchmark is informal serial timing, not machine-readable receipt

Non-claims:
- no ESP-NN/SIMD acceleration yet
- no fixed-point packed matmul yet
- no stable multi-seed benchmark yet
- no quality regression benchmark on-device yet
- no before/after receipt schema yet

---

## Benchmark doctrine

The benchmark must answer:

1. How much faster is local AI on ESP32-S3 after each optimization?
2. Does output change?
3. Does memory use change?
4. Does it remain stable across multiple seeds and runs?
5. Which bottleneck is still dominant?

Do not benchmark only one cherry-picked token. Use multiple seeds and enough generated tokens to smooth first-token effects.

### Benchmark modes

Add compile-time benchmark modes:

- `BENCH_LOAD_ONLY`: load and parse weights, no generation.
- `BENCH_STEP_ONLY`: run warmup seed, then time N generated tokens.
- `BENCH_OPS`: time isolated kernels: embed, wih matmul, whh matmul, activation, fc projection.
- `BENCH_FULL`: full generation across fixed seed set.
- `BENCH_QUALITY`: fixed greedy generation output string per seed.

### Required receipt fields

Firmware prints one JSON line beginning with `BENCH_RECEIPT `.

Minimum fields:

```json
{
  "schema": "ri-esp32s3-lstm-bench-v1",
  "firmware_variant": "baseline|string_cache|fast_paths|fixed_point|...",
  "git_or_build_id": "manual",
  "board": "esp32s3",
  "psram_size": 8386279,
  "free_heap_start": 0,
  "free_psram_start": 0,
  "weights_sha256": "709ff8...",
  "model_profile": "mixed_lstm_safe",
  "params": 6337569,
  "compressed_bytes": 4785276,
  "seeds": ["the esp32 sees ", "once upon a ", "sensor says "],
  "tokens_per_seed": 16,
  "warmup_tokens": 4,
  "total_measured_tokens": 48,
  "ms_total": 0,
  "ms_per_token_mean": 0.0,
  "ms_per_token_p50": 0.0,
  "ms_per_token_p95": 0.0,
  "tokens_per_sec": 0.0,
  "first_token_ms_mean": 0.0,
  "steady_token_ms_mean": 0.0,
  "op_breakdown_ms": {
    "embed": 0.0,
    "lstm_wih": 0.0,
    "lstm_whh": 0.0,
    "activation": 0.0,
    "fc": 0.0
  },
  "output_by_seed": {
    "the esp32 sees ": "..."
  },
  "changed_output_vs_baseline": false,
  "max_abs_logit_delta_vs_reference": null,
  "heap_after": 0,
  "psram_after": 0,
  "watchdog_resets": 0,
  "passed": true,
  "blockers": []
}
```

### Benchmark thresholds

P0 gate for any optimization:
- boots without panic/reset
- parses same weights SHA/profile
- generates exactly requested token count
- prints one valid BENCH_RECEIPT JSON line
- no watchdog reset

Speed reporting thresholds:
- <1.10x: noise / not worth claiming
- 1.10x-1.50x: small improvement
- 1.50x-2.50x: real improvement
- 2.50x-5.00x: strong improvement
- >5.00x: major improvement; require repeat run and op breakdown to trust

Quality/output gate:
- For pure refactors, greedy output should match baseline exactly.
- For approximate activation/fixed-point kernels, output may differ, but host-side or firmware-side logit/token comparison must be added before claiming quality preservation.
- If output changes, report it as changed; do not bury it.

---

## Optimization ladder

Expected highest ROI order:

1. Benchmark harness and receipt parser.
2. Remove serial prints and hot-path string lookups.
3. Cache tensor pointers and row metadata once.
4. Keep state buffers in internal SRAM, not PSRAM, where size permits.
5. Split dtype-specific fast matmul kernels.
6. Decode int4 by byte/pair instead of calling tensor_get per scalar.
7. Replace expf/tanhf with fast LUT or polynomial approximations.
8. Use fixed-point int8/int4 accumulation where possible.
9. Add per-op benchmark breakdown to target the next bottleneck.
10. Optional: ESP-NN / hand-SIMD FFI for fully-connected style kernels.
11. Optional: retrain/export smaller/faster model if 512x3 is too slow even optimized.

---

## Phase 0: Baseline benchmark harness

Objective: create a repeatable speed/quality baseline before changing runtime behavior.

Files:
- Modify: /home/sikmindz/projects/esp32-s3-lstm-proof/src/main.cpp
- Create: /home/sikmindz/projects/esp32-s3-lstm-proof/tools/run_bench.py
- Create: /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/baseline/README.md
- Create: /home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/baseline/*.jsonl after running

Tasks:

0.1 Add BENCH_RECEIPT schema constants in firmware.
- Add `static constexpr const char *BENCH_SCHEMA = "ri-esp32s3-lstm-bench-v1";`.
- No logic change yet.
- Verify: `pio run` passes.

0.2 Add per-token timing collection.
- Store token ms in a fixed C array, e.g. `uint32_t token_ms[MAX_BENCH_TOKENS]`.
- Do not allocate dynamically.
- Verify: firmware still generates same string.

0.3 Add fixed seed set.
- Seeds:
  - `the esp32 sees `
  - `once upon a `
  - `sensor says `
- Generate 16 measured tokens per seed after warmup.
- Verify: serial contains all seed labels.

0.4 Print one JSON receipt line.
- Prefix exactly: `BENCH_RECEIPT `.
- Include total, mean, p50, p95, output strings, memory, model metadata.
- Keep JSON compact; no pretty printing.
- Verify: host can parse it with Python `json.loads()`.

0.5 Create host runner.
- `tools/run_bench.py --port /dev/ttyACM0 --out benchmarks/baseline/run-001.jsonl`
- It should reset/monitor serial until `BENCH_RECEIPT`, parse JSON, write raw log and parsed receipt.
- Verify: receipt file exists and contains valid JSON.

0.6 Run baseline 3 times.
- Command:
  `python3 tools/run_bench.py --port /dev/ttyACM0 --variant baseline --repeat 3 --out-dir benchmarks/baseline`
- Acceptance: 3 valid receipts.
- Report mean tok/sec and variance.

Gate:
- `pio run` passes
- 3 baseline receipts saved
- generated outputs captured
- baseline mean/p50/p95 tok ms reported

---

## Phase 1: Remove measurement pollution

Objective: remove obvious timing noise without changing math.

Files:
- Modify: src/main.cpp

Tasks:

1.1 Disable per-layer serial prints under benchmark mode.
- Replace `Serial.printf(" layer %d done\n", l);` with `#if DEBUG_LAYER_PRINTS` guard.
- Default `DEBUG_LAYER_PRINTS=0`.
- Verify pure output still appears.

1.2 Lower CORE_DEBUG_LEVEL in platformio.ini.
- Change `-D CORE_DEBUG_LEVEL=3` to `-D CORE_DEBUG_LEVEL=0` for benchmark builds.
- Optionally create env `esp32s3_lstm_bench` so proof/debug env remains available.
- Verify build passes.

1.3 Re-run benchmark 3 times.
- Compare against baseline.
- Expected: small but real improvement, because serial prints were inside every token/layer.

Gate:
- no output generation regression unless only debug text changed
- receipts show speed ratio vs baseline

---

## Phase 2: Cache tensor pointers and metadata

Objective: remove string lookup and repeated name formatting from every token.

Files:
- Modify: src/main.cpp

Tasks:

2.1 Add `ResolvedModel` struct.

```cpp
struct ResolvedModel {
  Tensor *embed;
  Tensor *fcw;
  Tensor *fcb;
  Tensor *wih[LAYERS];
  Tensor *whh[LAYERS];
  Tensor *bih[LAYERS];
  bool ok;
};
```

2.2 Resolve once after load.
- Build names once in setup.
- Fail fast if missing.

2.3 Change hot path to use `resolved.wih[layer]`, etc.
- Remove `snprintf` and `find_tensor` from `lstm_layer()` and `model_step()`.

2.4 Benchmark.
- Pure refactor: output should match Phase 1 exactly.

Gate:
- exact same generated outputs as Phase 1
- speed receipt generated

---

## Phase 3: Put hot state in internal SRAM

Objective: avoid PSRAM latency for recurrent state and per-token buffers.

Current hot state size estimate:
- h/c per layer: 3 layers * 2 * 512 floats = 12 KiB
- x/next_h/next_c = 6 KiB
- gates = 2048 floats = 8 KiB
- logits = 132 bytes
- total under 32 KiB

This fits in internal RAM and should not live in PSRAM.

Files:
- Modify: src/main.cpp

Tasks:

3.1 Add `internal_alloc()`.

```cpp
void *internal_alloc(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_malloc(n, MALLOC_CAP_8BIT);
  ...
}
```

3.2 Use internal allocation for `LstmState` buffers.
- Keep model weights flash-mapped.
- State should report lower free heap but same free PSRAM.

3.3 Print memory placement receipt.
- Add `state_alloc="internal"`.

3.4 Benchmark.
- Output should match prior exactly.

Gate:
- exact output match
- no heap failure
- speed ratio reported

---

## Phase 4: dtype-specific matmul fast paths

Objective: remove `tensor_get()` branch from every multiply.

Files:
- Modify: src/main.cpp
- Optional create: src/kernels.h

Tasks:

4.1 Write host-compilable kernel tests if splitting into pure C++ header is practical.
- Test int8 dequant dot against generic tensor_get dot.
- Test int4 dequant dot against generic tensor_get dot.

If host tests are too costly in PlatformIO, add a firmware self-test mode that runs tiny fixed vectors and prints `KERNEL_SELFTEST_PASS` before model load.

4.2 Add int8 dot kernel.

```cpp
float dot_i8_f32(const int8_t *w, float scale, const float *x, int n);
```

4.3 Add int4 dot kernel.

```cpp
float dot_i4_f32(const uint8_t *packed, float scale, const float *x, int n);
```

Decode two weights per byte:
- low nibble signed
- high nibble signed
- accumulate two multiplies per loop iteration

4.4 Add dtype dispatch once per matrix, not per element.
- In `lstm_layer`, select `dot_i8_f32` or `dot_i4_f32` per tensor.

4.5 Benchmark.
- Output should be numerically identical or extremely close.
- Greedy tokens should likely match; if not, add logit delta report.

Gate:
- kernel self-test pass
- model still generates requested tokens
- receipt includes changed_output flag

---

## Phase 5: activation approximation

Objective: reduce expf/tanhf cost in LSTM gates.

Files:
- Modify: src/main.cpp

Tasks:

5.1 Add exact-vs-fast activation self-test.
- Sample x from -8 to +8.
- Compute max abs error for sigmoid and tanh approximation.
- Print `ACTIVATION_SELFTEST max_sigmoid_error=... max_tanh_error=...`.

5.2 Replace sigmoid with LUT or rational approximation.
- Current sigmoid already clamps but still calls expf.
- Use a 256-entry LUT over [-8, 8] or cheap approximation.

5.3 Replace tanhf with LUT or polynomial.
- Same domain clamp.

5.4 Benchmark.
- Output may change. This is acceptable only if quality drift is measured.

Gate:
- max sigmoid abs error <= 0.01 initially
- max tanh abs error <= 0.02 initially
- generated text captured before/after
- speed improvement reported separately from matmul changes

---

## Phase 6: fixed-point / quantized-state kernels

Objective: stop doing float dequant per multiply and use integer accumulation where possible.

This is the first high-risk/high-reward phase.

Files:
- Modify: src/main.cpp or src/kernels.h
- Possibly update export tooling on MSI if per-row scales are needed

Tasks:

6.1 Add benchmark-only int8 activation/state quantization.
- Quantize `x` and `h[layer]` to int8 each step using fixed scale.
- Measure quantization overhead separately.

6.2 Add int8 weight x int8 activation dot.
- Accumulate int32.
- Convert once at end: `acc * w_scale * x_scale`.

6.3 Add int4 weight x int8 activation dot.
- Decode int4 pair, multiply int8 activation, int32 accumulate.

6.4 Compare against float fast-path.
- Receipt must include output change and optional logit delta.

Gate:
- if speed improves but output collapses, mark failed
- acceptable only if generated output is coherent enough or host-side PPL/logit drift supports it

---

## Phase 7: memory/layout export improvements

Objective: make the binary format speed-friendly, not just size-friendly.

Files:
- MSI: /home/jstevenson/projects/esp32-max-lm-training/export_esp32_quantized.py
- Local proof: weights.bin refresh
- Firmware: parser/kernel code

Potential changes:
- 16-byte align tensor payloads
- row-block align recurrent matrices
- store dtype-specific matrix metadata table
- optionally transpose matrices for memory access pattern
- add per-row scales if quality allows better fixed-point kernels

Tasks:

7.1 Add binary version bump: RILM v2.
7.2 Add alignment and metadata.
7.3 Export new weights.
7.4 Flash new weights partition.
7.5 Benchmark v1 vs v2 with identical kernels.

Gate:
- parser supports v1 and v2 or explicitly fails with clear version error
- speed effect measured independently from kernel changes

---

## Phase 8: ESP-NN / SIMD investigation

Objective: determine whether ESP-NN C kernels can accelerate the LSTM matrix-vector path.

Files:
- Create: src/esp_nn_probe.cpp or managed component setup
- Modify: platformio.ini

Tasks:

8.1 Inspect ESP-NN callable APIs available in current PlatformIO/Arduino ESP32 stack.
8.2 Create tiny fully-connected benchmark independent of the LSTM.
8.3 Compare:
- scalar int8 dot
- local int8 kernel
- ESP-NN fully-connected if callable

Gate:
- do not wire into model until standalone kernel benchmark proves speedup
- if integration cost is high or speedup absent, stop and document kill

---

## Phase 9: model architecture fallback if kernels are not enough

Objective: decide whether 512x3 LSTM is too slow for useful on-device AI even after optimization.

Candidate alternatives:
- H=384, L=3 mixed profile
- H=256, L=3 int8/int4
- Atome-LM style ternary/SSM hybrid
- sensor-specific classifier/sentinel instead of text generator

Benchmark same task:
- local interpretation/confidence over sensor-conditioned prompt
- not just free-form story generation

Gate:
- If optimized 512x3 remains <2 tok/sec, it is a proof model, not a UX model.
- For OLED/user-facing local generation, target >=5 tok/sec.
- For sentinel/confidence routing, target latency <250 ms/event.

---

## Final benchmark report format

Create:

/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/SPEED_REPORT_2026-06-30.md

Required table:

| Variant | Mean ms/token | Tok/sec | Speedup | Output changed | Free heap | Free PSRAM | Notes |
|---|---:|---:|---:|---|---:|---:|---|
| baseline | 1636 | 0.61 | 1.00x | no | TBD | TBD | scalar, debug prints |
| no_debug | ... | ... | ... | no | ... | ... | serial removed |
| cached_tensors | ... | ... | ... | no | ... | ... | no string lookup |
| internal_state | ... | ... | ... | no | ... | ... | SRAM state |
| dtype_fastpath | ... | ... | ... | maybe | ... | ... | no tensor_get branch |
| fast_activation | ... | ... | ... | maybe | ... | ... | LUT/approx |
| fixed_point | ... | ... | ... | yes/no | ... | ... | int accum |

Required charts can be plain CSV/JSON if not plotting:
- token latency distributions
- op breakdown stacked numbers
- speedup by phase

Claim boundary in report:
- benchmark measures greedy char-LSTM inference on ESP32-S3, not general LLM quality
- speedup is local to this model/firmware/board
- compressed weights already improved deployability; speed improvements are runtime-specific

---

## Immediate implementation recommendation

Execute phases 0-4 first.

Reason:
- They should preserve output exactly.
- They attack obvious overhead.
- They produce defensible speedup receipts quickly.
- They avoid risky approximate math until the benchmark is solid.

Likely early win:
- removing serial prints + cached tensor pointers + internal SRAM state + dtype-specific dot kernels.

Do not start with ESP-NN.
ESP-NN may be worthwhile, but only after the local scalar code has a clean op breakdown proving matmul dominates and after a standalone ESP-NN fully-connected probe shows real speedup in this PlatformIO stack.
