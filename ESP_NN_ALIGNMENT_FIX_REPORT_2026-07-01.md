# ESP-NN alignment fix report — 2026-07-01

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

ESP-NN path is fixed and hardware-verified.

Current best deployed variant:

- `p12_domain_h256_all_int8_espnn_aligned_guarded`
- 25.0653 tok/s
- 39.90 ms/token
- 3 repeated hardware BENCH_RECEIPT runs
- Output preserved versus scalar-good p11

This replaces the prior useful scalar p11 path as the best current firmware.

## Root cause

The direct ESP-NN dot path was calling:

```cpp
esp_nn_dot_s8_unaligned_esp32s3(x, w, n / 16)
```

That function name is misleading. Reading ESP-NN source showed:

- `esp_nn_dot_s8_unaligned_esp32s3` allows an unaligned filter pointer.
- It still requires the input pointer to be 16-byte aligned.
- The ESP-NN FC dispatcher checks this explicitly:

```c
if (filter_offset != 0 || row_len < 16 || ((uintptr_t)input_data & 15)) {
    fallback...
}
```

Our custom direct-dot path skipped that guard.

The LSTM activation buffers (`st.qx`, `st.qh`) were allocated with `heap_caps_malloc`, which does not guarantee the 16-byte alignment required by the S3 SIMD dot kernel. Result: p10 was fast but corrupted output.

## Fix

1. Declare both ESP-NN dot kernels:

```cpp
int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len);
int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *a, const int8_t *b, int32_t len_div16);
```

2. Allocate model payload copies with 16-byte alignment:

```cpp
heap_caps_aligned_alloc(16, t->payload_len, caps)
```

3. Allocate runtime state/q buffers with 16-byte alignment:

```cpp
heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
```

4. Guard the ESP-NN path:

```cpp
if ((n & 15) == 0 && (((uintptr_t)x) & 15) == 0) {
    if ((((uintptr_t)w) & 15) == 0) {
        return esp_nn_dot_s8_aligned_esp32s3(x, w, n);
    }
    return esp_nn_dot_s8_unaligned_esp32s3(x, w, n / 16);
}
```

5. Fall back to scalar if the precondition is not met.

## Hardware verification

Build:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
~/.local/bin/pio run -e esp32s3_lstm
```

Flash:

```bash
~/.local/bin/pio run -t upload --upload-port /dev/ttyACM0
```

Benchmark:

```bash
PATH=/home/sikmindz/.local/bin:$PATH \
/home/sikmindz/.local/share/uv/tools/platformio/bin/python \
  tools/run_bench.py \
  --port /dev/ttyACM0 \
  --variant p12_domain_h256_espnn_aligned_guarded \
  --repeat 3 \
  --timeout 120 \
  --out-dir benchmarks/p12_domain_h256_espnn_aligned_guarded
```

Receipt:

`/home/sikmindz/projects/esp32-s3-lstm-proof/benchmarks/p12_domain_h256_espnn_aligned_guarded/p12_domain_h256_espnn_aligned_guarded-summary.json`

Aggregate:

```json
{
  "runs": 3,
  "tokens_per_sec_mean": 25.0653,
  "tokens_per_sec_min": 25.0653,
  "tokens_per_sec_max": 25.0653,
  "ms_per_token_mean": 39.9,
  "ms_per_token_min": 39.9,
  "ms_per_token_max": 39.9
}
```

Representative output:

```json
{
  "hot room. action is ": "heck airflow.\nth",
  "missing sensor. action is ": "o claim.\nthe rec",
  "the receipt says ": "tale reading.\nth"
}
```

Full recombined outputs:

- `hot room. action is check airflow.`
- `missing sensor. action is no claim.`
- `the receipt says stale reading.`

## Comparison

Previous useful scalar p11:

- 13.5517 tok/s
- 73.79 ms/token

Fixed ESP-NN p12:

- 25.0653 tok/s
- 39.90 ms/token

Delta:

- 1.85x faster than p11
- 46.0% lower latency than p11

Against H512 p7:

- H512 p7: 3.6864 tok/s, 271.27 ms/token
- p12 H256: 25.0653 tok/s, 39.90 ms/token
- 6.80x faster
- 85.3% lower latency

Interactive budget at 25.0653 tok/s:

- 16 chars: 0.64s
- 32 chars: 1.28s
- 64 chars: 2.55s

## Claim boundary

Safe to claim:

- The ESP-NN path corruption was caused by violating the SIMD kernel's 16-byte input alignment precondition.
- The firmware now aligns runtime buffers and payload copies, guards the ESP-NN precondition, and uses the aligned kernel when possible.
- Hardware p12 is both fast and output-correct on the tested domain H256 all-int8 model.

Do not claim yet:

- ESP-NN correctness for every possible model/layout. The current claim is for this H256 all-int8 RILM layout and benchmark prompts.
- Sampling quality improvements. This pass fixes speed/correctness for greedy generation.

## Next ROI

With p12 fixed, H320 domain retraining is now more attractive: if speed roughly scales by params from p12, H320 may still land around 16 tok/s instead of the earlier scalar estimate around 8.7 tok/s.
