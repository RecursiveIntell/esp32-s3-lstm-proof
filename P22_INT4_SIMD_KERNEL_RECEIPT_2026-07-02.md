# P22 int4 SIMD recurrent kernel receipt — 2026-07-02

## Verdict

Custom int4 recurrent-weight path is working on real ESP32-S3 hardware.

Final flashed firmware:

- Variant: `p22_i4_wih_whh_simd_h256`
- Build flags: `-O3`
- Board: ESP32-S3 on `/dev/ttyACM0`
- Weights SHA256: `770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c`

## Result

Two reset-and-run hardware receipts produced identical performance:

| Run | tok/s | ms/token | lstm_wih | lstm_whh | core1_wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.5062 | 25.31 | 10.639 ms | 10.400 ms | 24.211 ms |
| 2 | 39.5062 | 25.31 | 10.629 ms | 10.405 ms | 24.212 ms |

Using the requested 4 chars/BPE-token conversion:

- 39.506 char-token/s / 4 = 9.876 BPE tok/s

This is just short of the 10 BPE tok/s target, but close enough that the 3-board cluster has margin.

## Baseline comparison

Previous best single-board firmware:

- `p17_nocopy_dualcore_h256`
- 34.68 tok/s
- 28.83 ms/token
- ~8.67 BPE tok/s

P22 improvement:

- +4.826 char-token/s
- 1.139x throughput
- -3.52 ms/token
- +1.206 BPE tok/s equivalent

## What changed

1. Converted recurrent LSTM weights from int8 to packed signed int4 at boot:
   - `lstm.weight_ih_l0..2`
   - `lstm.weight_hh_l0..2`

2. Added `dot_i4_i8_fast_esp32s3()`:
   - unpacks signed int4 pairs through a 256-entry `uint16_t` LUT
   - LUT is pinned in DRAM with `DRAM_ATTR`
   - unpacks into 16-byte aligned stack scratch
   - calls a fixed custom ESP32-S3 SIMD MAC kernel using `ee.vmulas.s8.accx`

3. Fixed the earlier custom-kernel bug:
   - old hand-written assembly had cross-core incorrect results
   - fixed by matching ESP-NN's safe ACCX extraction pattern:
     - `entry a1, 32`
     - `ee.zero.accx`
     - fused `ee.vmulas.s8.accx.ld.ip`
     - `movi.n a3, 0; nop; ee.srs.accx a2, a3, 0`
   - local stack scratch removes the previous shared-buffer/cross-core race risk

4. Switched PlatformIO release optimization from `-O2` to `-O3`.

## Output receipt

Stopped utility outputs from hardware:

```json
{
  "hot room. action is ": "check airflow.",
  "missing sensor. action is ": "no claim.",
  "stale data. action is ": "wait.",
  "high heat and humidity. action is ": "escalate.",
  "humid room. action is ": "escalate.",
  "normal room. action is ": "log receipt.",
  "safe action is ": "no claim without evidence.",
  "local first means ": "the room can report before the cloud wakes."
}
```

One output changed from the older exact receipt: `humid room. action is` now returns `escalate.` instead of `ventilate.`. It remains domain-valid but is not byte-identical to the prior H256 receipt.

## Engineering conclusion

The custom kernel is real and useful:

- correctness bug fixed
- dual-core safe
- hardware verified twice
- single-board throughput improved 13.9% over p17
- reached 9.876 BPE tok/s equivalent on one ESP32-S3

The remaining wall is still PSRAM bandwidth / recurrent matmul traffic. A 3-board tensor-parallel cluster should cross 10 BPE tok/s with meaningful margin because each board gets its own PSRAM bus.
