# Three-board ESP32-S3 LM system v2 target — 2026-07-03

Status: implemented first aggregate-throughput pass. Live three-board receipt captured 2026-07-03.

Measured receipt:

- Raw serial log: `receipts/local_gen_cluster_serial_20260703T225103Z.log`
- Parsed JSON: `receipts/local_gen_cluster_serial_20260703T225103Z.json`
- Parsed Markdown: `receipts/local_gen_cluster_serial_20260703T225103Z.md`
- Boards: 0 coordinator/local generator, 1 worker/local generator, 2 worker/local generator
- Model profile: `domain_h256_all_int8`
- Weights SHA256: `770ed9012099a04abf7aebc7cbbe279abd289b27b181bc364e48ea491d3dbb6c`
- Aggregate throughput: `118.2266 chars/s`
- Per-board throughput: board0 `39.4089 chars/s`, board1 `39.5062 chars/s`, board2 `39.5062 chars/s`

Claim boundary: aggregate decoded character throughput across independent ESP32-S3 local recurrent generator workers; not a single-stream latency claim and not a universal BPE token/s claim.

## Current verified base

Repo state checked 2026-07-03:

- `python3 -m py_compile tools/*.py`: PASS
- `python3 tools/test_cluster_protocol.py`: PASS
- `python3 tools/test_parse_cluster_bench.py`: PASS
- `python3 tools/test_cluster_mode_tools.py`: PASS
- `pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard`: PASS, 3 succeeded
- Current final proof: one-token TinyStories H512 distributed recurrent generation over two WiFi workers.
- Final live pass: `CLUSTER_DIST_GEN_TOKEN ... elapsed_ms=5971 status=PASS`.
- New speed-lane live pass: three local-generator boards reached `118.2266 aggregate chars/s`.

Current claim boundary:

- Safe: live hardware-proven one-token distributed TinyStories H512 recurrent generation step over two ESP32-S3 WiFi workers.
- Unsafe: optimized throughput, multi-token streaming benchmark, production-ready distributed inference, fastest ESP32-S3 LM overall.

## Strategic conclusion

Do not optimize the existing proof by simply sending more recurrent gate chunks over WiFi.

The existing distributed recurrent path is valuable because it proves correctness of sharded recurrent gate computation, not because it is the speed path. A 5971 ms/token loop cannot beat public single-board examples by raw single-stream throughput.

The best 3-board system needs two lanes:

1. **Speed lane:** avoid per-token/per-layer WiFi dependencies.
2. **Research lane:** keep distributed recurrent proof as a correctness/evidence artifact, then optimize only after the fast lane is measured.

## Target architecture

### Board roles

- Board 0: coordinator / router / receipt authority.
  - Owns benchmark schedule, prompt set, aggregation, receipts, and AP mode.
  - Runs local sentinel or small H256 model if useful.
  - Does not block every token on worker WiFi replies in speed mode.

- Board 1: generator worker A.
  - Runs full local char-LSTM model from local flash/PSRAM.
  - Emits streamed generated chars plus per-run metrics.

- Board 2: generator worker B.
  - Runs full local char-LSTM model from local flash/PSRAM.
  - Emits streamed generated chars plus per-run metrics.

### Modes

#### Mode A — aggregate throughput mode

Each board generates independently from the same fixed prompt set or assigned prompt shards. Coordinator measures total generated chars / wall-clock second across all three boards.

Goal:

- Beat public examples on aggregate decoded chars/s with honest metric separation.
- Measured first pass with all three boards running p22 H256: `118.2266 chars/s aggregate`.
- Expected ceiling estimate was ~3 × 39.52 = ~118.56 chars/s aggregate; measured result landed within ~0.3% of that estimate.
- Even if only two generator boards run p22 and coordinator only schedules, expected ceiling is ~79 chars/s aggregate.

Claim if proven:

- “Three ESP32-S3 board cluster reaches X aggregate char-generation steps/s across independent local recurrent workers, with per-board serial/WiFi receipts.”

Do not claim:

- single-stream faster than BPE transformer projects.
- distributed single-token recurrent generation is fast.

#### Mode B — best-of-N / quality mode

Board 1 and Board 2 generate alternate continuations from the same prompt using different seeds, temperatures, or model profiles. Coordinator scores outputs by deterministic local criteria:

- expected prefix/phrase match for benchmark prompts,
- safe-action vocabulary constraints,
- no-claim / receipt-policy compliance,
- optional repetition penalty / entropy sanity.

Goal:

- Use three boards to improve bounded output quality or safety without claiming raw single-stream speed.

Claim if proven:

- “Three-board ESP32-S3 system performs parallel local candidate generation and deterministic coordinator selection under a receipt-backed policy.”

#### Mode C — distributed recurrent proof mode

Keep the current H512 gate-row distributed path as the research proof. Improve it only after Mode A and comparator receipts exist.

Possible later optimizations:

- larger chunk size if payload/UDP safety allows,
- one request per worker per layer with larger payload instead of 64-row chunks,
- persistent TCP/UDP scheduling with fewer waits,
- int16 result packing or quantized gate result payloads,
- worker-side layer fusion if state transfer is compressed.

Goal:

- Move from 5971 ms/token to repeated-token benchmark and then optimize. This is research, not the immediate public speed path.

## Competitor-aware metric table

Never collapse tokenizer-native `tok/s` into one universal number.

Track these separately:

| Metric | Meaning | Why |
|---|---|---|
| model steps/s | native generation steps per second | tokenizer-native; fair within same tokenizer family |
| decoded chars/s | actual emitted characters per wall-clock second | user-visible throughput; lets char and BPE projects compare cautiously |
| BPE-equivalent tok/s | decoded chars/s / 4.0 unless measured tokenizer says otherwise | rough LLM-comparison convention |
| aggregate chars/s | sum across all active boards | fair for cluster/throughput systems, not single-stream latency |
| single-stream latency | wall time for one continuation | fair for interactive chatbot-like comparison |
| params active | total model params actively used per generation | prevents tiny 260K vs 6.34M confusion |
| receipt strength | serial log / benchmark bundle / README only | public claim quality matters |

## Outdo target

The realistic “outdo most examples” target is not one global fastest claim. It is a set of narrower wins:

1. **Aggregate cluster throughput:** beat single-board ESP32 examples in decoded chars/s by using independent local generator workers.
2. **Receipt quality:** beat most public repos by publishing deterministic receipts, exact firmware/model hashes, and replayable benchmark scripts.
3. **Recurrent WiFi cluster novelty:** keep the H512 gate-row proof as a distinct distributed recurrent claim.
4. **Physical AI safety:** add DCP-like capability/receipt layer so the system is more than a chatbot demo.

## Binary acceptance gates

### Gate 1 — clean build/protocol

Command:

```bash
python3 -m py_compile tools/*.py
python3 tools/test_cluster_protocol.py
pio run -e cluster_coord_ap_lstm_shard -e cluster_worker1_ap_lstm_shard -e cluster_worker2_ap_lstm_shard
```

Pass condition: all commands exit 0.

### Gate 2 — aggregate benchmark protocol exists — PASS

Add protocol messages or serial schema for:

- benchmark start,
- worker ready,
- worker generation result,
- generated char count,
- elapsed ms,
- model profile,
- firmware variant,
- weights SHA,
- prompt ID,
- seed/config.

Pass condition: Python protocol tests roundtrip every new message and reject bad CRC/length.

Receipt: `python3 tools/test_cluster_protocol.py` passes.

### Gate 3 — host-side parser and report generator — PASS

Add a host script that consumes serial/WiFi benchmark lines and emits JSON + Markdown:

- per-board chars/s,
- aggregate chars/s,
- prompt outputs,
- model hashes,
- board IDs,
- claim boundary.

Pass condition: parser tests pass on checked-in sample logs before firmware code is trusted.

Receipt: `python3 tools/test_parse_cluster_bench.py` passes. Parser uses the latest receipt per board to avoid double-counting periodic benchmark emissions.

### Gate 4 — two-board smoke — PASS/SUPERSEDED

With current USB availability, run at least two visible boards first.

Pass condition: two ESP32-S3 boards emit distinct board IDs and benchmark receipts in one run.

Superseded by the three-board live receipt below.

### Gate 5 — three-board live receipt — PASS

Pass condition: coordinator sees board0/1/2, starts the benchmark, receives per-board result receipts, emits one aggregate receipt.

Live health receipt after update:

```text
ok=1 board_id=0 role=coord mode=local_generator ip=192.168.4.1
ok=1 board_id=1 role=worker mode=local_generator ip=192.168.4.2
ok=1 board_id=2 role=worker mode=local_generator ip=192.168.4.3
```

Parsed aggregate receipt:

```json
{
  "aggregate_chars_per_sec": 118.22660098522168,
  "boards": [0, 1, 2],
  "elapsed_ms_wall": 406,
  "generated_chars_total": 48,
  "per_board_chars_per_sec": {
    "0": 39.40886699507389,
    "1": 39.50617283950617,
    "2": 39.50617283950617
  },
  "result_count": 3
}
```

Minimum first target:

- aggregate chars/s > best single-board local char-LSTM result in this repo.

Stretch target:

- aggregate decoded chars/s plausibly exceeds public ESP32-S3 BPE examples when estimated decoded-char rates are normalized and caveated.

### Gate 6 — public claim table

Pass condition: README includes a comparison table with separate columns for native steps/s, decoded chars/s, BPE-equivalent tok/s, aggregate chars/s, model params, tokenizer type, and receipt strength.

## Immediate implementation order

1. Add protocol/test support for aggregate benchmark result packets.
2. Add sample-log parser and report generator with RED tests first.
3. Add firmware mode for worker local generation receipt emission.
4. Add coordinator aggregate benchmark mode.
5. Run two-board smoke with available USB devices.
6. Run three-board live benchmark when all boards are powered/visible.

## Claim language to aim for

If gates pass:

> A three-board ESP32-S3 recurrent LM cluster that reaches X aggregate char-generation steps/s across local generator workers, while retaining a separate live proof of distributed H512 recurrent gate-row generation over WiFi.

This is stronger and safer than claiming “fastest ESP32 LLM.” It says exactly what was proven.
