# Claude Code Routing for 3-Board ESP32-S3 Cluster

SUPERSEDED: Josh changed routing to Codex-first. Use `docs/plans/codex-cluster-routing.md` as the active routing document.

This file records the earlier Claude Code execution split for `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/plans/2026-07-02-three-board-esp32s3-cluster-plan.md`.

## Current blocker

`claude auth status --text` currently returns:

```text
Not logged in. Run claude auth login to authenticate.
```

Do not claim Claude Code execution until this is fixed.

## Model routing requested by Josh

Use Claude Code for implementation.

Default implementation model:
- `sonnet` / Sonnet 5 equivalent if available in Claude Code

High-context / architectural review model:
- `opus` / Opus 4.8 equivalent if available in Claude Code

Very difficult sections where Josh explicitly wants Fable:
- tensor-parallel protocol correctness
- transformer runtime and sharding design
- TinyStories-33M memory/shard/export design
- custom Xtensa SIMD kernel changes if we revisit kernels

Because Claude Code model aliases are installation/account dependent, verify aliases after auth with small dry-run prompts before starting work.

## Phase routing

Phase 0: Clean baseline and freeze p22
- Model: `sonnet`
- Reason: mechanical repo hygiene + benchmark receipt.

Phase 1: Multi-board transport proof
- Task 1.1 role config: `sonnet`
- Task 1.2 binary packet framing: `sonnet`
- Task 1.3 worker echo mode: `sonnet`
- Task 1.4 two-worker barrier sync: `opus` for review, `sonnet` for implementation

Phase 2: Sharded matmul proof
- Task 2.1 synthetic fixture: `sonnet`
- Task 2.2 worker matmul command: `opus`
- Task 2.3 coordinator gather: `opus`
- Task 2.4 int4 sharded matmul: `fable` if available, otherwise `opus`

Phase 3: Shard existing H256 LSTM
- Task 3.1 model shard exporter: `opus`
- Task 3.2 shard flashing flow: `sonnet`
- Task 3.3 full-token H256 cluster inference: `fable` if available, otherwise `opus`
- Task 3.4 utility suite on cluster: `sonnet`

Phase 4: Transformer micro-model
- Task 4.1 transformer RILM format: `opus`
- Task 4.2 transformer fixture exporter: `opus`
- Task 4.3 transformer primitives: `fable` if available, otherwise `opus`
- Task 4.4 transformer fixture cluster run: `fable` if available, otherwise `opus`

Phase 5: TinyStories reduced model
- Task 5.1 model target doc: `opus`
- Task 5.2 reduced TinyStories exporter: `fable` if available, otherwise `opus`
- Task 5.3 hardware run: `opus`

Phase 6: TinyStories-33M attempt
- Task 6.1 exact memory feasibility: `fable` if available, otherwise `opus`
- Task 6.2 33M shard exporter: `fable` if available, otherwise `opus`
- Task 6.3 hardware run and receipt: `opus`

Phase 7: Sensor-grounded demo integration
- Task 7.1 reconnect sensor policy: `sonnet`
- Task 7.2 public demo receipt: `opus` for claim-boundary review, `sonnet` for implementation

## Claude Code invocation pattern

Use print mode, one focused task per invocation:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
claude -p "$(cat /tmp/cluster-task.md)" \
  --dangerously-skip-permissions \
  --max-turns 20 \
  --model sonnet \
  --output-format json 2>&1 | tee /tmp/claude-cluster-task.log
```

For hard sections:

```bash
claude -p "$(cat /tmp/cluster-task.md)" \
  --dangerously-skip-permissions \
  --max-turns 30 \
  --model fable \
  --output-format json 2>&1 | tee /tmp/claude-cluster-task-fable.log
```

If `fable` alias is rejected, rerun with:

```bash
--model opus --effort max
```

## Controller verification rule

The controller, not Claude Code, owns final truth.

After each Claude Code task:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
pio run
python3 -m py_compile tools/*.py
python3 tools/test_cluster_protocol.py  # once it exists
git diff --stat
git status --short
```

For hardware tasks, the controller must run the board and capture receipts directly.
Claude Code self-reports are not receipts.
