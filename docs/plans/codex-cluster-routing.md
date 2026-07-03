# Codex / Fable Routing for 3-Board ESP32-S3 Cluster

SUPERSEDED: Josh clarified that Fable should be used through Claude Code, while Codex handles everything else. Use `docs/plans/hybrid-codex-fable-cluster-routing.md` as the active routing document.

This file superseded `docs/plans/claude-code-cluster-routing.md` for an earlier Codex-first routing pass.

Plan being executed:
- `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/plans/2026-07-02-three-board-esp32s3-cluster-plan.md`

## Live verification

Checked from `/home/sikmindz/projects/esp32-s3-lstm-proof`:

```text
codex path: /home/sikmindz/.npm-global/bin/codex
codex version: codex-cli 0.142.5
auth: Logged in using ChatGPT
```

Model smoke tests:

```text
gpt-5.3-codex-spark: OK
gpt-5.5: OK
fable via Codex ChatGPT auth: rejected
```

Fable rejection:

```text
The 'fable' model is not supported when using Codex with a ChatGPT account.
```

Therefore:
- Use Codex for most implementation.
- Use `gpt-5.5` for difficult normal implementation.
- Use `gpt-5.3-codex-spark` for small/mechanical tasks.
- Use Fable only if/when an available route exists outside current Codex ChatGPT auth, or if a future auth/provider exposes it.
- If Fable is unavailable, route those sections to `gpt-5.5` with a tighter spec and controller verification.

## Model routing requested by Josh

Josh's rule:
- Fable for actually difficult sections because it is very powerful.
- Everything else uses Codex, preferably `gpt-5.5` or `gpt-5.3-codex-spark`.
- He has more Codex usage than Claude usage.

Practical current rule:
- `gpt-5.3-codex-spark`: mechanical edits, docs, scripts, simple packet tests.
- `gpt-5.5`: multi-file firmware, protocol design, model export, sharding, verification harnesses.
- `fable`: only for hard design/implementation tasks if available; currently blocked under Codex ChatGPT auth.

## Phase routing

Phase 0: Clean baseline and freeze p22
- Model: `gpt-5.3-codex-spark`
- Reason: mechanical repo hygiene + benchmark receipt.

Phase 1: Multi-board transport proof
- Task 1.1 role config: `gpt-5.3-codex-spark`
- Task 1.2 binary packet framing: `gpt-5.5`
- Task 1.3 worker echo mode: `gpt-5.5`
- Task 1.4 two-worker barrier sync: Fable preferred; current fallback `gpt-5.5`

Phase 2: Sharded matmul proof
- Task 2.1 synthetic fixture: `gpt-5.3-codex-spark`
- Task 2.2 worker matmul command: `gpt-5.5`
- Task 2.3 coordinator gather: `gpt-5.5`
- Task 2.4 int4 sharded matmul: Fable preferred; current fallback `gpt-5.5`

Phase 3: Shard existing H256 LSTM
- Task 3.1 model shard exporter: `gpt-5.5`
- Task 3.2 shard flashing flow: `gpt-5.3-codex-spark`
- Task 3.3 full-token H256 cluster inference: Fable preferred; current fallback `gpt-5.5`
- Task 3.4 utility suite on cluster: `gpt-5.3-codex-spark`

Phase 4: Transformer micro-model
- Task 4.1 transformer RILM format: `gpt-5.5`
- Task 4.2 transformer fixture exporter: `gpt-5.5`
- Task 4.3 transformer primitives: Fable preferred; current fallback `gpt-5.5`
- Task 4.4 transformer fixture cluster run: Fable preferred; current fallback `gpt-5.5`

Phase 5: TinyStories reduced model
- Task 5.1 model target doc: `gpt-5.5`
- Task 5.2 reduced TinyStories exporter: Fable preferred; current fallback `gpt-5.5`
- Task 5.3 hardware run: `gpt-5.5`

Phase 6: TinyStories-33M attempt
- Task 6.1 exact memory feasibility: Fable preferred; current fallback `gpt-5.5`
- Task 6.2 33M shard exporter: Fable preferred; current fallback `gpt-5.5`
- Task 6.3 hardware run and receipt: `gpt-5.5`

Phase 7: Sensor-grounded demo integration
- Task 7.1 reconnect sensor policy: `gpt-5.3-codex-spark`
- Task 7.2 public demo receipt: `gpt-5.5` for claim-boundary review, `gpt-5.3-codex-spark` for mechanical wiring

## Codex invocation pattern

Use one focused task spec per invocation. Preserve full logs with `tee`.

Mechanical / small tasks:

```bash
cat /tmp/cluster-task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /home/sikmindz/projects/esp32-s3-lstm-proof \
  -m gpt-5.3-codex-spark \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-cluster-task.log | tail -150
```

Hard normal tasks:

```bash
cat /tmp/cluster-task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /home/sikmindz/projects/esp32-s3-lstm-proof \
  -m gpt-5.5 \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-cluster-task-gpt55.log | tail -150
```

Fable-preferred hard tasks, if a working route becomes available:

```bash
cat /tmp/cluster-task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /home/sikmindz/projects/esp32-s3-lstm-proof \
  -m fable \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-cluster-task-fable.log | tail -150
```

Current Fable status under Codex ChatGPT auth: not supported.

## Controller verification rule

The controller, not Codex, owns final truth.

After each Codex task:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
pio run
python3 -m py_compile tools/*.py
python3 tools/test_cluster_protocol.py  # once it exists
git diff --stat
git status --short
```

For hardware tasks, the controller must run the board and capture receipts directly.
Codex self-reports are not receipts.

## Agent discipline

- Do not run parallel agents that touch the same files.
- Use `gpt-5.5` for large `src/main.cpp` firmware work; `gpt-5.3-codex-spark` may read too much and write too little on large files.
- Time-box small tasks to one Codex session; if it writes nothing, controller patches directly.
- Preserve `/tmp/codex-*.log` for audit until the phase is committed.
- Wait for background Codex completion notifications before reimplementing the same task directly.
