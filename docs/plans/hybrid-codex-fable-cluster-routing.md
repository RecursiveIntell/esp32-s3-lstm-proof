# Hybrid Codex / Claude-Fable Routing for 3-Board ESP32-S3 Cluster

This is the active execution routing document for:

- `/home/sikmindz/projects/esp32-s3-lstm-proof/docs/plans/2026-07-02-three-board-esp32s3-cluster-plan.md`

It supersedes:

- `docs/plans/claude-code-cluster-routing.md`
- `docs/plans/codex-cluster-routing.md`

## Live verification

Checked from `/home/sikmindz/projects/esp32-s3-lstm-proof`.

Codex:

```text
codex path: /home/sikmindz/.npm-global/bin/codex
codex version: codex-cli 0.142.5
auth: Logged in using ChatGPT
gpt-5.3-codex-spark smoke test: OK
gpt-5.5 smoke test: OK
```

Claude Code:

```text
claude auth: Login method: Claude Pro account
claude email: j.stevenson.cs@gmail.com
claude version: 2.1.199
claude fable smoke test: OK
```

## Josh's rule

Use:

- Claude Code with `--model fable` only for actually difficult sections.
- Codex for everything else.
- Codex models:
  - `gpt-5.5` for normal hard implementation
  - `gpt-5.3-codex-spark` for simple/mechanical tasks

Reason:
- Josh has more Codex usage than Claude usage.
- Fable is powerful but should be conserved for the hard parts.

## Invocation patterns

### Codex mechanical tasks

```bash
cat /tmp/cluster-task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /home/sikmindz/projects/esp32-s3-lstm-proof \
  -m gpt-5.3-codex-spark \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-cluster-task.log | tail -150
```

### Codex normal hard tasks

```bash
cat /tmp/cluster-task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /home/sikmindz/projects/esp32-s3-lstm-proof \
  -m gpt-5.5 \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-cluster-task-gpt55.log | tail -150
```

### Claude Code Fable tasks

```bash
cat /tmp/cluster-task.md | claude -p "$(cat /tmp/cluster-task.md)" \
  --dangerously-skip-permissions \
  --max-turns 30 \
  --model fable \
  --add-dir /home/sikmindz/projects/esp32-s3-lstm-proof \
  2>&1 | tee /tmp/claude-fable-cluster-task.log
```

Do not use `--bare`; it skips OAuth and can fail despite Claude Pro auth.

## Phase routing

Phase 0: Clean baseline and freeze p22
- Task 0.1 clean/archive artifacts: Codex `gpt-5.3-codex-spark`
- Task 0.2 capture p22 baseline: controller-owned hardware receipt, not delegated
- Task 0.3 cluster receipt stub: Codex `gpt-5.3-codex-spark`

Phase 1: Multi-board transport proof
- Task 1.1 role config: Codex `gpt-5.3-codex-spark`
- Task 1.2 binary packet framing: Codex `gpt-5.5`
- Task 1.3 worker echo firmware mode: Codex `gpt-5.5`
- Task 1.4 two-worker barrier sync: Claude Code `fable` for design/review if the first Codex implementation shows timing/protocol ambiguity; otherwise Codex `gpt-5.5`

Phase 2: Sharded matmul proof
- Task 2.1 synthetic fixture: Codex `gpt-5.3-codex-spark`
- Task 2.2 worker matmul command: Codex `gpt-5.5`
- Task 2.3 coordinator gather: Codex `gpt-5.5`
- Task 2.4 int4 sharded matmul: Claude Code `fable`

Phase 3: Shard existing H256 LSTM
- Task 3.1 model shard exporter: Codex `gpt-5.5`
- Task 3.2 shard flashing flow: Codex `gpt-5.3-codex-spark`
- Task 3.3 full-token H256 cluster inference: Claude Code `fable`
- Task 3.4 utility suite on cluster: Codex `gpt-5.3-codex-spark`

Phase 4: Transformer micro-model
- Task 4.1 transformer RILM format: Codex `gpt-5.5`
- Task 4.2 transformer fixture exporter: Codex `gpt-5.5`
- Task 4.3 transformer primitives: Claude Code `fable`
- Task 4.4 transformer fixture cluster run: Claude Code `fable`

Phase 5: TinyStories reduced model
- Task 5.1 model target doc: Codex `gpt-5.5`
- Task 5.2 reduced TinyStories exporter: Claude Code `fable`
- Task 5.3 hardware run: controller-owned hardware receipt with Codex `gpt-5.5` for harness fixes only

Phase 6: TinyStories-33M attempt
- Task 6.1 exact memory feasibility: Claude Code `fable`
- Task 6.2 33M shard exporter: Claude Code `fable`
- Task 6.3 hardware run and receipt: controller-owned hardware receipt; Codex `gpt-5.5` for harness fixes only

Phase 7: Sensor-grounded demo integration
- Task 7.1 reconnect sensor policy: Codex `gpt-5.3-codex-spark`
- Task 7.2 public demo receipt: Codex `gpt-5.5` for claim-boundary review, Codex `gpt-5.3-codex-spark` for mechanical wiring

## Controller verification rule

The controller owns final truth. Agent self-reports are not receipts.

After each Codex or Claude-Fable task:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
pio run
python3 -m py_compile tools/*.py
python3 tools/test_cluster_protocol.py  # once it exists
git diff --stat
git status --short
```

For hardware tasks, the controller must run the board and capture receipts directly.

## Agent discipline

- Do not run parallel agents that touch the same files.
- Use Codex in-place in the repo; avoid worktrees unless file conflicts require them.
- Use `gpt-5.5` for large firmware work involving `src/main.cpp`.
- Use `gpt-5.3-codex-spark` for short docs/scripts/config tasks.
- Use Claude Fable sparingly, only for sections marked Fable above or when Codex hits a clear reasoning wall.
- Preserve `/tmp/codex-*.log` and `/tmp/claude-fable-*.log` until the phase is committed.
- Wait for background agent completion before reimplementing the same task directly.
- If an agent writes code, controller verifies before commit.
