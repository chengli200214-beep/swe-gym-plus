# CodeAgentBench

CodeAgentBench is a small, auditable Coding Agent harness built around the
SWE-Gym task shape. It keeps the upstream dataset/agent choices behind adapters
and owns the reliability layer: isolated workspaces, bash execution, durable
action receipts, conservative recovery, evidence-aware context compression,
whole-task budgets, independent patch evaluation, and multi-rollout selection.

The repository intentionally stops short of claiming a sealed SWE-Gym
benchmark result. The demo fixture is a deterministic integration test; real
claims require a frozen SWE-Gym revision, a quality report for every task, and
complete artifacts.

As of 2026-09-29, the current 7B Base has produced one nonempty patch that
passed independent tests in a frozen three-task **development** retest with
public visible-test commands. That Agent run nevertheless exhausted its step
budget, the other two patches did not pass, and a follow-up configuration
returned 0/3. There is no demonstrated stable repair-rate or SFT improvement.
The historical 20-task/250-action export fails the current typed-tool training
alignment gate; no new aligned SFT has been started. Details and caveats are in
[`docs/VGPU32_RETEST_20260929.md`](docs/VGPU32_RETEST_20260929.md) and
[`docs/TRAINING_DATA_READINESS_20260928.md`](docs/TRAINING_DATA_READINESS_20260928.md).
The latest completion matrix, including a passing but non-completed train-task
candidate, is in [`docs/PROJECT_STATUS_20260929.md`](docs/PROJECT_STATUS_20260929.md).

## Implemented scope

- P0: versioned task manifest, agent/evaluator data separation, local or git
  workspace creation, bash tool execution, action/event artifacts, formal test
  evaluation, and deterministic model-free quality controls.
- P1: checkpoint/recovery decisions, cumulative token/time/cost budgets,
  no-progress detection, evidence-aware compression, independent candidates,
  rule-based selection, honest metrics, and an optional FastAPI run index.
- P2: SFT data conversion and historical BF16 LoRA experiments with negative
  or inconclusive evaluation results; the current typed-tool training-data
  audit and new controlled comparison remain open.
- P3/GRPO: formal reward interface is present, but no GRPO benchmark claim is
  made. GRPO remains a separate follow-up experiment.

The latest source-observation implementation and newly frozen development gate is
[`docs/AUTODL_SOURCE_OBSERVATION_20260927.md`](docs/AUTODL_SOURCE_OBSERVATION_20260927.md).
Typed edits now require genuine versioned read evidence. The three
development tasks retain their original admission controls and evaluator
separation. This engineering change does not itself establish model repair
improvement.

The previous typed-editor implementation and three-task development diagnostic is
[`docs/AUTODL_TYPED_EDIT_20260927.md`](docs/AUTODL_TYPED_EDIT_20260927.md):
0/3 autonomous repairs at that time, with failures caused by guessed
paths/source before any actual source read. The edit/recovery contracts pass.
These previously exposed development tasks are not a fresh sealed evaluation.

The preceding same-machine AutoDL experiment (2026-09-27) is in
[`docs/AUTODL_EXPERIMENT_20260927.md`](docs/AUTODL_EXPERIMENT_20260927.md).
The NsJail backend runs agent tools, admission controls and independent tests
on the AutoDL instance without falling back to an unisolated host shell.
Early 7B diagnostics produced an incorrect edit; later retests are documented
above. Thirteen runs on one task are not thirteen successful trajectories.
The next acceptance gate is documented in
[`docs/EXECUTION_PLAN_20260927.md`](docs/EXECUTION_PLAN_20260927.md).

Earlier training results (2026-09-27) are in
[`docs/SERVER_HANDOFF_20260927.md`](docs/SERVER_HANDOFF_20260927.md),
with a machine-readable summary in
[`experiments/progress-20260927/summary.json`](experiments/progress-20260927/summary.json).
Two 3B BF16 LoRA rounds used 20 successful training tasks / 250 action examples;
each admitted evaluation set scored Base 1/12 and SFT 0/12. A subsequent
three-task context intervention probe produced no patches. These are diagnostic
results, not evidence of a repair-rate improvement. Private adapters and raw
traces are not stored in this public repository; the handoff describes their
separate migration package and pending action-protocol fix.

Earlier results are recorded in
[`experiments/sft-v0/evaluation/comparison.md`](experiments/sft-v0/evaluation/comparison.md),
and the remote reproduction steps are in
[`docs/experiment-reproduction.md`](docs/experiment-reproduction.md).
The current ModelScope workspace status and cross-machine handoff are in
[`docs/modelscope-experiment-handoff.md`](docs/modelscope-experiment-handoff.md).
The stage-by-stage completion audit and current blockers are in
[`docs/completion-status-2026-09-25.md`](docs/completion-status-2026-09-25.md).

The original Hercules checkout remains in `../upstream/testzeus-hercules` as a
reference. Its browser-specific AgentTestBench code is not imported by this
project.

## Quick start

```powershell
cd swe-gym-plus
python -m pip install -e ".[dev,service]"
python -m pytest
python -m codeagentbench validate-manifest data/manifests/demo.json
```

To run the deterministic demo agent:

```powershell
python -m codeagentbench run data/manifests/demo.json demo-calculator --script examples/demo_script.json
```

The resulting run directory contains `run.json`, `checkpoint.json`,
`actions.jsonl`, `events.jsonl`, the isolated workspace and `summary.json`.

For the API baseline, set `DEEPSEEK_API_KEY` in a local environment and use a
manifest whose `repo`, `base_commit` and evaluation fields have been frozen.
The adapter records the returned model usage; it never stores the API key in an
artifact.
For `deepseek-flash`, events also contain `estimated_cost_cny` from returned
cache-hit/miss and output-token usage at the published peak-hour rates. This is
an estimate, not a provider bill or a spending limit; it is null if usage is
missing or the model has no configured rate. The adapter does not know a real
USD charge, so `--max-cost-usd` is **not** an effective DeepSeek spending cap.
Check the provider balance before and after a bounded experiment; use token,
step and time budgets as additional guards, not as a substitute for a hard
currency cap.
Recheck the [official pricing](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)
before each paid experiment, since rates may change.

Real DeepSeek CLI runs require an explicitly configured isolated executor
(`CODEAGENTBENCH_EXECUTOR=bwrap` or `nsjail` where provisioned),
`DEEPSEEK_MIN_BALANCE_CNY` (the account balance floor), and
`DEEPSEEK_MAX_OUTPUT_TOKENS<=1024`. Before each paid request the adapter checks
the official CNY balance and rejects requests over 100 KB. Configure the
isolated backend so the agent can access only its per-run checkout and required
runtime dependencies, not private experiment files, credentials or the network.
This is defense in depth, **not a guaranteed
financial hard cap**: billing can lag, prices can change, and other users of
the same account can spend concurrently. For a ¥30 maximum from a ¥35.40
starting balance, set the floor to ¥5.40 and verify the provider balance after
each short run.
Use `run ... --skip-evaluation` for a paid DeepSeek rollout, then exit that
process and run `evaluate-run MANIFEST RUN_ID --repo-root ROOT` in a **new
process without `DEEPSEEK_API_KEY`**. Formal tests use the same explicit
isolated backend; do not execute candidate code in a process
that still has model credentials.

Security boundary: the current local bash executor checks its starting working
directory but does **not** confine the shell process to that directory. Model-
generated commands may access other files visible to the process. Run real
rollouts only in a disposable, least-privilege environment without credentials
or private data mounted into the agent process; this is not a hardened sandbox.

## Design boundaries

`AgentTaskView` contains the issue, base repository and permitted test context.
`EvalSpec` contains the gold patch and formal evaluation mapping and is only
consumed by `Evaluator`. A selector ranks candidates from agent-visible
evidence; `oracle_coverage_at_k` is reported separately after evaluation and is
never used to select a candidate.

Every action follows:

```text
persist intent -> execute -> persist receipt -> checkpoint
```

An unacknowledged side-effecting action is stopped during recovery. The harness
does not replay it just because a worker restarted. Budgets remain cumulative
across retries and candidates.

## Project layout

```text
src/codeagentbench/
  adapters/       SWE-Gym, mini-swe-agent and model boundaries
  tasks/          manifest import and quality controls
  harness/        budget, context and recovery
  sandbox/        per-run workspace and bash executor
  storage/        artifact/checkpoint store
  verification/   independent fresh-workspace evaluation
  rollout/        candidate selection
  evaluation/     aggregate metrics
  training/       SFT records and formal reward
  service/        optional minimal FastAPI index
```

The intended delivery order remains: trusted task/evaluation controls → API
baseline → reliable harness → multi-candidate comparison → service → SFT →
optional GRPO research extension.
