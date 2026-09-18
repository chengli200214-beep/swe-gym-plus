# CodeAgentBench

CodeAgentBench is a small, auditable Coding Agent harness built around the
SWE-Gym task shape. It keeps the upstream dataset/agent choices behind adapters
and owns the reliability layer: isolated workspaces, bash execution, durable
action receipts, conservative recovery, evidence-aware context compression,
whole-task budgets, independent patch evaluation, and multi-rollout selection.

The repository intentionally stops short of claiming SWE-Gym benchmark results.
The demo fixture is a deterministic integration test; real claims require a
frozen SWE-Gym revision, a quality report for every task, and complete artifacts.

## Implemented scope

- P0: versioned task manifest, agent/evaluator data separation, local or git
  workspace creation, bash tool execution, action/event artifacts, formal test
  evaluation, and deterministic model-free quality controls.
- P1: checkpoint/recovery decisions, cumulative token/time/cost budgets,
  no-progress detection, evidence-aware compression, independent candidates,
  rule-based selection, honest metrics, and an optional FastAPI run index.
- P2/P3 interfaces: interaction-preserving SFT conversion and binary formal
  reward. They are extension points, not fabricated training results.

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
GRPO.
