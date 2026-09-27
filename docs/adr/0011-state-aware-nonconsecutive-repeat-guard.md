# ADR 0011: Block non-consecutive repeated commands on an unchanged workspace

Date: 2026-09-27. Status: implemented and cloud-regressed; model probe completed.

## Context

The earlier loop guard detected only consecutive identical commands. The three-task
development traces showed that repeated commands could be separated by unrelated
actions, so they were executed again without a workspace change. This wasted tool
budget and let the agent remain stuck. The desired behavior is to preserve the
historical evidence and reminder while giving the model one opportunity to choose
a different action.

## Decision

- Define an action signature as SHA-256 of the exact command and its pre-action
  workspace digest. The same command remains legal after the workspace changes.
- Reconstruct completed signatures and their first real receipts from durable
  `events.jsonl` and the action journal on every run/resume. Do not add checkpoint
  fields that duplicate this durable history.
- On the first repeat for an unchanged workspace, do not invoke the executor or
  consume tool budget. Append a `no_progress` event with `executed: false`; return
  a protocol result containing the signature and a bounded excerpt of the actual
  prior receipt, followed by a separate Harness warning.
- If the model requests that same signature again after the warning, stop the run.
  Preserve all events and messages; do not fabricate a tool receipt.
- At context compaction, retain the immediately preceding real action/receipt,
  the rejected assistant action, the non-executed protocol result, and the warning.
  Apply this to both supported prompt policies.

## Verification and limits

Focused tests cover non-consecutive repeats, receipt preservation, context
compaction, workspace-change allowance, and resume without re-execution. The local
full suite passed 277 tests with 12 platform-gated skips and no failures; the
AutoDL Linux full suite passed 277/277 with no skips. On the fixed three-task
probe, each run stopped after the model repeated a blocked command; no patch
passed independent evaluation. The guard is verified, but it did not improve the
autonomous-repair gate (still 0/3). See
`docs/AUTODL_REPEAT_GUARD_PROBE_20260927.md` for sanitized results.

This guard catches only exact command repetitions for an unchanged workspace.
Semantically equivalent commands with different text are not classified as exact
repeats and remain subject to the existing step/tool/time budgets and independent
evaluation.
