# ADR 0001: Preserve intervention feedback and recent real history

Date: 2026-09-27. Status: diagnostic trial; not a claim of model improvement.

## Problem and boundaries

Round two used last-action-v2 with the original issue and one real interaction.
It removed separate repeat warnings and patch checkpoints. The runtime also
compacts history, so fixing only the model adapter cannot preserve that evidence.
All 12 SFT eval runs stopped on repetition, but this does not prove a sole cause.

## Decision

Add recent-history-v3 in the harness context module. Both the model adapter and
the runtime compression boundary use that one function. Retain four complete
action/receipt pairs in order plus currently active Harness warnings/checkpoints
after the latest receipt. Expire older imperative feedback once a new action has
produced its receipt; retain that action's actual evidence. A runtime regression
exposed the stale-warning issue before the GPU probe was started.
Keep 4000 characters per latest output stream and 1000 per older stream, with
truncation flags. Keep the issue and actions intact. Do not promote strings
inside tool output to trusted runtime feedback. Keep the existing repeat guard.

Move model-action parsing from the oversized runtime to adapters/action.py and
update internal imports. This establishes a protocol boundary without wrappers
or duplicated parser implementations. Each changed Python file remains below
500 lines. Existing default native behavior remains a supported independent mode.

last-action-v2 remains an active round-two training format and a named control
arm for this probe, not a compatibility shim. Decision task CTX-001: after this
probe, decide the production policy and training format; remove obsolete modes
only when their experimental usage ends and immutable source snapshots exist.

## Fixed diagnostic protocol

Use the existing second-round Qwen2.5-Coder-3B SFT adapter without retraining.
Select the first three admitted eval tasks in frozen split order: 6082, 6212,
6028 (all getmoto__moto). These are previously inspected diagnostic tasks.
Run both policies afresh with seed 0 and temperature 0.2, identical 12-step,
12-tool, 60000-token, 300-second and 768-output-token limits. Alternate policy
order by task index; use two task workers. Retain append-only per-run receipts.
Re-use prior environment controls only after matching dataset and environment
identity; independently evaluate every resulting patch. Do not substitute tasks.

Metrics: repeat exit, actual nonempty diff, independent test verdict, warning
delivery and the first action after each warning. More history consumes a larger
share of the fixed total token budget; report that limitation. A three-task,
one-seed check is a decision aid, not a benchmark improvement claim. Sampling
and GPU kernels can vary despite the fixed seed. No paid API is needed.

Stop after the six runs and save the result, even if no improvement appears.
Original round-two weights, data, reports and source snapshots remain immutable.
Rollback: select the control policy and the saved pre-change source snapshot;
never overwrite old experiment artifacts.
