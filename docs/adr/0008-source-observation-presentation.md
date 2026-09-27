# ADR 0008 — Separate raw receipts from decoded source presentation

## Context and evidence

Read outputs a versioned JSON packet into stdout. Serializing the whole receipt
into a Tool result JSON string previously encoded its text twice. The 7B
development retest produced doubly escaped/malformed source edits, as well as
other failures (wrong paths, oversized reads, and a semantically wrong patch).
This is a presentation issue, not proof of a unique cause or a repair gain.

## Decision and boundaries

Raw sandbox receipts remain in the action journal and event stream. Controller
source grounding and stale-file checks continue to use the full raw packet.
A shared tool_observation conversion validates a successful typed read and
projects its exact decoded text to model-visible stdout. source_read carries
the actual path, SHA256, start/end/next line; stdout_format labels the transform.
No whitespace or newlines are normalized, no source is guessed or completed.
Arbitrary command stdout is never promoted to a typed read. Format/metadata
inconsistencies fail closed. Original task/history/interventions remain.

The same converter is used in live receipt recovery, history compaction, and
SFT event replay. Older read contexts are normalized by the same history path;
ordinary historical shell receipts keep their existing representation. Older
observations may be explicitly truncated by the existing history limit, while
raw evidence remains intact. No additional tool execution or retry is added.

## Verification and interpretation

Tests compare real journal stdout with decoded model text, confirm matching
live/SFT views, idempotent compression and recovery, and reject shell attempts
to impersonate source metadata. Exact before matching and syntax/version checks
remain unchanged. A unit test or a syntactically applied edit is not a task pass.
The frozen three-task development retest is needed to assess model behavior;
its outcome must be recorded even if it remains 0/3. Final eval stays separate.

## Rollback

Revert this presentation change as a unit (live, history, SFT converter and
prompt description). Keep raw receipts, prior trials and frozen partitions.
Use a new trial directory; never overwrite an older run to claim improvement.
