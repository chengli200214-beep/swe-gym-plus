# ADR 0006: versioned source observation before typed editing

Status: accepted for development, model repair gate not yet proven.

## Evidence and decision

The prior three-task 7B diagnostic produced no source reads, guessed paths or
before strings, and 0/3 independent passes. A longer warning alone did not
produce evidence. The normal loop now offers a typed read action, chosen by
the model after its own shell search. The controller supplies no target path,
source replacement, gold patch or fabricated observation.

Read returns a relative path, whole source lines and SHA256 of the entire file.
It runs a fixed program inside the same sandbox, rejects metadata/symlinks/
non-regular or multi-link files and bounds source size to 1 MiB. Each request
covers at most 80 lines. Its JSON observation is under 4000 characters, so the
latest real receipt survives existing recent-history-v3 projection intact.
Longer reads return the actual last line and next_line, never a partial line.

The host retains at most eight genuine read receipts, not arbitrary shell
stdout or assistant-authored claims. A typed edit must quote before from an
observed chunk of the exact path. The sandbox compares the full file SHA256
before matching/syntax validation and atomic replacement. A successful edit
invalidates old read evidence for that path. Failed/stale reads are not evidence.
The journal records canonical action JSON alongside the executed command;
offline next-action conversion replays this same observation/command contract.

Receipt recovery restores read evidence without rerunning the tool. Existing
whole-run format/edit retry limits, original issue, real errors and repeated
command warnings remain. The no-patch intervention allows a necessary read;
it no longer instructs an ungrounded immediate edit.

## Limits, provenance and rollback

This is a correctness/provenance gate for typed edits, not a new security
sandbox or proof that a model understands the source. Shell remains an agent
tool; this mechanism does not classify every arbitrary shell expression as
read-only. Neither syntactic acceptance nor a scripted fixture is repair success.
Only an autonomous non-empty patch passing independent evaluation opens training.

New development tasks are selected from the pinned upstream snapshot, excluding
historical partitions/run identities and matching issue/base-commit groups.
Selection uses repository/test-count constraints and a fixed hash order, not
gold patch contents, admission outcomes or model successes. Keep rejected
admission and negative model runs; do not substitute easier tasks afterward.

Rollback uses the prior public commit and a new run ID. Do not overwrite prior
checkpoints, weights, manifests or reports, and do not rewrite historical run
commit/hash fields to the post-publication HEAD. Old successful shell-action
trajectories remain genuine historical data; do not invent typed read calls
that never occurred to make them look aligned.

Windows tests initially caught a missing fixed-program separator and console
encoding mismatch. The program separator and UTF-8 streams were corrected;
exact source matching was not relaxed. Initial cloud full tests lacked the
training environment's bin directory in PATH; retain that failed report and
rerun with an explicit test-process PATH, not a new host Python alias.
