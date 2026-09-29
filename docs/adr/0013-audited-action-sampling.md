# ADR 0013: Weight real completion actions at sampling time

Date: 2026-09-29

## Context

The first current-protocol train set contains six distinct, independently
passed tasks and 46 next-action records, but only six final `done` targets.
In a bounded in-sample replay, the first 7B LoRA selected a further command
instead of `done` for all six final contexts. The Base model selected `done`
for all six. This is a diagnostic on training contexts, not a holdout score.

Copying the six records into the source JSONL was rejected by the readiness
gate as duplicate action identities. That gate must remain strict: repeated
rows are not new trajectories or independent evidence.

## Decision

Keep the audited source JSONL unchanged. An explicit, finite
`done_sampling_weight >= 1` may instead use PyTorch's seeded
`WeightedRandomSampler` with replacement at training time. Every draw still
points to a real, already-audited action. The number of draws per epoch equals
the number of encoded source records; a weight of four changes *probability*,
not the source task count or a guaranteed draw count. Weight one preserves the
original Trainer sampler.

The aligned training wrapper runs readiness before model allocation. Weighting
requires `next_action_only_loss` and a parseable final assistant action. The
training metrics record the sampler mode, weight, source `done` count and draws
per epoch. API credentials are not used for this training change.

## Consequences and validation

This is a single-variable training diagnostic. It may fix stopping while
increasing premature finishes or reducing exploration. Check both `done` and
`edit` behavior on the same in-sample contexts, then compare Base and SFT on
the frozen, disjoint dev set with independent evaluation. Do not claim more
than six unique train tasks or a repair-rate improvement from a lower loss.

Rollback is setting `done_sampling_weight: 1` or using the prior checkpoint.
Remove the feature if it fails to change the stopping failure mode, or if a
larger diverse train set makes weighting unnecessary.

## Follow-up: typed source-action weighting

The weight-four finish variant selected `done` on 5/6 recorded finish contexts,
but still made **54 shell commands and no typed actions** across four disjoint
development runs; all four had no patch and failed independent evaluation.
The edit tool requires a real versioned typed-read receipt, so shell `cat` or
`sed` output alone does not establish edit evidence. A generic one-step
reminder on those development contexts still yielded four shell actions.

The same audited, unique source file may now set
`typed_action_sampling_weight >= 1` for real `search`, `read` and `edit`
targets. `command` retains weight one; `done` keeps its independent weight.
Both weights are finite and validated before GPU allocation. This follow-up
holds `done_sampling_weight: 4` and changes only typed action exposure from
one to four. It does **not** add trajectories, change prompts, or treat
sampled draws as independent tasks. The source kind counts and both weights
are included in training metrics. Re-evaluate stopping, editing and
autonomous repair separately; no typed-action or repair gain is presumed.
