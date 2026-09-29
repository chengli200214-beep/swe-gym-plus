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
