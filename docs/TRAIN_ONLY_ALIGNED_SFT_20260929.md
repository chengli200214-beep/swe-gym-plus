# Train-only aligned SFT experiment — 2026-09-29

This is an experiment ledger, not a claim of improved SWE-Gym repair rate.
DeepSeek was used **only** to collect real `train`-split Coding Agent runs.
Local 7B inference, training, quality controls and independent evaluation ran
without the API credential. Private trajectories and adapters remain on the
AutoDL instance, not in this public repository.

## Source data and gates

The six distinct admitted and independently passed source tasks are
`7524`, `7023`, `5258`, `7434`, `7635` and `6387` (all `getmoto__moto-*`).
Each source has an actual model-emitted finish, a nonempty patch, a recorded
post-edit visible test and a separate credential-free formal pass. The agent's
visible test is an existing public file in the unmodified base checkout; it
was not selected from evaluator selectors. Other attempts, including `6208`
(no patch) and `5614` (patch but no valid finish), were retained for diagnosis
and excluded from SFT. The first `6387` attempt failed during sandbox creation
because the data disk filled; the separate successful retry is the source.

The original source export has **six tasks, not 46 tasks**. After event-level
receipt verification and exact context reconstruction it produced 46
next-action records: 24 commands, 8 reads, 6 edits, 6 finishes and 2 searches.
It includes two real recovery examples, no invented actions, no target
truncation and no rejected records. Its SHA256 is
`0b97b786e7e5bb0d0f37aaadd6fce99caaeeb42b4da4319e6db0ecfe268a5a8d`.
The source, audit, split and public-test manifest are under
`/root/autodl-tmp/experiments/train-collection-20260929/` on AutoDL.

## First 7B LoRA and disjoint dev comparison

On code commit `1289657`, the local Qwen2.5-Coder-7B-Instruct ran a BF16 LoRA
for one epoch: 46/46 records encoded, zero dropped or truncated, 46 optimizer
steps, approximately 133.5 seconds and training loss 0.5689. This is a
training result, **not** task-solving evidence.

The fixed four-task `dev` split (`5876`, `5085`, `5212`, `5386`) was disjoint
from training and passed unfixed/gold admission. Existing base-checkout public
test files and all budgets were frozen before any paired run. Under the same
NsJail executor, `recent-history-v3` context policy, temperature 0, 16-step
and 16-tool-call limits, 180,000-token and 600-second caps, and 768 output
tokens, Base and SFT were alternated by task order. Both scored **0/4** on
credential-free independent evaluation; none of the eight runs made a patch.
These are development diagnostics, not a sealed final benchmark score.

For causal diagnosis, Base and SFT were then given the *same twelve recorded
training contexts* immediately before a real edit or accepted finish. Base
chose an edit on 3/6 edit contexts and `done` on 6/6 finish contexts. SFT chose
an edit on 4/6 edit contexts but `done` on 0/6 finish contexts: five outputs
requested `git diff` and one reran a test. This replay is in-sample and does
not measure generalization; it isolates a plausible stopping regression.

## Controlled follow-up and limits

Directly duplicating finish records in the JSONL was correctly rejected by
the readiness gate (`duplicate_action_identity: 18`). That rejected file is
not a training source. ADR 0013 instead records a seeded, with-replacement
sampler that weights the **same audited finish records** at training time,
without claiming additional unique trajectories. The default weight 1 keeps
the previous training behavior. The weight-4 follow-up on code commit
`cd347b9` used the original 46-line JSONL, encoded 46/46 examples with zero
truncations and zero dropped examples, ran one epoch / 46 optimizer steps in
about 127.4 seconds, and produced a separate adapter. Its trainer reported
`weighted_with_replacement`, six source finish records and 46 draws per epoch.
Training loss was 0.5872; neither that loss nor draw weighting is a task score.

A credential-free, deterministic replay of the same twelve recorded train
contexts was run with both adapters at 768 output tokens. The weighted adapter
chose `done` in **5/6** real pre-finish contexts (versus 0/6 for the first SFT)
and `edit` in **3/6** pre-edit contexts (versus 4/6 for the first SFT). The
weighted adapter's other outputs were one command at pre-finish contexts, and
one command plus two searches at pre-edit contexts. This supports a narrower
conclusion: weighting repaired much of the *in-sample finish-choice regression*
but did not improve the in-sample edit-choice count. It does not demonstrate
autonomous repair or transfer to new tasks.

The weighted adapter then ran on the **same four frozen development tasks**,
with the unchanged manifest, public visible-test commands, NsJail isolation,
context policy and budgets. All four runs produced **no patch**, all four
failed a separate credential-free evaluation, and none passed its visible
test. Tasks `5876`, `5085` and `5212` were stopped by the repeated-action
guard after 13, 11 and 14 steps; `5386` hit the 16-step cap. Across all four
runs, **54/54 model actions were shell commands**, with no typed search, read,
edit or finish. Of 45 executed tool calls, 40 exited zero and five exited
nonzero; nine additional repeated actions were blocked. The command sequence
often found and displayed implementation files but continued searching or
re-reading instead of editing. This is stronger evidence for a new-task
*exploration-to-edit* bottleneck than for an isolation or GPU failure.

More specifically, the edit adapter requires a successful versioned **typed
read** on the exact path and source before it will accept an edit. None of the
54 shell actions established that receipt. A generic single-step development
reminder to use typed read still produced shell output on **4/4** saved
contexts; it was not promoted into the normal runtime.

As a separate, train-only off-policy diagnostic, four genuine pre-edit
contexts from the unsuccessful `5614` teacher run were replayed without
executing the generated actions. Base, first SFT and weighted SFT each chose
`edit` on **2/4** contexts. These contexts were not part of the six-task
training set, but all four come from one train task and are correlated. This
shows a conditional ability to propose edits when the teacher's prior context
is supplied, not a generalization gain or a successful repair.

The development result is **0/4**, not a demonstrated gain over the original
Base 0/4 and first SFT 0/4 on these tasks. Because the four tasks have now
informed a follow-up, they are development diagnostics, not an untouched
final test. Raw events, per-run summaries, evaluator output and the hashed
private result receipt are under the same AutoDL experiment root. The eight
reproducible run/evaluation sandbox copies were removed after the journal and
evaluation receipts were verified; their ~5.6 GB of space was recovered.

Next model work should target the observed transition on **train-only**
contexts: after a bounded successful source read, the model must choose an
exact edit instead of another shell search. More epochs or finish weighting
alone are not justified by these development results. Any new data collection
must preserve unique task/action identities, actual receipts, train-only API
scope and a separate uninspected evaluation partition.

## Third 7B LoRA: typed-action exposure diagnostic

On code commit `8173ada`, the training wrapper kept the same 46 audited
records, base model, system prompt, context policy, optimizer budget and
`done_sampling_weight: 4`; it added
`typed_action_sampling_weight: 4` for the 16 existing `search`, `read` and
`edit` targets. It did **not** treat replacement draws as new tasks. The
pre-GPU readiness check passed with six unique train tasks, two real recovery
examples and zero errors. Training encoded 46/46 examples with no truncation
or drop, ran one epoch / 46 steps and produced a separate adapter. Training
loss was 0.4658, not a task outcome.

On the same twelve train contexts, the third adapter chose `edit` on **5/6**
pre-edit contexts, with one invalid response, and `done` on **6/6** pre-finish
contexts. On the saved first-step contexts of the four dev tasks, it chose
typed `search` **4/4** times, compared with four shell commands for the
finish-weight-only adapter. These probes motivated a full run; neither is a
repair-rate measurement.

Under the unchanged four-task development manifest and budgets, the third
adapter obtained **0/4** independent passes and **0/4** normally completed
visible-test-passing runs. It did produce eight successful typed-read receipts,
two successful exact edits, and one nonempty candidate patch. On `5876`, an
attempted edit was atomically rejected for `SyntaxError: unmatched ')'`; the
model then repeated edits without a patch. On `5085`, two edits wrote a patch,
but the agent hit the 16-step cap without a passing visible test and the
separate evaluator reported formal test failure. On `5212` and `5386`, search
and read became repetitive before any edit. Thus typed-action weighting
changed the observed failure stage from shell-only exploration to source
location, edit correctness and post-edit validation; it did **not** satisfy
the autonomous repair gate.

The four development tasks have now been used repeatedly for diagnosis and
cannot be advertised as a sealed holdout. Before another training run, gather
more **distinct train tasks** with real typed read→edit→test→finish receipts,
including actual rejected-edit recovery when available. Verify source quality
and split hygiene, then compare against a separately untouched evaluation
partition. Do not train on these dev traces or send them to the paid API.

The paid teacher data is small and task diversity is limited. A low training
loss, additional tool calls or an in-sample action improvement cannot be
reported as a higher autonomous repair rate. The final project still needs a
positive result or an explicit negative-result handoff, reproducibility and
security scope, and a genuinely untouched evaluation only when conditions
are frozen. No GRPO result is claimed.

Storage cleanup removed only verified, reproducible test workspaces and
completed-run/evaluator sandbox copies after their events, patches, source
hashes and evaluation receipts were checked. Those live workspaces are no
longer directly resumable, but the raw event journals, reports, aligned data,
LoRA checkpoints and their provenance remain in the private experiment root.
External backup was deferred by the user, so those assets still have
single-instance risk. The AutoDL instance remains billable while running.
