# Training-data readiness audit — 2026-09-28

## Decision

**Do not start another SFT run on the existing 250 action records.** This is a
deployment-alignment and evidence gate, not a judgment that all 20 historical
solutions were wrong. No model training, DeepSeek API call, or GPU-intensive
collection was performed for this audit.

The source is the private `aligned-round2-20260927` export: `actions.jsonl`
(SHA-256 `4d8412ee8bd558a659762441349124c5366e6a24b5b23ec9ea1d4ede299b6350`)
and `trajectories.jsonl`
(SHA-256 `fcb396ace5383cd5046f697dae1ad62226b579018722402cf68b05dec137eab2`).
The same action-file hash was checked on the new AutoDL host. This document
contains aggregates only; the JSONL files, task text, tool outputs, and model
weights remain outside the public repository.

## What was measured

| Grain / check | Result | Interpretation |
|---|---:|---|
| Supervised actions / distinct tasks / runs | 250 / 20 / 20 | 250 is **not** 250 independent tasks. |
| Action targets | 230 `command`, 20 `done` | No typed `search`, `read`, or `edit` targets in this export. |
| Duplicate action identities / unlinked runs | 0 / 0 | Basic action-to-trajectory linkage is intact. |
| Shell actions resembling edit / test / diff | 31 / 15 / 11 | Text heuristics, **not** verified executions. |
| Runs with a possible edit followed by a possible test | 8/20 | Sequence heuristic; does not show the test passed. |
| Final-action contexts with visible test action and exit-0 receipt | 3/20 | Only visibility in the *retained context*. The other 17 may have tested outside it. |
| Declared completed/passed trajectories; nonempty patches | 20/20; 20/20 | Historical export declarations; not freshly recertified here. |
| Declared original event paths still resolvable | 0/20 | Raw event receipts cannot be re-audited at the recorded paths on this host. |

The patch syntax scan found no modified test files, no comment/blank-only
patches, and no newly added `TODO` tokens. These are limited structural
proxies, not a semantic review or proof of correct code changes. The historical
`actions.audit.json` states `real_receipts_verified=true`, but its raw event
files are not available at the paths recorded in `trajectories.jsonl`; preserve
the historical claim without upgrading it to a fresh verification.

## Deployment-alignment gate

Running `audit_readiness` against the current `AgentRuntime._system_prompt()`
and `recent-history-v3` returned `ready=false`:

- `context_policy_mismatch`: 250/250 (`last-action-v2` in the export).
- `system_prompt_mismatch`: 250/250.
- No typed `search`, `read`, or `edit` training targets, and no observed changed
  executable action after a rejected protocol result.

The gate runs before GPU allocation in `scripts/train_aligned_sft.py`. Relaxing
it or relabeling old commands as typed actions would not make the examples
faithful to the deployed tool protocol. The old data may remain useful for a
separately labeled legacy ablation, but cannot support a claim that the current
typed-tool agent was trained on matching behavior.

## Split and provenance checks

All 20 training task IDs belong to the current 67-task training partition;
none appears in dev, eval, or `excluded_prior_exposure`. The D4c assisted
diagnostic task `getmoto__moto-4895` is not one of these 20 training tasks.
The local `split.json` byte hash initially differed from the recorded cloud
hash because Windows used CRLF line endings. Replacing CRLF with LF produces
SHA-256 `14b2e53b181ae647fc18a1bdbf340ebaa943e9f9d1cb379920de23346c16ce57`,
the historical/cloud hash. This is **not** a logical split change. Record the
normalization when reproducing the audit rather than silently accepting a
byte-level mismatch.

## Next collection and training gate

1. Keep dev/eval sealed. Collect new **train-only** trajectories with the
   actual current runtime, prompt, `recent-history-v3`, typed tool protocol,
   and independent post-run evaluation. Save original events, tool receipts,
   patch, status, task ID, split revision, runtime commit, and prompt hash.
2. Admit only completed runs whose patch independently passes; keep failed
   runs separately for diagnosis. Include genuine `search`, `read`, `edit`,
   `done`, and changed-action-after-rejection examples if they occur. Do not
   synthesize missing behavior or rewrite old histories to satisfy coverage.
3. Audit the new export at **task**, **run**, and **action** grains: receipt
   linkage, edit execution vs. claimed edit, edit-to-test sequence, patch
   structure, duplicates, split leakage, prompt-policy match, tokenizer target
   span, and truncation. Preserve original event files in a recoverable private
   location before relying on them as training evidence.
   If the source prompts expose a public `allowed_test_command`, pass an
   explicit train-only `--visible-test-manifest` to
   `scripts/prepare_aligned_sft.py`. Its command must match the source prompt
   exactly and use the bounded `python -m pytest -q -x tests/...` form. The
   operator must also verify the target path existed in the unmodified base
   checkout; syntax alone cannot establish that provenance. Without that
   explicit manifest, such rows remain rejected by default.
4. Only after the gate passes, run a small learnability check, then a new 7B
   LoRA. Freeze the model/runtime/decoding/selection conditions and compare
   Base vs SFT on the same untouched holdout tasks with independent grading.
   Report a negative result if no improvement occurs.

Reproduce the aggregate behavioral profile with
`scripts/audit_training_behavior.py` (private input/output paths); the unit
tests are in `tests/test_audit_training_behavior.py`. Its shell labels are
deliberately heuristic, and it never executes commands from the dataset.

## One new train-only pilot (Base 7B; not an eligible training trajectory)

After the audit, a bounded current-protocol pilot used
`getmoto__moto-5321`, one of the 20 training tasks. The same task passed
environment controls on the new AutoDL host: unfixed checkout **1 failed,
30 passed**; official patch **31 passed**. This validates the task environment,
not the model. The local `Qwen2.5-Coder-7B-Instruct` Base run used
`recent-history-v3`, NsJail, temperature 0, at most 12 model steps/tool calls,
100,000 tokens and 360 seconds. It did not use DeepSeek or gold source in its
model input. Remote artifacts are at
`/root/autodl-tmp/experiments/trainpilot-5321-base-20260928/`; the control
report is `/root/autodl-tmp/experiments/trainpilot-5321-controls-20260928.json`.

Result: **failed, empty patch, independent evaluation failed**. The journal
shows the first model action was a typed `edit` to the correct implementation
file, rejected *without execution* because the model had not read that source.
The next four typed searches were syntactically valid; three executed with
zero literal matches, one targeted a nonexistent path and exited 1. The model
then requested three already-used searches, all blocked as no-progress, and
stopped with `no progress: repeated command requested again after it was
blocked`. There were no typed reads, executed edits, or visible post-edit tests.
The run consumed 23,325 token-accounting units, about 19 seconds of recorded
agent time, and five actual tool calls. These are **one train-task pilot**
counts, not a holdout repair rate.

The first actionable divergence is thus *edit before source evidence*;
subsequent exact-string searches compounded it by using guessed complete
signatures, even though the search tool does literal substring matching. The
no-progress guard correctly prevented repeated execution but cannot teach the
model where to read. Subsequent controlled same-task retests of literal-search
guidance and a more precise directory-path error are recorded in
[the 2026-09-29 retest](VGPU32_RETEST_20260929.md). The first yielded real
search/read/edit actions but no passing patch; the second did not change the
repeated invalid directory request. The original run had Git HEAD `99b17d7` plus a
working-tree overlay; its `runtime.py` and `source_search.py` SHA-256 values
were respectively `5cd7db52119b85e8a8e446a27f89599f43033b41daad3966bbfd45d58792115c`
and `39cbc1446fb316f1f6f3c4833357629c8890b70e8a941235a609e5c10529ba25`.
