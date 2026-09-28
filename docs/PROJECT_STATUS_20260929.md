# CodeAgentBench completion status — 2026-09-29

This is an evidence ledger, not a benchmark announcement. The project has a
working engineering loop, but the current 7B model/training line is **not yet
complete**. Private event journals, patches, LoRA adapters, and API credentials are
not included in this public-tree report.

| Gate | Evidence | Status |
|---|---|---|
| Isolated execution and independent evaluation | AutoDL NsJail controls and frozen three-task reports; local regression suite | Implemented and exercised |
| Browser task lifecycle | Private loopback scripted submit, cancel, resume, trajectory/patch/summary/checkpoint display; selected artifact refresh fixed | Scripted UI acceptance passed; not a real-model score or public-security certification |
| 7B autonomous patch gate | Frozen dev 5164/5255/6535 visible-test retest: one **independently passing patch**, zero normally completed passing runs; follow-up retest 0/3 | Narrow patch gate observed once; reliable completion unproven |
| Current-protocol training source | Historical 20 tasks / 250 actions are shell-only with `last-action-v2`; no current typed search/read/edit targets, and raw source event paths are absent on this host | **Not ready** for the proposed typed-tool SFT |
| New LoRA and fair Base/SFT comparison | No qualifying current-protocol dataset, therefore no new aligned LoRA or sealed paired comparison | **Not started** |
| Public result claim | No sealed SWE-Gym repair-rate improvement established | **Do not claim improvement** |

## Bounded train-only collection on the current instance

No DeepSeek API was used. All examples below are train-split diagnostics with
the same local Qwen2.5-Coder-7B-Instruct and an isolated agent/evaluator.

- `getmoto__moto-5699` passed admission (unfixed fails, official patch passes).
  The first run made a semantically insufficient one-line edit and failed
  independent evaluation. A visible-test-command variant stopped after
  repeating a guessed nonexistent test file; it produced no patch.
- `getmoto__moto-7524` passed admission: unfixed 1 failed / 18 passed,
  official patch 19 passed. With the public base-checkout command
  `python -m pytest -q -x tests/test_sns`, the model performed genuine typed
  search → read → edit in `moto/sns/models.py`. The patch independently passed
  all 19 formal tests. The Agent did **not** run the visible test or emit a
  valid finish: after the edit it repeatedly searched for a nonexistent
  `tests/test_sns.py` and was stopped by the no-progress guard at step 6.
  Thus it is a **passing candidate patch, not a normally completed training
  trajectory**. The independent evaluator did not use this result to choose
  or alter the candidate.
- `getmoto__moto-5136` passed the same unfixed/gold controls and used a test
  file confirmed in the unmodified base checkout. In one bounded local 7B
  run, it searched the EC2 implementation repeatedly, hit the no-progress
  guard at step 5, made no edit and failed independent evaluation. No second
  same-task prompt search is planned.

A single-variable development retest of 7524 made the system prompt explicitly
say to execute the supplied visible-test command. The hint reached the model's
recorded request, but the model repeated the same nonexistent test-file search;
the patch again independently passed while the run failed. This did not
produce an eligible training trajectory, so the ineffective hint was reverted.
The normal runtime now additionally requires the **exact supplied visible
command** (when that completion gate is enabled) before recording a test pass;
an unrelated passing test can no longer satisfy it.

The aligned SFT exporter previously rejected any nonempty visible-test command
through its legacy blind-prompt filter. It now accepts one only when a separate
train-only manifest explicitly lists the identical, bounded public-test
command. The default remains rejection; the operator still needs base-checkout
path evidence. This changes export eligibility, **not** the failed status of
the two 7524 runs or the readiness verdict on the old dataset.

Remote evidence is under
`/root/autodl-tmp/experiments/train-collection-20260929/` on the AutoDL host:
`quality-7524.json`, `visible-7524.json`, and
`7524-base/runs/base7b-getmoto__moto-7524-visible-v1/` with the run summary,
event journal and patch. The retest is in
`7524-exact-visible-v2/runs/base7b-getmoto__moto-7524-exact-visible-v2/`.
The 5136 admission report and failed run are `quality-5136.json` and
`5136-base/runs/base7b-getmoto__moto-5136-visible-v1/`.
The private files are not copied into GitHub. Do not
reinterpret a formal pass after the failed run as a source `done` action.

## Remaining acceptance gates

1. Collect train-only runs with the **deployed prompt, context policy and
   typed actions**, retaining actual tool receipts, a post-edit visible test,
   a model-emitted `done`, a normally completed run and an independent pass.
   Keep candidate-patch-only runs separate. Do not invent missing finish or
   test actions. Any paid teacher collection requires a separate decision.
2. Audit the new source events at task/run/action grain. The training script's
   readiness gate must pass before allocating GPU for another 7B LoRA.
3. Run a bounded learnability check, then one frozen new LoRA and paired
   Base/SFT independent evaluation on untouched holdout tasks. Report a
   negative result if that is what the data show.
4. Finish the reproducibility handoff and security scope statement. The local
   UI is a private, unauthenticated demo and must not be exposed publicly as a
   multi-user service without authentication and authorization work.

The AutoDL instance remains running and billable unless the user shuts it
down. External private backup was previously deferred by the user, so remote
experiment assets still have single-instance risk.
