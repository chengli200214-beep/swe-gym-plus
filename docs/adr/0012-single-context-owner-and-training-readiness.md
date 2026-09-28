# ADR 0012: One inference-context owner and a deployment-alignment training gate

Date: 2026-09-28. Status: implementation and offline verification; no new model claim.

## Evidence and problem

The post-gate 7B BF16 LoRA used 250 rows from 20 tasks. All targets were shell
commands (230) or completion (20). There were no typed search/read/edit targets,
typed source observations, or protocol-rejection recovery examples. All six paired
pilot runs instead attempted typed search/read actions and produced no patch.
The dataset system prompt differs from the deployed prompt. Shared policy name
`last-action-v2` and valid JSON therefore do not establish deployment alignment.

Separately, Runtime calls `prepare_context` and records the resulting messages;
LocalHFModel called it again during both token estimation and generation. The
legacy policy is not idempotent around protocol recovery. Replaying captured
requests demonstrates loss of real history and active runtime interventions.
This is a verified input-transport defect, not proof of a solve-rate cause.

## Decision and boundaries

Runtime owns context selection. LocalHFModel renders the exact supplied messages
for both token estimation and generation. Direct diagnostic callers that bypass
Runtime must explicitly select their context themselves. Observability records
must describe the message sequence actually presented to the model.
New model events carry `context_contract=runtime-once-v1`. The aligned exporter
consumes those recorded messages verbatim after checking the policy, deployed
system prompt and task identity. It refuses unmarked historical events; old
events may have been logged before a second adapter-side selection. An
unexecuted protocol rejection is not a training target, but its following
valid changed action can be, provided the action has a real matching receipt.
The protocol result explicitly records `executed=false` for this audit.
Runtime's bounded policies run at the request boundary only, not again after a
large tool result. A regression test observed three selections for two requests
before this correction. Native evidence-summary compression is unchanged.
This does not silently change the historical last-action policy into a policy
that retains all interventions; measure `recent-history-v3` separately.

The aligned-SFT entry point checks the frozen deployment system prompt, explicit
deployment prompt policy, next-action supervision metadata, successful source
provenance claims, typed action coverage and protocol-recovery coverage before
entering the training stack. This is a necessary readiness gate, not a substitute
for checking original event bytes, task splits, tokenizer spans or independent
task evaluations. Passing coverage checks does not establish data quality or
generalization; the gate must not be satisfied by inventing tool receipts.
Configs must declare `deployment_prompt_policy` and
`deployment_system_prompt_sha256`; the latter must match `AgentRuntime` on the
actual deployment host. Historical config files are not rewritten to pass.

Keep old datasets, adapters and reports unchanged. New collection must use the
same frozen contract as its evaluation. Do not merely replace the system string
in old shell trajectories and call them current typed-tool supervision. Use the
already available `recent-history-v3` as a candidate in a controlled diagnostic,
not as an unmeasured replacement of published control results.

## Verification and follow-up

Use actual captured contexts for offline transport replay and small synthetic
protocol fixtures for regression tests; never run captured shell commands.
Validate all 250 rows using the original tokenizer and last-action label encoder.
Add negative tests showing that old-protocol data is rejected before GPU access.

Then, with compute resumed, compare context handling on the three consumed dev
tasks at fixed weights, prompts and budgets. Measure path discovery, successful
reads, valid edits and stop behavior separately. Only after that comparison and
data regeneration should another SFT run be considered. Any fresh final test set
must exclude all previously used tasks, not just the training split.

Rollback is a source revision plus its explicit context policy; it must never
overwrite historical artifacts. No GPU experiment or external API is started by
this change. The diagnostic report tracks unresolved navigation and supervision
coverage work.
