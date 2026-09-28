# Private task service acceptance

This is a single-user engineering console, **not an authenticated public service**.
Bind Uvicorn to `127.0.0.1`; use an authenticated SSH/IDE tunnel. Never expose its
port publicly. The browser cannot supply a model key, command, repository path or
model path. Operators configure manifest paths and model backends server-side.

## Start

Install the `service` extra, plus `postgres` for PostgreSQL. In the repository:

```sh
export CODEAGENTBENCH_DATABASE='postgresql+psycopg:///cab_queue'
export CODEAGENTBENCH_ARTIFACT_ROOT='/private/cab-artifacts'
export CODEAGENTBENCH_MANIFESTS='["data/manifests/demo.json"]'
python -m uvicorn codeagentbench.service.app:from_environment --factory --host 127.0.0.1 --port 8765
```

In a second terminal, start a worker with the same database/artifact root:

```sh
python -m codeagentbench.service.worker --database "$CODEAGENTBENCH_DATABASE" \
  --artifact-root "$CODEAGENTBENCH_ARTIFACT_ROOT" --manifests data/manifests/demo.json \
  --executor local --script examples/demo_script.json
```

The script is a trusted deterministic demo. Real tasks require
`--executor bwrap` or `--executor nsjail` and a configured `--backend local --model-path ...` (or DeepSeek
injected into the worker process environment). Never write credentials in Git.

On this AutoDL container, use the explicit NsJail backend (no automatic fallback):

```sh
export CODEAGENTBENCH_ROOTFS='/root/autodl-tmp/nsjail-rootfs-source-20260927'
python -m codeagentbench.service.worker --database /private/acceptance-queue.db \
  --artifact-root /private/acceptance-artifacts --manifests data/manifests/demo.json \
  --executor nsjail --script examples/demo_script.json --once
```

The scripted demo checks service plumbing only; it is not a real-model repair.
Do not reuse a production queue for acceptance checks. GIT_CONFIG_* injection is
removed as a complete group in children, together with credentials, rather than
leaving an invalid GIT_CONFIG_COUNT after stripping its key fields.

## Contracts and limitations

- Atomic queue claim: SQLite `BEGIN IMMEDIATE`, PostgreSQL `FOR UPDATE SKIP LOCKED`.
- A fencing lease prevents a stale worker publishing over a recovered job.
- Heartbeat expiry marks jobs interrupted; paid or unacknowledged operations are
  never automatically replayed. Resume is an explicit operation.
- Checkpoints preserve tokens/tool calls/time, model messages, next step and
  command receipt. An acknowledged action is recovered without execution again.
- Unknown model response or tool effect refuses resume. Create a fresh run instead.
- With explicit NsJail, cancellation is observed within the running tool or
  evaluator supervisor, including when output pipes have already closed. Its
  owned process group is killed and reaped before recording a cancelled receipt.
  Linux worker sends SIGTERM then SIGKILL to the CLI group after 10 seconds as a
  final guard. Other executors retain boundary cancellation; Windows is demo-only.
- Worker deadline covers workspace preparation and execution, separately from
  formal evaluation. It is `max_seconds + 150` per subprocess, not a guarantee
  that the entire job finishes within `max_seconds`.
- DeepSeek balance guards are conservative protection, not an invoice-enforced
  CNY cap. Model transport ambiguities cannot be billed exactly by this harness.
- UI displays run status, budget, patch, events and checkpoints. Secret-shaped
  API keys are redacted as a defense-in-depth measure; artifacts remain private.
- No external private backup has been made, at the user's request.

## Acceptance

Run `python -m pytest tests/test_runtime_resume.py tests/test_service.py tests/test_contracts.py`.
For a real PostgreSQL run, set `TEST_POSTGRES_URL` to a **dedicated empty test DB**;
the fixture drops its own jobs table after each test. Never point this at an
existing deployment. Scripted k=1/2/4 tests validate engineering contracts, not
real-model accuracy or speedup.

For credential-free same-machine AutoDL acceptance, use a **new output directory**:

```sh
env -u DEEPSEEK_API_KEY PYTHONPATH=src CODEAGENTBENCH_EXECUTOR=nsjail \
  CODEAGENTBENCH_ROOTFS=/root/autodl-tmp/nsjail-rootfs-source-20260927 \
  python scripts/autodl_service_acceptance.py --root /private/new-service-acceptance
```

This uses TestClient, a private SQLite queue and real CLI/NsJail subprocesses.
It verifies queued cancellation, patch/evaluation artifact endpoints, both
running-tool and formal-evaluation cancellation, and recovery after an
acknowledged edit without replaying that edit. It does **not** validate browser
interaction, PostgreSQL deployment, authentication or real-model accuracy.
See [the measured AutoDL results](AUTODL_SERVICE_ACCEPTANCE_20260927.md).

## Local browser acceptance (2026-09-29)

With the service bound to local loopback, a real browser submitted the private
`demo-calculator` task. A separate local scripted worker completed it, and
the list showed `completed`, independent evaluation `passed`, and 96 tokens.
The summary, real event trajectory, patch, and checkpoint tabs were opened.
A second queued run was cancelled in the UI, returned `cancelled`, then
resumed from the UI and completed with independent evaluation `passed`.
The empty state was visible before the first submission.

The browser test also found that a selected run's artifact panel stayed on
`尚无此项产物` after its worker completed, even as the list updated. The page now
refreshes the selected artifact once when its status changes; a third queued
run visibly changed to `completed / passed` and the panel filled with its
summary without another click. These three runs use the trusted scripted
calculator demo on Windows; they are service/UI acceptance, **not** SWE-Gym
repair scores or proof of public-service security.
