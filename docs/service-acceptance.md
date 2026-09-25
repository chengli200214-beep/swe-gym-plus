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
`--executor bwrap` and a configured `--backend local --model-path ...` (or DeepSeek
injected into the worker process environment). Never write credentials in Git.

## Contracts and limitations

- Atomic queue claim: SQLite `BEGIN IMMEDIATE`, PostgreSQL `FOR UPDATE SKIP LOCKED`.
- A fencing lease prevents a stale worker publishing over a recovered job.
- Heartbeat expiry marks jobs interrupted; paid or unacknowledged operations are
  never automatically replayed. Resume is an explicit operation.
- Checkpoints preserve tokens/tool calls/time, model messages, next step and
  command receipt. An acknowledged action is recovered without execution again.
- Unknown model response or tool effect refuses resume. Create a fresh run instead.
- Cancel is cooperative at runtime boundaries; Linux worker sends SIGTERM then
  SIGKILL to its child process group after 10 seconds. Windows is demo-only.
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
