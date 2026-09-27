"""Private API/worker acceptance on AutoDL; trusted demo, not model accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from fastapi.testclient import TestClient

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.models import RunConfig
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.service.app import create_app
from codeagentbench.service.worker import Worker
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest


def wait_until(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("acceptance condition did not become true")
        time.sleep(0.1)


def guest_processes(uid):
    rows = subprocess.run(["ps", "-eo", "uid,pid,state"], capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    return [int(p) for u, p, state in (r.split() for r in rows) if int(u) == uid and state != "Z"]


def cancel_running(client, repository, worker, limits, artifacts, *, phase):
    job = client.post("/runs", json=limits).json()
    if phase == "evaluation":
        prepared = artifacts / "evaluations" / (job["run_id"] + "-eval-candidate-0") / "nsjail-prepared.json"
    else:
        prepared = artifacts / job["run_id"] / "nsjail-prepared.json"
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    try:
        wait_until(prepared.exists)
        uid = json.loads(prepared.read_text())["uid"]
        wait_until(lambda: bool(guest_processes(uid)))
        requested_at = time.monotonic()
        client.post('/runs/' + job["run_id"] + '/cancel')
        thread.join(5)
        assert not thread.is_alive(), "worker did not stop promptly after cancellation"
        assert repository.get(job["run_id"])["status"] == "cancelled"
        wait_until(lambda: not guest_processes(uid), seconds=3)
        if phase == "evaluation":
            lines = artifacts / "runs" / job["run_id"] / "events.jsonl"
            events = [json.loads(s) for s in lines.read_text().splitlines()]
            evaluation = next(e for e in reversed(events) if e.get("type") == "evaluation")
            assert evaluation["verdict"] == "blocked"
            assert evaluation["reason"] == "formal evaluation cancelled"
        return {"seconds": round(time.monotonic() - requested_at, 3), "guest_processes_remaining": 0}
    finally:
        if thread.is_alive():
            repository.cancel(job["run_id"])
            thread.join(20)


def run(root: Path):
    if selected_backend() != "nsjail" or os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("acceptance requires credential-free NsJail; no fallback")
    root.mkdir(parents=True, exist_ok=False)
    manifest = Path("data/manifests/demo.json").resolve()
    original = load_manifest(manifest)
    task = original.tasks[0]
    # A second trusted fixture exercises cancellation during formal evaluation,
    # independently of AgentRuntime. It is never a model training/eval sample.
    eval_task = replace(task, instance_id=task.instance_id + "-slow-evaluation",
        eval_spec=replace(task.eval_spec, test_command="python -c 'import time; time.sleep(30)'"))
    eval_manifest = root / "trusted-evaluation-manifest.json"
    eval_manifest.write_text(json.dumps({"dataset": original.dataset, "revision": original.revision,
                                       "tasks": [eval_task.to_dict()]}))
    artifacts = root / "artifacts"
    app = create_app(database=root / "queue.db", artifact_root=artifacts, manifests=[manifest, eval_manifest])
    repository = app.state.repository
    limits = {"task_id": task.instance_id, "max_steps": 8, "max_tool_calls": 8,
              "max_tokens": 32000, "max_seconds": 60}
    actions = [{"read": {"path": "src/calculator.py", "start_line": 1, "end_line": 80}},
               {"edit": {"path": "src/calculator.py", "before": "left - right", "after": "left + right"}},
               {"command": "python -m pytest -q"}, {"done": True}]
    script = root / "trusted-demo-actions.json"
    script.write_text(json.dumps(actions))
    worker = Worker(repository, artifacts, [manifest], script=script, executor="nsjail")
    report = {"kind": "same-machine-service-acceptance", "scripted_fixture": True,
              "model_accuracy_claim": False, "public_port_opened": False, "checks": {},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path("src/codeagentbench") / name for name in (
                      "runtime.py", "cli.py", "harness/cancellation.py", "sandbox/executor.py",
                      "sandbox/nsjail.py", "sandbox/bounded_process.py", "verification/evaluator.py",
                      "service/worker.py")] + [Path(__file__)]},
              "packages": {name: importlib.metadata.version(name) for name in ("fastapi", "sqlalchemy", "httpx")}}

    def passed(name, evidence):
        report["checks"][name] = evidence
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"check": name, "evidence": evidence}), flush=True)

    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.get("/").status_code == 200
            assert client.get("/tasks").json()[0]["task_id"] == task.instance_id
            assert client.post("/runs", json=limits | {"command": "untrusted"}).status_code == 422
            queued = client.post("/runs", json=limits).json()
            assert client.post('/runs/' + queued["run_id"] + '/cancel').json()["status"] == "cancelled"
            passed("queue-and-input-boundary", {"queued_cancelled": True, "command_input_rejected": True})

            job = client.post("/runs", json=limits).json()
            assert worker.run_once()
            finished = client.get('/runs/' + job["run_id"]).json()
            assert finished["status"] == "completed", finished["summary"]
            assert finished["summary"]["evaluation"]["verdict"] == "passed"
            for kind in ("events", "patch", "checkpoint", "summary"):
                assert client.get('/runs/' + job["run_id"] + '/artifacts/' + kind).json()["content"]
            assert client.post('/runs/' + job["run_id"] + '/resume').status_code == 409
            passed("execute-and-display", {"independent_fixture_evaluation": "passed", "four_artifact_endpoints": True})

            slow = root / "trusted-sleep-actions.json"
            slow.write_text(json.dumps([{"command": "python -c 'import time; time.sleep(30)'"}]))
            cancelling_worker = Worker(repository, artifacts, [manifest], script=slow, executor="nsjail")
            passed("running-cancellation", cancel_running(client, repository, cancelling_worker, limits, artifacts, phase="agent"))
            evaluation_worker = Worker(repository, artifacts, [eval_manifest], script=script, executor="nsjail")
            passed("evaluation-cancellation", cancel_running(client, repository, evaluation_worker,
                limits | {"task_id": eval_task.instance_id}, artifacts, phase="evaluation"))

            recovering = client.post("/runs", json=limits).json()
            claimed = repository.claim()
            assert claimed["run_id"] == recovering["run_id"]
            workspace = WorkspaceManager(artifacts).create(task, recovering["run_id"])
            config = RunConfig(model="scripted", temperature=0.2, **{k: v for k, v in limits.items() if k != "task_id"})
            acknowledged = 0
            def crash(phase):
                nonlocal acknowledged
                if phase == "after_receipt":
                    acknowledged += 1
                    if acknowledged == 2:
                        raise RuntimeError("controlled acknowledged-edit crash")
            try:
                AgentRuntime(ArtifactStore(artifacts)).run(task, workspace, ScriptedModel(actions), config,
                    run_id=recovering["run_id"], failure_injector=crash)
            except RuntimeError as exc:
                assert str(exc) == "controlled acknowledged-edit crash"
            else:
                raise AssertionError("expected controlled crash")
            assert repository.finish(recovering["run_id"], claimed["lease"], "interrupted", {"reason": "controlled crash"})
            assert client.post('/runs/' + recovering["run_id"] + '/resume').json()["status"] == "queued"
            tail = root / "trusted-resume-actions.json"
            tail.write_text(json.dumps(actions[2:]))
            recovery_worker = Worker(repository, artifacts, [manifest], script=tail, executor="nsjail")
            assert recovery_worker.run_once()
            row = repository.get(recovering["run_id"])
            assert row["status"] == "completed", row["summary"]
            run_dir = artifacts / "runs" / recovering["run_id"]
            journal = [json.loads(s) for s in (run_dir / "actions.jsonl").read_text().splitlines()]
            receipts = Counter(e["action_id"] for e in journal if e["type"] == "receipt")
            assert len(receipts) == 3 and all(n == 1 for n in receipts.values())
            intents = [json.loads(e["action_json"]) for e in journal if e["type"] == "intent"]
            assert sum("edit" in action for action in intents) == 1
            events = [json.loads(s) for s in (run_dir / "events.jsonl").read_text().splitlines()]
            assert any(e.get("type") == "tool" and e.get("recovered") for e in events)
            passed("acknowledged-recovery", {"evaluation": "passed", "receipt_ids_executed_once": True,
                "acknowledged_edit_executions": 1, "resume_count": row["resume_count"]})
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:500]}
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2))
        repository.engine.dispose()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root)), flush=True)
