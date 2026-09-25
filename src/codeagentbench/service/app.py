"""Private task console backed by the transactional worker queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from codeagentbench.harness.recovery import ActionJournal
from codeagentbench.service.repository import JobRepository
from codeagentbench.tasks.manifest import load_manifest


def create_app(*, database="artifacts/codeagentbench.sqlite3", artifact_root="artifacts", manifests=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    root = Path(artifact_root).resolve()
    repository = JobRepository(database)
    task_paths = manifests or [Path("data/manifests/demo.json")]
    tasks = {t.instance_id: t for p in task_paths for t in load_manifest(p).tasks}
    app = FastAPI(title="CodeAgentBench", version="0.2.0")
    app.state.repository = repository

    def require_job(run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise HTTPException(status_code=404, detail="run not found")
        job = repository.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="run not found")
        return job

    def public_job(job):
        return json.loads(re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", json.dumps({k: v for k, v in job.items() if k != "lease"})))

    def read_artifact(run_id, name):
        require_job(run_id)
        path = root / "runs" / run_id / name
        if not path.exists():
            return ""
        if not path.resolve().is_relative_to(root):
            raise HTTPException(status_code=400, detail="artifact path escapes root")
        if name.endswith(".json") and path.stat().st_size > 1_000_000:
            raise HTTPException(status_code=413, detail="artifact too large for the console; inspect privately on the worker")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            content = stream.read(1_000_000)
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", content)

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        repository.list(1)
        return {"status": "ok", "database": repository.engine.dialect.name}

    @app.get("/tasks")
    def list_tasks():
        return [{"task_id": t.instance_id, "repo": t.repo, "issue": t.issue} for t in tasks.values()]

    @app.post("/runs", status_code=201)
    def submit_run(payload: dict[str, Any]):
        if set(payload) - {"task_id", "max_steps", "max_tool_calls", "max_tokens", "max_seconds"}:
            raise HTTPException(status_code=422, detail="unsupported run option")
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or task_id not in tasks:
            raise HTTPException(status_code=422, detail="unknown task_id")
        limits = {"max_steps": (12, 1, 64), "max_tool_calls": (16, 1, 128), "max_tokens": (32000, 1, 200000), "max_seconds": (600, 1, 3600)}
        config = {}
        for name, (default, lower, upper) in limits.items():
            value = payload.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
                raise HTTPException(status_code=422, detail=f"{name} must be an integer from {lower} to {upper}")
            config[name] = value
        return public_job(repository.submit(task_id, config))

    @app.get("/runs")
    def list_runs():
        return [public_job(row) for row in repository.list()]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        return public_job(require_job(run_id))

    @app.post("/runs/{run_id}/cancel")
    def cancel(run_id: str):
        require_job(run_id)
        return public_job(repository.cancel(run_id))

    @app.post("/runs/{run_id}/resume")
    def resume(run_id: str):
        job = require_job(run_id)
        checkpoint_text = read_artifact(run_id, "checkpoint.json")
        if checkpoint_text:
            state = json.loads(checkpoint_text)["state"]
            if state.get("pending_model") or ActionJournal(root / "runs" / run_id / "actions.jsonl").pending():
                raise HTTPException(status_code=409, detail="unacknowledged operation; create a new run instead of replaying")
            if state.get("status") == "completed":
                raise HTTPException(status_code=409, detail="agent already finished; evaluation recovery is separate")
        try:
            return public_job(repository.resume(job["run_id"]))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/artifacts/{kind}")
    def artifact(run_id: str, kind: str):
        names = {"events": "events.jsonl", "checkpoint": "checkpoint.json", "summary": "summary.json", "actions": "actions.jsonl"}
        if kind == "patch":
            content = read_artifact(run_id, "summary.json")
            return {"content": json.loads(content).get("diff", "") if content else ""}
        if kind not in names:
            raise HTTPException(status_code=404, detail="unknown artifact")
        return {"content": read_artifact(run_id, names[kind])}

    return app


def from_environment():
    return create_app(database=os.environ.get("CODEAGENTBENCH_DATABASE", "artifacts/codeagentbench.sqlite3"), artifact_root=os.environ.get("CODEAGENTBENCH_ARTIFACT_ROOT", "artifacts"), manifests=json.loads(os.environ.get("CODEAGENTBENCH_MANIFESTS", '["data/manifests/demo.json"]')))
