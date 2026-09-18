"""Small API surface for submitting and inspecting runs.

The default SQLite index keeps the project runnable on a laptop. Deployment can
replace the repository with PostgreSQL without moving logs out of artifacts.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


def create_app(*, database: str | Path = "artifacts/codeagentbench.sqlite3", artifact_root: str | Path = "artifacts"):
    """Create the service lazily so the core harness has no web dependency."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("service requires `pip install codeagentbench[service]`") from exc
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(artifact_root)
    app = FastAPI(title="CodeAgentBench", version="0.1.0")

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL, summary_json TEXT NOT NULL DEFAULT '{}')")
        connection.commit()
        return connection

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs")
    def submit_run(payload: dict[str, Any]) -> dict[str, str]:
        task_id = str(payload.get("task_id", ""))
        if not task_id:
            raise HTTPException(status_code=422, detail="task_id is required")
        run_id = str(uuid.uuid4())
        with connect() as connection:
            connection.execute("INSERT INTO runs(run_id, task_id, status) VALUES (?, ?, ?)", (run_id, task_id, "queued"))
        return {"run_id": run_id, "status": "queued"}

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        with connect() as connection:
            rows = connection.execute("SELECT run_id, task_id, status, summary_json FROM runs ORDER BY rowid DESC").fetchall()
        return [{"run_id": row["run_id"], "task_id": row["task_id"], "status": row["status"], "summary": json.loads(row["summary_json"])} for row in rows]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        with connect() as connection:
            row = connection.execute("SELECT run_id, task_id, status, summary_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        run_dir = artifact_root / "runs" / run_id
        events = run_dir / "events.jsonl"
        return {"run_id": row["run_id"], "task_id": row["task_id"], "status": row["status"], "summary": json.loads(row["summary_json"]), "artifact_dir": str(run_dir), "events_file": str(events) if events.exists() else None}

    return app
