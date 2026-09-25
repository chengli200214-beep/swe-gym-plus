import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest
pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from codeagentbench.service.app import create_app
from codeagentbench.service.repository import JobRepository
from codeagentbench.service.worker import Worker

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "data/manifests/demo.json"
LIMITS = {"max_steps": 4, "max_tool_calls": 4, "max_tokens": 32000, "max_seconds": 60}


@pytest.fixture(params=["sqlite", "postgres"] if os.getenv("TEST_POSTGRES_URL") else ["sqlite"])
def repository(tmp_path, request):
    database = os.environ["TEST_POSTGRES_URL"] if request.param == "postgres" else tmp_path / "queue.db"
    repo = JobRepository(database)
    if request.param == "postgres":
        # The dedicated test DB must never be shared with a deployed queue.
        with repo.engine.begin() as c:
            if c.execute(repo.jobs.select().limit(1)).first():
                pytest.fail("TEST_POSTGRES_URL must refer to an empty, disposable test DB")
    yield repo
    if request.param == "postgres":
        repo.jobs.drop(repo.engine)
    repo.engine.dispose()


def test_concurrent_workers_claim_each_job_once(repository):
    ids = {repository.submit("demo", LIMITS)["run_id"] for _ in range(12)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(lambda _: repository.claim(), range(16)))
    claimed = [row["run_id"] for row in jobs if row]
    assert len(claimed) == len(set(claimed)) == 12
    assert set(claimed) == ids


def test_stale_lease_cannot_publish_after_recovery(repository):
    from sqlalchemy import update
    row = repository.submit("demo", LIMITS)
    old = repository.claim()
    with repository.engine.begin() as c:
        c.execute(update(repository.jobs).where(repository.jobs.c.run_id == row["run_id"]).values(heartbeat=0))
    assert repository.reap_stale() == 1
    repository.resume(row["run_id"])
    new = repository.claim()
    assert old["lease"] != new["lease"]
    assert not repository.finish(row["run_id"], old["lease"], "completed", {})
    assert repository.finish(row["run_id"], new["lease"], "completed", {})


def test_api_validates_and_cancels_queued_job(tmp_path):
    app = create_app(database=tmp_path / "db", artifact_root=tmp_path / "artifacts", manifests=[MANIFEST])
    with TestClient(app) as client:
        task = client.get("/tasks").json()[0]["task_id"]
        assert client.get("/").status_code == 200
        assert client.post("/runs", json={"task_id": "unknown"}).status_code == 422
        assert client.post("/runs", json={"task_id": task, "max_steps": -1}).status_code == 422
        assert client.post("/runs", json={"task_id": task, "api_key": "private"}).status_code == 422
        row = client.post("/runs", json={"task_id": task}).json()
        assert "lease" not in row
        assert client.post(f'/runs/{row["run_id"]}/cancel').json()["status"] == "cancelled"
        assert client.post(f'/runs/{row["run_id"]}/resume').json()["status"] == "queued"
        assert client.get("/runs/not-a-valid-id").status_code == 404


def test_worker_runs_real_scripted_patch_and_independent_evaluation(tmp_path, repository):
    from codeagentbench.tasks.manifest import load_manifest
    task = load_manifest(MANIFEST).tasks[0]
    script = tmp_path / "actions.json"
    script.write_text(json.dumps([
        {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text().replace('left - right','left + right'))\""},
        {"done": True}]))
    root = tmp_path / "artifacts"
    worker = Worker(repository, root, [MANIFEST], script=script, executor="local")
    row = repository.submit(task.instance_id, LIMITS)
    assert worker.run_once()
    result = repository.get(row["run_id"])
    assert result["status"] == "completed"
    assert result["summary"]["evaluation"]["verdict"] == "passed"
    assert result["summary"]["budget"]["tool_calls"] == 1
    assert not worker.run_once()


def test_worker_honours_running_cancel(tmp_path, repository):
    from codeagentbench.tasks.manifest import load_manifest
    task = load_manifest(MANIFEST).tasks[0]
    script = tmp_path / "slow.json"
    script.write_text(json.dumps([{"command": "python -c \"import time; time.sleep(30)\""}]))
    worker = Worker(repository, tmp_path / "artifacts", [MANIFEST], script=script, executor="local")
    row = repository.submit(task.instance_id, LIMITS)
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    deadline = time.monotonic() + 15
    while repository.get(row["run_id"])["status"] == "queued" and time.monotonic() < deadline:
        time.sleep(.05)
    repository.cancel(row["run_id"])
    thread.join(20)
    assert not thread.is_alive()
    assert repository.get(row["run_id"])["status"] == "cancelled"
