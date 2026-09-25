import json
import sys
from types import SimpleNamespace
import pytest

from scripts import collect_clean


@pytest.mark.parametrize("status", ["interrupted", "cancelled"])
def test_saved_interruption_summary_stops_next_paid_task(tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(collect_clean, "os", SimpleNamespace(name="posix", environ=collect_clean.os.environ))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    split, manifest = tmp_path / "split.json", tmp_path / "manifest.json"
    split.write_text(json.dumps({"train": ["task-one", "task-two"]}))
    manifest.write_text(json.dumps({"tasks": [{"instance_id": "task-one"}, {"instance_id": "task-two"}]}))
    root = tmp_path / "campaign"
    (root / "quality").mkdir(parents=True)
    (root / ".repo_cache").mkdir()
    for task in ["task-one", "task-two"]:
        (root / "quality" / (task + ".json")).write_text('{"admitted":true}')
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    monkeypatch.setattr(collect_clean.subprocess, "check_output", lambda *a, **kw: "test-commit")
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        run = root / "runs/task-one-clean-0"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({"status": status}))
        return SimpleNamespace(wait=lambda timeout=None: 0)

    monkeypatch.setattr(collect_clean.subprocess, "Popen", popen)
    monkeypatch.setattr(sys, "argv", ["collect_clean", "--root", str(root), "--split", str(split), "--manifests", str(manifest), "--tasks", "task-one", "task-two"])
    assert collect_clean.main() == 2
    assert len(calls) == 1
    assert json.loads((root / "results.json").read_text())[-1]["campaign_halted"] is True
