import json
from types import SimpleNamespace

import pytest

from codeagentbench.models import TaskRecord, EvaluationResult, Verdict
from codeagentbench.verification.evaluator import Evaluator
from scripts.autodl_dev_gate import freeze, load_frozen
from scripts import autodl_dev_gate as gate


def source(path, splits=("dev", "dev", "dev")):
    tasks = [TaskRecord("task-" + str(i), "public/repo", "base-" + str(i), "fix", split=split)
             for i, split in enumerate(splits)]
    path.write_text(json.dumps({"dataset": "SWE-Gym", "revision": "pinned", "tasks": [t.to_dict() for t in tasks]}))
    return [t.instance_id for t in tasks]


def test_gate_freezes_three_tasks_without_outcome_replacement(tmp_path):
    manifest = tmp_path / "source.json"
    ids = source(manifest)
    root = tmp_path / "gate"
    identity = freeze(manifest, root, ids)
    assert identity["tasks"] == ids and identity["no_outcome_replacement"]
    assert [t.instance_id for t in load_frozen(root)] == ids
    with pytest.raises(FileExistsError):
        freeze(manifest, root, ids)
    changed = json.loads((root / "manifest.json").read_text())
    changed["tasks"][0]["issue"] = "different"
    (root / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        load_frozen(root)


def test_gate_does_not_relabel_eval_or_reuse_training_tasks(tmp_path):
    manifest = tmp_path / "source.json"
    ids = source(manifest, ("dev", "eval", "train"))
    with pytest.raises(ValueError, match="silently"):
        freeze(manifest, tmp_path / "gate", ids)
    with pytest.raises(ValueError, match="three distinct"):
        freeze(manifest, tmp_path / "gate", [ids[0]] * 3)


def test_explicit_evaluation_cache_reuses_admission_snapshots(tmp_path):
    cache = tmp_path / "immutable-cache"
    evaluator = Evaluator(tmp_path / "evaluations", cache_root=cache)
    assert evaluator.cache_root == cache.resolve()


def test_development_retest_keeps_original_results_and_frozen_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "source.json"
    ids = source(manifest)
    root = tmp_path / "gate"
    freeze(manifest, root, ids)
    original = (root / "manifest.json").read_bytes()
    (root / "run-report.json").write_text("original-negative-result")
    (root / "quality").mkdir()
    for task_id in ids:
        (root / "quality" / (task_id + ".json")).write_text(json.dumps({"task_id": task_id, "admitted": True}))
    monkeypatch.setattr(gate, "LocalHFModel", lambda *a, **k: object())
    monkeypatch.setattr(gate, "WorkspaceManager", lambda *a, **k: SimpleNamespace(create=lambda *a: object()))
    result = SimpleNamespace(diff="", status="failed", failure_reason="fixture", steps=1)
    monkeypatch.setattr(gate, "AgentRuntime", lambda *a: SimpleNamespace(run=lambda *a, **k: result))
    evaluation = EvaluationResult(Verdict.FAILED, 1, False, False, reason="fixture")
    monkeypatch.setattr(gate, "Evaluator", lambda *a, **k: SimpleNamespace(evaluate=lambda *a, **k: evaluation))
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="fixture-commit\n"))
    monkeypatch.setattr(gate, "digest", lambda p: "fixture-hash")
    monkeypatch.setattr(gate, "load_frozen", lambda p: [TaskRecord(i, "repo", "base", "issue") for i in ids])
    monkeypatch.setattr(gate.importlib.metadata, "version", lambda name: "fixture-version")
    trial = tmp_path / "retest"
    assert gate.run(root, tmp_path / "model", output_root=trial)["passed"] == 0
    report = json.loads((trial / "run-report.json").read_text())
    assert report["development_retest"] is True
    assert [t["task_id"] for t in report["tasks"]] == ids
    assert (root / "manifest.json").read_bytes() == original
    assert (root / "run-report.json").read_text() == "original-negative-result"
    with pytest.raises(ValueError, match="preserve"):
        gate.run(root, tmp_path / "model", output_root=trial)
