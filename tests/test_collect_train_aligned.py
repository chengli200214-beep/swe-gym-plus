"""Train-only collection preflight never needs a paid model call."""

import hashlib
import json
from pathlib import Path

import pytest

from codeagentbench.models import EvalSpec, TaskRecord
from codeagentbench.tasks.manifest import Manifest, load_manifest, save_manifest
from scripts.collect_train_aligned import prepare


TASK_ID = "getmoto__moto-5155"
TEST_FILE = "tests/test_ec2/test_route_tables.py"


def fixture_paths(tmp_path: Path):
    source = tmp_path / "source.json"
    split = tmp_path / "split.json"
    quality = tmp_path / "quality.json"
    cache = tmp_path / "cache"
    task = TaskRecord(TASK_ID, "getmoto/moto", "base-sha", "Fix route filter",
                      split="train", eval_spec=EvalSpec(gold_patch="secret gold"))
    save_manifest(Manifest("SWE-Gym", "fixed", (task,)), source)
    split.write_text(json.dumps({"train": [TASK_ID], "dev": []}), encoding="utf-8")
    quality.write_text(json.dumps({"task_id": TASK_ID, "admitted": True}), encoding="utf-8")
    key = hashlib.sha256(b"getmoto/moto\0base-sha").hexdigest()[:24]
    snapshot = cache / key
    (snapshot / ".git").mkdir(parents=True)
    path = snapshot / TEST_FILE
    path.parent.mkdir(parents=True)
    path.write_text("def test_public(): pass\n", encoding="utf-8")
    return source, split, quality, cache


def test_prepares_separate_blind_and_formal_manifests(tmp_path, monkeypatch):
    paths = fixture_paths(tmp_path)
    root = tmp_path / "campaign"
    # Windows CI does not grant ordinary users symlink creation privileges.
    linked = []
    def record_link(path, target, target_is_directory=False):
        linked.append((path, target, target_is_directory))
        path.mkdir()
    monkeypatch.setattr(Path, "symlink_to", record_link)
    blind, formal = prepare(*paths, root, TASK_ID, TEST_FILE)
    blind_task = load_manifest(blind).tasks[0]
    formal_task = load_manifest(formal).tasks[0]
    assert blind_task.eval_spec.gold_patch == ""
    assert formal_task.eval_spec.gold_patch == "secret gold"
    assert blind_task.agent_view().allowed_test_command == (
        "python -m pytest -q -x " + TEST_FILE)
    assert linked == [(root / ".repo_cache", paths[3].resolve(), True)]


def test_rejects_non_train_task_before_creating_campaign(tmp_path):
    paths = fixture_paths(tmp_path)
    paths[1].write_text(json.dumps({"train": [], "dev": [TASK_ID]}), encoding="utf-8")
    root = tmp_path / "campaign"
    with pytest.raises(ValueError, match="train split"):
        prepare(*paths, root, TASK_ID, TEST_FILE)
    assert not root.exists()


def test_rejects_missing_or_unsafe_visible_test(tmp_path):
    paths = fixture_paths(tmp_path)
    for visible in ("../tests/test_ec2/test_route_tables.py", "tests/private.py",
                    "tests/test_ec2/../../hidden.py"):
        root = tmp_path / visible.replace("/", "_")
        with pytest.raises((ValueError, FileNotFoundError)):
            prepare(*paths, root, TASK_ID, visible)
        assert not root.exists()
