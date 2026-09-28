"""Visible-test allowlist syntax checks; no real training examples are created."""
from dataclasses import replace

import pytest

from codeagentbench.models import TaskRecord
from codeagentbench.tasks.manifest import Manifest, save_manifest
from scripts.prepare_aligned_sft import visible_test_commands_from_manifest


def test_visible_test_manifest_accepts_bounded_train_public_path(tmp_path):
    task = TaskRecord("train-1", "repo", "abc", "issue", split="train",
                      metadata={"agent_test_command": "python -m pytest -q -x tests/test_public.py"})
    path = tmp_path / "manifest.json"
    save_manifest(Manifest("fixture", "r1", (task,)), path)
    assert visible_test_commands_from_manifest(path) == {
        "train-1": "python -m pytest -q -x tests/test_public.py"}


@pytest.mark.parametrize("command", [
    "python -m pytest -q -x tests/../private.py",
    "python -m pytest -q -x /tmp/private.py",
    "python -m pytest -q -x tests/test_public.py::hidden_case",
    "python -m pytest -q -x tests/test_public.py; echo leak",
    "python -m pytest -q tests/test_public.py",
])
def test_visible_test_manifest_rejects_nonpublic_or_unbounded_command(tmp_path, command):
    task = TaskRecord("train-1", "repo", "abc", "issue", split="train",
                      metadata={"agent_test_command": command})
    path = tmp_path / "manifest.json"
    save_manifest(Manifest("fixture", "r1", (task,)), path)
    with pytest.raises(ValueError, match="visible-test command"):
        visible_test_commands_from_manifest(path)


def test_visible_test_manifest_rejects_development_task(tmp_path):
    task = TaskRecord("dev-1", "repo", "abc", "issue", split="train",
                      metadata={"agent_test_command": "python -m pytest -q -x tests/test_public.py"})
    path = tmp_path / "manifest.json"
    save_manifest(Manifest("fixture", "r1", (replace(task, split="dev"),)), path)
    with pytest.raises(ValueError, match="train tasks"):
        visible_test_commands_from_manifest(path)
