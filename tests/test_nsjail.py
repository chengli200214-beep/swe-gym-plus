from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeagentbench.models import ToolIntent
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.tasks.manifest import load_manifest

ROOT = Path(__file__).parents[1]


def test_unknown_backend_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "misspelled-isolation")
    with pytest.raises(ValueError, match="unknown executor"):
        selected_backend()
    with pytest.raises(ValueError, match="unknown executor"):
        WorkspaceManager(tmp_path).create(load_manifest(ROOT / "data/manifests/demo.json").tasks[0], "x")


def test_nsjail_layout_dissociates_git_objects(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    workspace = WorkspaceManager(tmp_path).create(load_manifest(ROOT / "data/manifests/demo.json").tasks[0], "x")
    assert workspace.path == tmp_path / "x/sandbox/workspace"
    assert not (workspace.path / ".git/objects/info/alternates").exists()


def test_nsjail_policy_denies_escape_operations():
    from codeagentbench.sandbox.nsjail import SECCOMP
    for denied in ("socket,", "socketpair", "ptrace", "process_vm", "setsid", "setpgid", "mknod", "io_uring", "mount,", "setuid"):
        assert denied not in SECCOMP
    assert "ERRNO(38) { clone3 }" in SECCOMP
    assert "clone_flags & 0xFE82A080" in SECCOMP


@pytest.mark.skipif(os.name != "posix", reason="Linux descriptor traversal")
def test_disk_accounting_does_not_follow_external_symlink(tmp_path):
    from codeagentbench.sandbox.bounded_process import writable_usage
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "large").write_bytes(b"x" * 10000)
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "link").symlink_to(outside, target_is_directory=True)
    writable_usage((inside,), bytes_limit=1000, files_limit=10)
    (inside / "large").write_bytes(b"x" * 1001)
    with pytest.raises(RuntimeError, match="budget"):
        writable_usage((inside,), bytes_limit=1000, files_limit=10)


@pytest.mark.skipif(os.getenv("CODEAGENTBENCH_TEST_NSJAIL") != "1", reason="explicit live Linux sandbox admission only")
def test_live_nsjail_negative_admission(tmp_path, monkeypatch):
    from scripts.check_nsjail import check_admission
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    check_admission(tmp_path)
