from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from codeagentbench.models import ToolIntent
from codeagentbench.sandbox.executor import BashExecutor


def test_native_bash_skips_login_profile(tmp_path, monkeypatch) -> None:
    executor = BashExecutor(tmp_path)
    executor.bash = "bash"
    monkeypatch.setattr(executor, "_is_wsl_bash", lambda: False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("codeagentbench.sandbox.executor.subprocess.run", fake_run)
    receipt = executor.execute(ToolIntent("a1", "printf ok", str(tmp_path)))

    assert calls[0][0] == ["bash", "-c", "printf ok"]
    assert calls[0][1]["shell"] is False
    assert receipt.stdout == "ok"


def test_bwrap_exposes_only_checkout_and_clears_credentials(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    executor = object.__new__(BashExecutor)
    executor.workspace = Path(tmp_path).resolve()
    executor.bwrap = "/usr/bin/bwrap"
    command = executor._bwrap_command("pwd", executor.workspace)
    assert command[0] == "/usr/bin/bwrap"
    assert ["--bind", str(tmp_path.resolve()), "/workspace"] == command[command.index("--bind"):command.index("--bind") + 3]
    assert ["--ro-bind", str((tmp_path / ".git").resolve()), "/workspace/.git"] in [command[i:i + 3] for i in range(len(command) - 2)]
    assert "--unshare-net" in command
    assert "--cap-drop" in command
    assert "--clearenv" in command
    assert ["--symlink", str(Path("/usr/bin/python3").resolve()), "/toolbin/python"] in [command[i:i + 3] for i in range(len(command) - 2)]
    assert ["--setenv", "PATH", "/toolbin:/usr/bin:/bin"] in [command[i:i + 3] for i in range(len(command) - 2)]
    assert "--proc" not in command
    assert command[-3:] == ["/usr/bin/bash", "-c", "pwd"]


def test_bwrap_binds_only_approved_git_alternate(tmp_path) -> None:
    workspace = tmp_path / "artifacts" / "run" / "workspace"
    info = workspace / ".git" / "objects" / "info"
    info.mkdir(parents=True)
    objects = tmp_path / "artifacts" / ".repo_cache" / "snapshot" / ".git" / "objects"
    objects.mkdir(parents=True)
    (info / "alternates").write_text(str(objects), encoding="utf-8")
    executor = object.__new__(BashExecutor)
    executor.workspace = workspace.resolve()
    executor.bwrap = "/usr/bin/bwrap"

    command = executor._bwrap_command("git status --short", executor.workspace)

    assert ["--ro-bind", str(objects.resolve()), str(objects.resolve())] in [command[i:i + 3] for i in range(len(command) - 2)]
    assert str((tmp_path / "artifacts" / ".repo_cache").resolve()) not in command


def test_bwrap_evaluation_uses_shared_approved_git_cache(tmp_path) -> None:
    workspace = tmp_path / "artifacts" / "evaluations" / "run-eval" / "workspace"
    info = workspace / ".git" / "objects" / "info"
    info.mkdir(parents=True)
    objects = tmp_path / "artifacts" / ".repo_cache" / "snapshot" / ".git" / "objects"
    objects.mkdir(parents=True)
    (info / "alternates").write_text(str(objects), encoding="utf-8")
    executor = object.__new__(BashExecutor)
    executor.workspace = workspace.resolve()
    executor.bwrap = "/usr/bin/bwrap"

    command = executor._bwrap_command("git status --short", executor.workspace)

    assert ["--ro-bind", str(objects.resolve()), str(objects.resolve())] in [command[i:i + 3] for i in range(len(command) - 2)]


def test_bwrap_quality_uses_explicit_approved_git_cache(tmp_path) -> None:
    workspace = tmp_path / "cab-quality-temp" / "unfixed" / "workspace"
    info = workspace / ".git" / "objects" / "info"
    info.mkdir(parents=True)
    cache_root = tmp_path / "artifacts" / ".repo_cache"
    objects = cache_root / "snapshot" / ".git" / "objects"
    objects.mkdir(parents=True)
    (info / "alternates").write_text(str(objects), encoding="utf-8")
    executor = object.__new__(BashExecutor)
    executor.workspace = workspace.resolve()
    executor.bwrap = "/usr/bin/bwrap"
    executor.approved_cache_root = cache_root.resolve()

    command = executor._bwrap_command("git status --short", executor.workspace)

    assert ["--ro-bind", str(objects.resolve()), str(objects.resolve())] in [command[i:i + 3] for i in range(len(command) - 2)]


def test_bwrap_rejects_git_alternate_outside_cache(tmp_path) -> None:
    workspace = tmp_path / "artifacts" / "run" / "workspace"
    info = workspace / ".git" / "objects" / "info"
    info.mkdir(parents=True)
    (info / "alternates").write_text(str(tmp_path), encoding="utf-8")
    executor = object.__new__(BashExecutor)
    executor.workspace = workspace.resolve()
    executor.bwrap = "/usr/bin/bwrap"

    with pytest.raises(RuntimeError, match="outside the approved"):
        executor._bwrap_command("git status --short", executor.workspace)
