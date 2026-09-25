from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    assert "--proc" not in command
    assert command[-3:] == ["/usr/bin/bash", "-c", "pwd"]
