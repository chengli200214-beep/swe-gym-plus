from __future__ import annotations

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
