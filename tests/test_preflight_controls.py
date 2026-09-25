import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import preflight_controls


def test_preflight_is_credential_free_and_receipts_are_reusable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-not-to-copy")
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "split.json").write_text(json.dumps({"train": ["train"], "dev": ["dev"], "eval": ["eval"]}))
    (pool / "manifest.json").write_text('{"tasks":[]}')
    monkeypatch.setattr(preflight_controls.subprocess, "check_output", lambda *a, **kw: "commit")
    calls = []

    def popen(command, **kwargs):
        assert "DEEPSEEK_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["CODEAGENTBENCH_EXECUTOR"] == "bwrap"
        assert "train" not in command
        calls.append(command)
        Path(command[-1]).write_text('{"admitted":true}')
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr(preflight_controls.subprocess, "Popen", popen)
    root = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["preflight", "--pool", str(pool), "--root", str(root)])
    preflight_controls.main()
    preflight_controls.main()
    assert len(calls) == 2
    assert len(list((root / "logs").glob("*.receipt.json"))) == 2
