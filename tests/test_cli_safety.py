from __future__ import annotations

from pathlib import Path

from codeagentbench.cli import main


MANIFEST = Path(__file__).parents[1] / "data" / "manifests" / "demo.json"


def test_deepseek_run_requires_separate_evaluation(capsys) -> None:
    code = main(["run", str(MANIFEST), "demo-calculator"])
    assert code == 2
    assert "--skip-evaluation" in capsys.readouterr().err


def test_evaluation_rejects_api_key_in_environment(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    code = main(["evaluate-run", str(MANIFEST), "demo-run", "--repo-root", str(tmp_path)])
    assert code == 2
    assert "without DEEPSEEK_API_KEY" in capsys.readouterr().err
