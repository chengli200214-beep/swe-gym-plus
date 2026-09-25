from __future__ import annotations

import json
from pathlib import Path

from codeagentbench.cli import main
from codeagentbench.tasks.quality import ControlResult, QualityReport, _run_tests


def test_quality_check_can_persist_control_report(tmp_path: Path, monkeypatch) -> None:
    report = QualityReport(
        "demo-calculator",
        True,
        (
            ControlResult("unfixed", 1, False, False, "failed", "", 0.1),
            ControlResult("gold", 0, True, True, "passed", "", 0.1),
        ),
    )
    monkeypatch.setattr("codeagentbench.tasks.quality.run_controls", lambda *args, **kwargs: report)
    manifest = Path(__file__).parents[1] / "data/manifests/demo.json"
    output = tmp_path / "quality" / "demo.json"
    assert main(["quality-check", str(manifest), "demo-calculator", "--output", str(output)]) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["admitted"] is True
    assert [item["name"] for item in saved["controls"]] == ["unfixed", "gold"]


def test_swe_gym_quality_requires_bwrap(monkeypatch, capsys) -> None:
    monkeypatch.delenv("CODEAGENTBENCH_EXECUTOR", raising=False)
    manifest = Path(__file__).parents[1] / "data/manifests/swegym-moto-candidates.json"
    assert main(["quality-check", str(manifest), "getmoto__moto-5699"]) == 2
    assert "require CODEAGENTBENCH_EXECUTOR=bwrap" in capsys.readouterr().err


def test_quality_bwrap_uses_isolated_executor(tmp_path: Path, monkeypatch) -> None:
    from codeagentbench.models import ToolReceipt

    seen = {}

    class FakeExecutor:
        def __init__(self, workspace, *, output_limit, backend, approved_cache_root):
            seen.update(workspace=workspace, output_limit=output_limit, backend=backend, approved_cache_root=approved_cache_root)

        def execute(self, intent):
            seen["intent"] = intent
            return ToolReceipt(intent.action_id, intent.command, 0, "1 passed", "", 0.01, False, "digest")

    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "bwrap")
    monkeypatch.setattr("codeagentbench.tasks.quality.BashExecutor", FakeExecutor)
    exit_code, stdout, stderr, _ = _run_tests(tmp_path, "python -m pytest -q", 10, cache_root=tmp_path / ".repo_cache")
    assert (exit_code, stdout, stderr) == (0, "1 passed", "")
    assert seen["backend"] == "bwrap"
    assert seen["approved_cache_root"] == tmp_path / ".repo_cache"
    assert seen["intent"].cwd == str(tmp_path)
