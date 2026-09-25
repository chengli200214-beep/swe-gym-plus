from __future__ import annotations

import json
from pathlib import Path

from codeagentbench.cli import main
from codeagentbench.tasks.quality import ControlResult, QualityReport


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
