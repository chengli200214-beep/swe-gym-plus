import json
from types import SimpleNamespace

import pytest

from scripts.autodl_edit_proposal import validate_proposal


def test_complete_proposal_accepts_only_the_known_file():
    proposal = {"path": "moto/athena/models.py", "before": "value = old", "after": "value = new"}
    assert validate_proposal(json.dumps(proposal)) == proposal


@pytest.mark.parametrize("proposal", [
    {"path": "/root/.ssh/config", "before": "a", "after": "b"},
    {"path": "moto/athena/models.py", "before": "a", "after": "a"},
    {"path": "moto/athena/models.py", "before": "a", "after": "b", "command": "echo no"},
    {"path": "moto/athena/models.py", "before": [], "after": "b"},
])
def test_invalid_proposal_is_rejected_before_execution(proposal):
    with pytest.raises(ValueError):
        validate_proposal(json.dumps(proposal))


def test_truncated_proposal_is_never_repaired():
    with pytest.raises(ValueError):
        validate_proposal('{"path":"moto/athena/models.py","before":"a","after":"b')


@pytest.mark.parametrize("issue_view", ["full", "before-traceback", "repro-and-exception"])
def test_correction_retains_real_failure_and_stops_after_two_attempts(monkeypatch, tmp_path, issue_view):
    import scripts.autodl_edit_proposal as probe
    from codeagentbench.adapters.model import ModelResponse
    from codeagentbench.models import ToolReceipt

    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"admitted": True, "task_id": "getmoto__moto-6212"}))
    calls = []
    proposal = {"path": "moto/athena/models.py", "before": "missing", "after": "replacement"}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, messages, **kwargs):
            assert self.prompt_policy == "native"
            calls.append([dict(message) for message in messages])
            return ModelResponse(json.dumps(proposal))

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, intent):
            is_read = intent.action_id == "public-source"
            return ToolReceipt(intent.action_id, intent.command, 0 if is_read else 1,
                               "actual source" if is_read else "", "" if is_read else "before did not match",
                               0, False, "same")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(probe, "selected_backend", lambda: "nsjail")
    monkeypatch.setattr(probe, "load_manifest", lambda _: SimpleNamespace(tasks=[
        SimpleNamespace(instance_id="getmoto__moto-6212",
                        issue="public issue\nTraceback:\ndependency code\nE ValueError: example failure")]))
    monkeypatch.setattr(probe, "WorkspaceManager", lambda _: SimpleNamespace(
        create=lambda *args: SimpleNamespace(path=tmp_path, diff=lambda: "")))
    monkeypatch.setattr(probe, "BashExecutor", FakeExecutor)
    monkeypatch.setattr(probe, "LocalHFModel", FakeModel)
    monkeypatch.setattr("sys.argv", ["probe", "--manifest", str(tmp_path / "manifest.json"),
        "--artifact-root", str(tmp_path), "--admission", str(admission), "--model", str(tmp_path / "model"),
        "--run-id", "correction", "--max-attempts", "2", "--issue-view", issue_view])
    probe.main()
    assert len(calls) == 2
    assert len(calls[1]) == 4
    assert calls[1][2] == {"role": "assistant", "content": json.dumps(proposal)}
    assert "before did not match" in calls[1][3]["content"]
    assert "actual source" in calls[1][1]["content"]
    assert ("dependency code" in calls[0][1]["content"]) == (issue_view == "full")
    assert ("ValueError: example failure" in calls[0][1]["content"]) == (issue_view != "before-traceback")
    inputs = json.loads((tmp_path / "structured-probes/correction/input.json").read_text())
    assert "dependency code" in inputs["issue_original"]
    report = json.loads((tmp_path / "structured-probes/correction/report.json").read_text())
    assert len(report["attempts"]) == 2
    assert report["training_eligible"] is False and report["evaluation"] is None
    assert report["status"] == "rejected" and not report["diff_present"]
