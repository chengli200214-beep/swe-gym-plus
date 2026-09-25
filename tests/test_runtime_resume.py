import json
from pathlib import Path

import pytest

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.models import RunConfig, TaskRecord
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore


@pytest.mark.parametrize("crash_phase", ["after_receipt", "after_checkpoint"])
def test_resume_reuses_receipt_without_replaying_effect(tmp_path, crash_phase):
    source = tmp_path / "source"
    source.mkdir()
    (source / "counter.txt").write_text("0")
    task = TaskRecord("test", str(source), "local", "increment once")
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "resume-test")
    store = ArtifactStore(tmp_path / "artifacts")
    runtime = AgentRuntime(store)
    config = RunConfig(max_steps=3, max_seconds=60)
    command = "python -c \"from pathlib import Path; p=Path('counter.txt'); p.write_text(str(int(p.read_text())+1))\""

    def crash(phase):
        if phase == crash_phase:
            raise RuntimeError("worker crash")

    with pytest.raises(RuntimeError, match="worker crash"):
        runtime.run(task, workspace, ScriptedModel([{"command": command}]), config, run_id="resume-test", failure_injector=crash)
    result = runtime.run(task, workspace, ScriptedModel([{"done": True}]), config, run_id="resume-test", resume=True)
    assert result.status == "completed"
    assert (workspace.path / "counter.txt").read_text() == "1"
    assert result.state.spent_tokens == 64
    assert result.state.next_step == 2
    summary = json.loads((store.run_dir("resume-test") / "summary.json").read_text())
    assert summary["budget"]["tool_calls"] == 1
    events = [json.loads(line) for line in (store.run_dir("resume-test") / "events.jsonl").read_text().splitlines()]
    assert sum(e["type"] == "tool" for e in events) == 1


def test_resume_refuses_unknown_model_outcome(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("unchanged")
    task = TaskRecord("test", str(source), "local", "fix")
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "unknown-model")
    runtime = AgentRuntime(ArtifactStore(tmp_path / "artifacts"))

    class BrokenModel:
        def complete(self, *args, **kwargs):
            raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError, match="connection lost"):
        runtime.run(task, workspace, BrokenModel(), RunConfig(), run_id="unknown-model")
    with pytest.raises(RuntimeError, match="model request outcome is unknown"):
        runtime.run(task, workspace, ScriptedModel([]), RunConfig(), run_id="unknown-model", resume=True)


def test_artifact_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="invalid run id"):
        ArtifactStore(tmp_path).run_dir("../../outside")
