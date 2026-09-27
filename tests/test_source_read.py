"""Actual source observations, stale versions and recovery, not model scores."""
import hashlib
import json

import pytest

from codeagentbench.adapters.action import parse_action
from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.harness.context_history import recent_history
from codeagentbench.harness.source_evidence import grounded_command, observe_source
from codeagentbench.models import RunConfig, TaskRecord, ToolIntent
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import Workspace
from codeagentbench.storage.artifacts import ArtifactStore
from test_text_edit import edit, make_run, read


def execute(path, action, observations):
    command = grounded_command(parse_action(json.dumps(action)), observations)
    receipt = BashExecutor(path).execute(ToolIntent("test", command, str(path)))
    observe_source(observations, parse_action(json.dumps(action)), vars(receipt))
    return receipt


@pytest.mark.parametrize("payload", [
    {"path": "../value.py", "start_line": 1, "end_line": 2},
    {"path": ".GIT/config", "start_line": 1, "end_line": 2},
    {"path": "value.py", "start_line": True, "end_line": 2},
    {"path": "value.py", "start_line": 0, "end_line": 2},
    {"path": "value.py", "start_line": 2, "end_line": 1},
    {"path": "value.py", "start_line": 1, "end_line": 81},
    {"path": "value.py", "start_line": 1},
])
def test_read_boundary_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        parse_action(json.dumps({"read": payload}))


def test_read_cannot_mix_with_another_action():
    for extra in ({"command": "echo x"}, {"done": True}, edit(), {"unknown": True}):
        with pytest.raises(ValueError):
            parse_action(json.dumps(read() | extra))


def test_actual_read_is_bounded_whole_lines_versioned_and_idempotent_in_context(tmp_path):
    raw = ("value = '中文 quoted \\\" text'\n" * 100).encode()
    (tmp_path / "value.py").write_bytes(raw)
    observations = []
    receipt = execute(tmp_path, read(), observations)
    assert receipt.exit_code == 0, receipt.stderr
    data = json.loads(receipt.stdout)
    assert data["sha256"] == hashlib.sha256(raw).hexdigest()
    assert raw.decode().startswith(data["text"]) and data["text"].endswith("\n")
    assert len(receipt.stdout) < 4000 and data["end_line"] <= 80
    messages = [{"role": "system", "content": "protocol"}, {"role": "user", "content": "issue"},
                {"role": "assistant", "content": json.dumps(read())},
                {"role": "user", "content": "Tool result:\n" + json.dumps({k: vars(receipt)[k] for k in ("exit_code", "stdout", "stderr", "timed_out")})}]
    view = recent_history(messages)
    assert json.loads(json.loads(view[-1]["content"].split("\n", 1)[1])["stdout"]) == data
    assert recent_history(view) == view


def test_guessed_or_unread_before_never_executes(tmp_path):
    observations = []
    with pytest.raises(ValueError, match="successful read"):
        grounded_command(parse_action(json.dumps(edit())), observations)
    (tmp_path / "value.py").write_bytes(b"value = 1\nother = 3\n")
    execute(tmp_path, {"read": {"path": "value.py", "start_line": 2, "end_line": 2}}, observations)
    with pytest.raises(ValueError, match="successful read"):
        grounded_command(parse_action(json.dumps(edit())), observations)


def test_stale_file_fails_closed_even_when_before_still_matches(tmp_path):
    path = tmp_path / "value.py"
    path.write_bytes(b"value = 1\nother = 3\n")
    observations = []
    execute(tmp_path, read(), observations)
    path.write_bytes(b"value = 1\nother = 4\n")
    receipt = execute(tmp_path, edit(), observations)
    assert receipt.exit_code != 0 and "changed since read" in receipt.stderr
    assert path.read_bytes() == b"value = 1\nother = 4\n"
    execute(tmp_path, read(), observations)
    receipt = execute(tmp_path, edit(), observations)
    assert receipt.exit_code == 0 and observations == []
    assert path.read_bytes() == b"value = 2\nother = 4\n"


def test_oversized_single_line_returns_no_partial_evidence(tmp_path):
    (tmp_path / "value.py").write_bytes(b"x" * 4000 + b"\n")
    observations = []
    receipt = execute(tmp_path, read(), observations)
    assert receipt.exit_code != 0 and "no partial line" in receipt.stderr
    assert observations == []


def test_runtime_corrects_unobserved_edit_with_real_read_not_host_supplied_answer(tmp_path):
    result, workspace, store, _ = make_run(tmp_path, [edit(), read(), edit(), {"done": True}])
    assert result.diff and result.state.protocol_corrections == 1
    events = [json.loads(s) for s in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    tools = [e for e in events if e["type"] == "tool"]
    assert len(tools) == 2 and json.loads(tools[0]["intent"]["action_json"]).get("read")
    assert "successful read" in str(result.state.messages)
    assert result.state.source_observations == []


def test_read_receipt_survives_crash_without_reexecuting_or_losing_evidence(tmp_path):
    def crash(where):
        if where == "after_receipt":
            raise RuntimeError("read receipt recorded")

    with pytest.raises(RuntimeError, match="read receipt"):
        make_run(tmp_path, [read()], injector=crash)
    task = TaskRecord("typed-edit-test", str(tmp_path / "source"), "local", "Change value to 2")
    workspace = Workspace("edit-test", tmp_path / "workspaces/edit-test/workspace", "local")
    store = ArtifactStore(tmp_path / "artifacts")
    result = AgentRuntime(store).run(task, workspace, ScriptedModel([edit(), {"done": True}]),
                                    RunConfig(max_steps=8, max_seconds=60), run_id="edit-test", resume=True)
    assert result.status == "completed" and "value = 2" in result.diff
    events = [json.loads(s) for s in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["type"] == "tool"]) == 2
