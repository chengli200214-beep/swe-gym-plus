"""Typed editing contracts; scripted tests do not prove model repair ability."""
import json
import os

import pytest

from codeagentbench.adapters.action import parse_action
from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.adapters.text_edit import TextEdit, edit_command, validate_edit
from codeagentbench.harness.context_history import recent_history
from codeagentbench.models import RunConfig, TaskRecord, ToolIntent
from codeagentbench.runtime import AgentRuntime, _is_visible_test_run
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import Workspace, WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.training.action_context import compact_context
from codeagentbench.training.readiness import audit_readiness
from scripts.prepare_aligned_sft import examples


def edit(before="value = 1", after="value = 2", path="value.py"):
    return {"edit": {"path": path, "before": before, "after": after}, "done": False}


def read():
    return {"read": {"path": "value.py", "start_line": 1, "end_line": 80}}


@pytest.mark.parametrize("path", ["../x.py", "/tmp/x.py", "C:/x.py", "x\\y.py", ".git/config", "a/.git/x", "./x", "a//x", "a/../x", "x\n.py"])
def test_invalid_paths_rejected(path):
    with pytest.raises(ValueError, match="path"):
        parse_action(json.dumps(edit(path=path)))


@pytest.mark.parametrize("extra", [{"command": "echo nope"}, {"done": True}, {"other": 1}])
def test_ambiguous_edit_envelope_rejected(extra):
    with pytest.raises(ValueError):
        parse_action(json.dumps(edit() | extra))


def test_typed_edit_is_executable_without_a_model_shell_program():
    action = parse_action(json.dumps(edit()))
    assert action.executable and not action.command and not action.done
    assert action.edit == TextEdit("value.py", "value = 1", "value = 2")
    assert parse_action(json.dumps(action.to_dict())) == action
    with pytest.raises(ValueError):
        validate_edit({"path": "x", "before": "a", "after": "a"})
    with pytest.raises(ValueError):
        validate_edit({"path": "x", "before": "a" * 8001, "after": "b"})
    assert validate_edit({"path": "x", "before": "a", "after": ""}).after == ""


def test_both_context_policies_keep_typed_edit_and_real_failure():
    messages = [{"role": "system", "content": "protocol"}, {"role": "user", "content": "original exception"},
        {"role": "assistant", "content": json.dumps(edit())},
        {"role": "user", "content": 'Tool result:\n{"exit_code":1,"stdout":"","stderr":"before mismatch","timed_out":false}'},
        {"role": "user", "content": "Harness warning: read exact file before editing."}]
    view = recent_history(messages)
    assert view[-1] == messages[-1] and "before mismatch" in str(view)
    assert recent_history(view) == view
    assert '"edit"' in compact_context(messages)[2]["content"]


@pytest.mark.parametrize("before,after,error", [
    ("absent", "value = 2", "match exactly once"),
    ("value = 1", "value = (", "SyntaxError"),
])
def test_rejected_edit_does_not_touch_source(tmp_path, before, after, error):
    path = tmp_path / "value.py"
    path.write_text("value = 1\n")
    executor = BashExecutor(tmp_path)
    receipt = executor.execute(ToolIntent("e", edit_command(TextEdit("value.py", before, after)), str(tmp_path)))
    assert receipt.exit_code != 0 and error in receipt.stderr
    assert path.read_text() == "value = 1\n"
    assert not list(tmp_path.glob(".agent-edit-*"))


def test_non_unique_edit_reports_bounded_matching_line_numbers(tmp_path):
    path = tmp_path / "value.py"
    path.write_text("value = 1\nother = 0\nvalue = 1\n")
    receipt = BashExecutor(tmp_path).execute(
        ToolIntent("e", edit_command(TextEdit("value.py", "value = 1", "value = 2")), str(tmp_path))
    )
    assert receipt.exit_code != 0
    assert "observed 2" in receipt.stderr
    assert "matching 1-based lines: 1, 3" in receipt.stderr
    assert path.read_text() == "value = 1\nother = 0\nvalue = 1\n"


def test_quotes_unicode_and_newlines_are_data_not_executable_code(tmp_path):
    path = tmp_path / "value.py"
    path.write_bytes(b"value = 'old'\n")
    replacement = 'value = "中文 \'quoted\' ; touch SHOULD_NOT_EXIST"\n'
    receipt = BashExecutor(tmp_path).execute(ToolIntent("e", edit_command(TextEdit("value.py", "value = 'old'\n", replacement)), str(tmp_path)))
    assert receipt.exit_code == 0, receipt.stderr
    assert path.read_text(encoding="utf-8") == replacement
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


@pytest.mark.skipif(os.name == "nt", reason="Linux symlink contract")
def test_symlink_parent_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "value.py").write_text("value = 1\n")
    jail = tmp_path / "jail"
    jail.mkdir()
    (jail / "link").symlink_to(outside, target_is_directory=True)
    receipt = BashExecutor(jail).execute(ToolIntent("e", edit_command(TextEdit("link/value.py", "value = 1", "value = 2")), str(jail)))
    assert receipt.exit_code != 0 and "symlink" in receipt.stderr
    assert (outside / "value.py").read_text() == "value = 1\n"


def make_run(tmp_path, responses, *, steps=8, injector=None,
             require_visible_test_before_done=False, allowed_test_command=""):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.py").write_text("value = 1\n")
    (source / "test_value.py").write_text("import unittest\nimport value\nclass TestValue(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(value.value, 2)\n")
    metadata = {"agent_test_command": allowed_test_command} if allowed_test_command else {}
    task = TaskRecord("typed-edit-test", str(source), "local", "Change value to 2", metadata=metadata)
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "edit-test")
    store = ArtifactStore(tmp_path / "artifacts")
    result = AgentRuntime(store).run(task, workspace, ScriptedModel(responses),
        RunConfig(max_steps=steps, max_seconds=60,
                  require_visible_test_before_done=require_visible_test_before_done),
        run_id="edit-test", failure_injector=injector)
    return result, workspace, store, task


def test_runtime_executes_edit_records_receipt_and_preserves_context(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", "recent-history-v3")
    result, workspace, store, _ = make_run(tmp_path, [read(), edit(), {"command": 'python -m unittest -q'}, {"done": True}])
    assert result.diff and result.status == "completed" and result.visible_test_passed
    assert (workspace.path / "value.py").read_text() == "value = 2\n"
    events = [json.loads(s) for s in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["type"] == "model"]
    assert '"edit"' in calls[2]["context_messages"][4]["content"]
    assert "EDIT_APPLIED value.py" in str(calls[2]["context_messages"])
    row = {"task_id": "typed-edit-test", "run_id": result.run_id, "agent_status": "completed", "evaluation_verdict": "passed", "messages": [{"role": "user", "content": next(e["content"] for e in events if e["type"] == "prompt")}] + [{"role": "assistant", "content": e["content"]} for e in calls]}
    # A synthetic passed row is only a converter contract, not eligibility evidence.
    converted = examples(row, events, policy="recent-history-v3")
    assert converted[1]["messages"][:-1] == calls[1]["context_messages"]


def test_format_retry_keeps_real_rejection_and_never_runs_truncated_command(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", "recent-history-v3")
    result, workspace, store, _ = make_run(tmp_path, ['{"command":"echo truncated', read(), edit(), {"done": True}])
    assert result.diff and result.state.protocol_corrections == 1
    assert (workspace.path / "value.py").read_text() == "value = 2\n"
    assert any(m["content"].startswith("Protocol result:") for m in result.state.messages)
    view = recent_history(result.state.messages[:-1])
    assert "echo truncated" in str(view) and "not valid JSON" in str(view)
    events = [json.loads(s) for s in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    row = {"task_id": "typed-edit-test", "run_id": result.run_id,
           "agent_status": "completed", "evaluation_verdict": "passed",
           "messages": [{"role": "user", "content": next(e["content"] for e in events if e["type"] == "prompt")}] +
           [{"role": "assistant", "content": e["content"]} for e in events if e["type"] == "model"]}
    converted = examples(row, events, policy="recent-history-v3")
    assert len(converted) == 3  # The rejected, unexecuted action is never a target.
    assert "Protocol result:" in str(converted[0]["messages"])
    assert '"executed": false' in str(converted[0]["messages"])
    audit = audit_readiness(converted, expected_system_prompt=AgentRuntime._system_prompt(),
                            expected_prompt_policy="recent-history-v3")
    assert audit["recovery_examples"] == 1
    assert audit["errors"] == {"missing_search_targets": 1}  # This one run is not a full corpus.


def test_format_retry_and_edit_retry_are_bounded(tmp_path):
    result, workspace, _, _ = make_run(tmp_path, ["bad", "bad", edit()])
    assert "correction exhausted" in result.failure_reason and result.steps == 2
    assert not result.diff


def test_protocol_correction_streak_resets_after_successful_tool_progress(tmp_path):
    responses = ["bad", read(), "bad", read(), edit(),
                 {"command": "python -m unittest -q"}, {"done": True}]
    result, workspace, _, _ = make_run(tmp_path, responses)
    assert result.diff and result.visible_test_passed and result.status == "completed"
    assert result.state.protocol_corrections == 2
    assert result.state.protocol_correction_streak == 0


def test_unverified_done_requires_visible_test_before_completion(tmp_path):
    result, _, store, _ = make_run(
        tmp_path,
        [read(), edit(), {"done": True}, {"command": "python -m unittest -q"}, {"done": True}],
        require_visible_test_before_done=True, allowed_test_command="python -m unittest -q",
    )
    assert result.status == "completed" and result.visible_test_passed
    assert result.state.unverified_finish_rejections == 1
    events = [json.loads(line) for line in
              (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    assert any("Exact command: python -m unittest -q" in message["content"]
               and "no pipe" in message["content"]
               for event in events if event.get("type") == "model"
               for message in event.get("context_messages", []))


def test_aligned_export_skips_only_a_recorded_rejected_done(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", "recent-history-v3")
    result, _, store, _ = make_run(
        tmp_path,
        [read(), edit(), {"done": True}, {"command": "python -m unittest -q"}, {"done": True}],
        require_visible_test_before_done=True, allowed_test_command="python -m unittest -q",
    )
    assert result.status == "completed" and result.visible_test_passed
    events = [json.loads(line) for line in
              (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    models = [event for event in events if event["type"] == "model"]
    row = {"task_id": result.state.task_id, "run_id": result.run_id,
           "agent_status": "completed", "evaluation_verdict": "passed",
           "messages": [{"role": "user", "content": next(event["content"] for event in events
                                                    if event["type"] == "prompt")}] +
                       [{"role": "assistant", "content": event["content"]} for event in models]}
    converted = examples(row, events, policy="recent-history-v3")
    assert len(converted) == 4  # read, edit, real test, accepted done
    assert sum(json.loads(example["messages"][-1]["content"])["done"] for example in converted) == 1
    assert json.loads(converted[-2]["messages"][-1]["content"])["command"] == "python -m unittest -q"

    tampered = json.loads(json.dumps(events))
    following = [event for event in tampered if event["type"] == "model"][3]
    following["context_messages"][-1]["content"] = "unverified finish maybe rejected"
    with pytest.raises(ValueError, match="verified finish rejection"):
        examples(row, tampered, policy="recent-history-v3")


def test_piped_test_cannot_certify_visible_pass():
    assert _is_visible_test_run("python -m pytest -q tests/test_public.py")
    assert not _is_visible_test_run("python -m pytest -q tests/test_public.py | tail -20")
    assert not _is_visible_test_run("python -m pytest -q tests/test_public.py; true")


def test_visible_test_gate_requires_the_exact_allowed_command(tmp_path):
    result, _, _, _ = make_run(
        tmp_path,
        [read(), edit(), {"command": "python -m unittest -q test_value"},
         {"done": True}, {"command": "python -m unittest -q"}, {"done": True}],
        require_visible_test_before_done=True, allowed_test_command="python -m unittest -q",
    )
    assert result.status == "completed" and result.visible_test_passed
    assert result.state.unverified_finish_rejections == 1


def test_missing_visible_test_command_does_not_block_independent_evaluation(tmp_path):
    result, _, _, _ = make_run(
        tmp_path, [read(), edit(), {"done": True}], require_visible_test_before_done=True,
    )
    assert result.status == "completed" and result.diff
    assert not result.visible_test_passed
    assert result.state.unverified_finish_rejections == 0


def test_rejected_python_edit_feedback_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", "recent-history-v3")
    result, _, store, _ = make_run(
        tmp_path,
        [read(), edit(after="value = ("), edit(), {"command": "python -m unittest -q"}, {"done": True}],
    )
    assert result.status == "completed" and result.diff
    events = [json.loads(line) for line in
              (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    warnings = [message["content"] for event in events if event.get("type") == "model"
                for message in event.get("context_messages", [])
                if message["content"].startswith("Harness warning:")]
    assert any("rejected before writing" in warning and "SyntaxError" in warning
               for warning in warnings)


def test_two_failed_edits_stop_before_third_write(tmp_path):
    result, workspace, _, _ = make_run(tmp_path, [read(), edit(after="value = ("), edit(after="value = ["), edit()])
    assert result.failure_reason == "edit correction exhausted: two rejected edits"
    assert result.steps == 3 and result.state.failed_edits == 2 and not result.diff


def test_edit_recovery_uses_existing_once_only_journal(tmp_path):
    calls = 0

    def crash(where):
        nonlocal calls
        if where == "after_receipt":
            calls += 1
            if calls == 2:
                raise RuntimeError("crash after edit receipt")

    with pytest.raises(RuntimeError, match="crash"):
        make_run(tmp_path, [read(), edit()], injector=crash)
    source = tmp_path / "source"
    task = TaskRecord("typed-edit-test", str(source), "local", "Change value to 2")
    workspace = Workspace("edit-test", tmp_path / "workspaces/edit-test/workspace", task.base_commit)
    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(task, workspace, ScriptedModel([{"done": True}]), RunConfig(max_steps=8, max_seconds=60), run_id="edit-test", resume=True)
    assert result.status == "completed" and "value = 2" in result.diff
    assert result.state.spent_tokens == 96


def test_recovery_cannot_erase_rejected_edit_limit(tmp_path):
    calls = 0

    def crash(where):
        nonlocal calls
        if where == "after_receipt":
            calls += 1
            if calls == 3:
                raise RuntimeError("second rejection recorded")

    with pytest.raises(RuntimeError, match="second rejection"):
        make_run(tmp_path, [read(), edit(after="value = ("), edit(after="value = [")], injector=crash)
    task = TaskRecord("typed-edit-test", str(tmp_path / "source"), "local", "Change value to 2")
    workspace = Workspace("edit-test", tmp_path / "workspaces/edit-test/workspace", task.base_commit)
    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(task, workspace, ScriptedModel([edit()]), RunConfig(max_steps=8, max_seconds=60), run_id="edit-test", resume=True)
    assert not result.diff and result.state.failed_edits == 2 and result.steps == 3
    assert result.failure_reason == "edit correction exhausted: two rejected edits"


@pytest.mark.skipif(os.getenv("CODEAGENTBENCH_TEST_NSJAIL") != "1", reason="explicit live Linux editor contract")
def test_live_editor_and_retry_in_nsjail(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", "recent-history-v3")
    result, workspace, _, _ = make_run(tmp_path, [read(), edit(after="value = ("), edit(), {"command": "python -m unittest -q"}, {"done": True}])
    assert result.status == "completed" and result.visible_test_passed, result.failure_reason
    assert result.state.failed_edits == 1
    assert "value = 2" in result.diff
