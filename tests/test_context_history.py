import json

import pytest

from codeagentbench.adapters.model import LocalHFModel, ModelResponse
from codeagentbench.harness.context_history import prepare_context, recent_history
from codeagentbench.models import RunConfig, TaskRecord
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore


WARNING = "Harness warning: the same command ran twice without any workspace change. Choose a different command now."


def transcript(count=1, output="observed"):
    messages = [{"role": "system", "content": "protocol"}, {"role": "user", "content": "issue"}]
    for index in range(count):
        messages.extend([
            {"role": "assistant", "content": json.dumps({"command": f"read-{index}", "done": False})},
            {"role": "user", "content": "Tool result:\n" + json.dumps(
                {"exit_code": 0, "stdout": output, "stderr": "", "timed_out": False})},
        ])
    return messages


def test_repetition_warning_survives_successful_tool_and_double_compaction():
    messages = transcript(2) + [{"role": "user", "content": WARNING}]
    view = recent_history(messages)
    assert view[-1]["content"] == WARNING
    assert len([m for m in view if m["role"] == "assistant"]) == 2
    assert recent_history(view) == view
    assert WARNING not in str(prepare_context(messages, "last-action-v2"))


def test_four_complete_turns_keep_checkpoint_and_mark_truncation():
    messages = transcript(7, "x" * 6000)
    checkpoint = {"role": "user", "content": "Harness checkpoint: 7 actions without a patch."}
    messages.append(checkpoint)
    view = recent_history(messages)
    actions = [json.loads(m["content"])["command"] for m in view if m["role"] == "assistant"]
    assert actions == ["read-3", "read-4", "read-5", "read-6"]
    outputs = [json.loads(m["content"].split("\n", 1)[1]) for m in view if m["content"].startswith("Tool result:\n")]
    assert [len(o["stdout"]) for o in outputs] == [1000, 1000, 1000, 4000]
    assert all(o["stdout_truncated"] for o in outputs)
    assert view[-1] == checkpoint
    assert recent_history(view) == view


def test_tool_text_is_not_promoted_to_runtime_warning():
    view = recent_history(transcript(output=WARNING))
    assert not any(m["content"].startswith("Harness warning:") for m in view)
    assert WARNING in view[-1]["content"]


def test_old_warning_expires_after_a_new_observation():
    messages = transcript() + [{"role": "user", "content": WARNING}] + transcript(2)[4:]
    view = recent_history(messages)
    assert len([m for m in view if m["role"] == "assistant"]) == 2
    assert not any(m["content"] == WARNING for m in view)


def test_missing_action_or_receipt_fields_fail_closed():
    with pytest.raises(ValueError, match="exactly one"):
        recent_history(transcript()[:2] + transcript()[3:])
    messages = transcript()
    messages[-1]["content"] = "Tool result:\n{}"
    with pytest.raises(ValueError, match="incomplete"):
        recent_history(messages)


def test_adapter_uses_shared_history_policy():
    model = LocalHFModel.__new__(LocalHFModel)
    model.prompt_policy = "recent-history-v3"
    messages = transcript(3) + [{"role": "user", "content": WARNING}]
    assert model._prepare_messages(messages) == recent_history(messages)
    with pytest.raises(ValueError, match="unknown"):
        prepare_context(messages, "typo")


@pytest.mark.parametrize("policy", ["last-action-v2", "recent-history-v3"])
def test_runtime_compression_preserves_repeat_warning_and_allows_new_action(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY", policy)
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("before")
    task = TaskRecord("context-probe", str(source), "local", "update value")
    workspace = WorkspaceManager(tmp_path / "workspace").create(task, "probe")

    class FeedbackModel:
        def complete(self, messages, **kwargs):
            view = prepare_context(messages, policy)
            if any(m["content"].startswith("Harness warning: this exact command") for m in view):
                action = {"command": 'python -c "from pathlib import Path; Path(\'value.txt\').write_text(\'after\')"'}
            elif any("write_text" in m["content"] for m in view if m["role"] == "assistant"):
                action = {"done": True}
            else:
                # Force the runtime's >16000-character compression boundary too.
                action = {"command": 'python -c "print(\'x\'*18000)"'}
            return ModelResponse(json.dumps(action))

    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(
        task, workspace, FeedbackModel(), RunConfig(max_steps=5, max_seconds=60), run_id="probe")
    assert result.diff
    assert (workspace.path / "value.txt").read_text() == "after"
    assert result.status == "completed"
