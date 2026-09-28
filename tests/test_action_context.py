import json

import pytest

from codeagentbench.adapters.action import AgentAction
from codeagentbench.adapters.source_read import SourceRead
from codeagentbench.adapters.source_search import SourceSearch
from codeagentbench.adapters.text_edit import TextEdit
from codeagentbench.harness.tool_observation import tool_observation
from codeagentbench.training.action_context import compact_context, encode_next_action
from codeagentbench.harness.context_history import prepare_context
from codeagentbench.runtime import AgentRuntime, _repeat_recovery_guidance
from codeagentbench.training.readiness import _changed_after_rejection
from scripts.prepare_aligned_sft import examples


def history(stdout="actual observation", code=0):
    return [
        {"role": "system", "content": "protocol"},
        {"role": "user", "content": "original issue"},
        {"role": "assistant", "content": '{"command":"grep target src.py","done":false,"message":"inspect"}'},
        {"role": "user", "content": "Tool result:\n" + json.dumps({"exit_code": code, "stdout": stdout, "stderr": "actual error" if code else "", "timed_out": False})},
    ]


def test_compact_preserves_issue_action_and_real_receipt():
    result = compact_context(history())
    assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]
    assert result[1]["content"] == "original issue"
    assert result[-1]["content"].endswith('"timed_out": false}')


def test_compact_read_receipt_preserves_projection_for_protocol_recovery():
    action = AgentAction(read=SourceRead("src/module.py", 1, 10), message="read source")
    raw_receipt = {
        "exit_code": 0,
        "stdout": json.dumps({"path": "src/module.py", "sha256": "a" * 64,
                              "start_line": 1, "end_line": 1, "next_line": 2,
                              "text": "class Example:"}),
        "stderr": "", "timed_out": False,
    }
    projected = tool_observation(action, raw_receipt)
    messages = [
        {"role": "system", "content": "protocol"},
        {"role": "user", "content": "original issue"},
        {"role": "assistant", "content": json.dumps(action.to_dict())},
        {"role": "user", "content": "Tool result:\n" + json.dumps(projected)},
    ]
    compacted = compact_context(messages)
    compacted_receipt = json.loads(compacted[-1]["content"].split("\n", 1)[1])
    assert compacted_receipt["stdout_format"] == "decoded-source-text-v1"
    assert compacted_receipt["source_read"]["path"] == "src/module.py"
    recovered = compact_context(compacted + [
        {"role": "assistant", "content": '{"command":"grep Example src/module.py","done":false}'},
        {"role": "user", "content": "Protocol result:\n" + json.dumps({"error": "repeated command", "executed": False})},
    ])
    recovered_receipt = json.loads(recovered[3]["content"].split("\n", 1)[1])
    assert recovered_receipt["stdout_format"] == "decoded-source-text-v1"
    assert "class Example" in recovered_receipt["stdout"]


def test_only_latest_real_interaction_and_no_synthetic_output():
    messages = history() + [
        {"role": "assistant", "content": '{"command":"git diff","done":false,"message":"verify"}'},
        {"role": "user", "content": history("real diff")[3]["content"]},
        {"role": "user", "content": "Evidence summary after context compression:\nnot a real receipt"},
    ]
    result = compact_context(messages)
    assert "git diff" in result[2]["content"]
    assert "real diff" in result[3]["content"]
    assert "grep target" not in str(result)
    assert "not a real receipt" not in str(result)


def test_bounded_output_and_idempotent_failed_feedback():
    result = compact_context(history("x" * 6000, 1))
    assert json.loads(result[3]["content"].split("\n", 1)[1])["stdout_truncated"]
    assert "previous command failed" in result[-1]["content"]
    assert compact_context(result) == result


def test_repeat_block_keeps_prior_receipt_and_harness_warning():
    messages = history("latest unrelated output") + [
        {"role": "assistant", "content": '{"command":"grep target src.py","done":false,"message":"repeat"}'},
        {"role": "user", "content": "Protocol result:\n" + json.dumps({
            "error": "exact command already executed",
            "executed": False,
            "prior_real_tool_receipt": {
                "action_id": "actual-action-1",
                "exit_code": 1,
                "stdout": "matching real source evidence",
                "stderr": "",
                "timed_out": False,
            },
        })},
        {"role": "user", "content": (
            "Harness warning: this exact command already ran against the unchanged workspace version. "
            "It was not executed again. Use the prior real receipt above; preserve that evidence and choose a different action. "
            "For this source read, do not request the same path/range again. Use the prior real receipt; "
            "if it shows relevant implementation, make one minimal exact edit anchored to that observed text; "
            "otherwise inspect a different targeted source range or search. Never edit tests."
        )},
    ]
    result = compact_context(messages)
    rendered = "\n".join(message["content"] for message in result)
    assert "latest unrelated output" in rendered
    assert "matching real source evidence" in rendered
    assert "preserve that evidence" in rendered
    assert "same path/range again" in rendered
    assert "make one minimal exact edit" in rendered
    assert "Never edit tests" in rendered
    assert '"executed": false' in rendered
    assert rendered.count("Tool result:\n") == 1


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (AgentAction(read=SourceRead("src/module.py", 10, 20)), "same path/range again"),
        (AgentAction(search=SourceSearch("src/module.py", "Thing")), "Do not repeat this exact search"),
        (AgentAction(edit=TextEdit("src/module.py", "old", "new")), "non-unique before match"),
        (AgentAction(command="grep -R target src"), "same shell command"),
    ],
)
def test_repeat_recovery_guidance_matches_action_kind(action, expected):
    guidance = _repeat_recovery_guidance(action)
    assert expected in guidance
    assert "prior real receipt" in guidance
    assert "Never edit tests" in guidance


def test_receipt_without_action_fails_closed():
    with pytest.raises(ValueError, match="lost"):
        compact_context(history()[:2] + history()[3:])


def test_runtime_selects_legacy_or_native_policy_explicitly():
    assert prepare_context(history(), "last-action-v2") == compact_context(history())
    assert prepare_context(history(), "native") == history()


class CharTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return result + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}


def test_only_next_action_is_supervised_not_prior_assistant():
    tokenizer = CharTokenizer()
    messages = compact_context(history()) + [{"role": "assistant", "content": "NEXT_ACTION"}]
    result = encode_next_action(tokenizer, messages, 8192)
    target = bytes(x for x, label in zip(result["input_ids"], result["labels"]) if label != -100).decode()
    assert target == "NEXT_ACTION<end>"
    assert "grep target" not in target


def test_encoder_never_truncates_target():
    with pytest.raises(ValueError, match="never truncate"):
        encode_next_action(CharTokenizer(), history()[:2] + [{"role": "assistant", "content": "action"}], 2)


def test_preparation_pairs_actual_receipts_and_done():
    issue = json.dumps({"instance_id": "task-1", "issue": "fix", "allowed_test_command": ""})
    action = history()[2]["content"]
    done = '{"command":"","done":true,"message":"verified"}'
    row = {"task_id": "task-1", "run_id": "run-1", "agent_status": "completed", "evaluation_verdict": "passed", "messages": [{"role": "user", "content": issue}, {"role": "assistant", "content": action}, {"role": "tool", "content": "observed"}, {"role": "assistant", "content": done}]}
    receipt = {"exit_code": 0, "stdout": "observed", "stderr": "", "timed_out": False}
    prefix = [{"role": "system", "content": AgentRuntime._system_prompt()}, {"role": "user", "content": issue}]
    second = prepare_context(prefix + [{"role": "assistant", "content": action},
                                       {"role": "user", "content": "Tool result:\n" + json.dumps(receipt)}], "last-action-v2")
    events = [{"type": "prompt", "content": issue},
              {"type": "model", "content": action, "context_messages": prefix,
               "prompt_policy": "last-action-v2", "context_contract": "runtime-once-v1"},
              {"type": "tool", "intent": {"command": "grep target src.py"}, "receipt": receipt},
              {"type": "model", "content": done, "context_messages": second,
               "prompt_policy": "last-action-v2", "context_contract": "runtime-once-v1"}]
    converted = examples(row, events)
    assert len(converted) == 2
    assert [m["role"] for m in converted[1]["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert "observed" in converted[1]["messages"][3]["content"]
    events[2]["intent"]["command"] = "different command"
    with pytest.raises(ValueError, match="does not match"):
        examples(row, events)


def test_export_uses_recorded_input_and_skips_rejected_action():
    issue = json.dumps({"instance_id": "task-1", "issue": "fix", "allowed_test_command": ""})
    system = AgentRuntime._system_prompt()
    first = '{"command":"grep target src.py"}'
    rejected = '{"edit":{"path":"src.py","before":"old","after":"new"}}'
    second = '{"command":"grep another src.py"}'
    third = '{"command":"git diff"}'
    done = '{"done":true}'
    receipt = {"exit_code": 0, "stdout": "observed", "stderr": "", "timed_out": False}
    prefix = [{"role": "system", "content": system}, {"role": "user", "content": issue}]
    row = {"task_id": "task-1", "run_id": "run-1", "agent_status": "completed",
           "evaluation_verdict": "passed", "messages": [{"role": "user", "content": issue}] +
           [{"role": "assistant", "content": text} for text in (first, rejected, second, third, done)]}
    events = [{"type": "prompt", "content": issue}]
    live = prefix
    for action, outcome in ((first, "tool"), (rejected, "reject"), (second, "tool"), (third, "tool")):
        live = prepare_context(live, "last-action-v2")
        events.append({"type": "model", "content": action, "context_messages": live,
                       "prompt_policy": "last-action-v2", "context_contract": "runtime-once-v1"})
        live = live + [{"role": "assistant", "content": action}]
        if outcome == "tool":
            events.append({"type": "tool", "intent": {"command": json.loads(action)["command"]},
                           "receipt": receipt})
            live.append({"role": "user", "content": "Tool result:\n" + json.dumps(receipt)})
        else:
            events.append({"type": "protocol_rejection", "reason": "read required"})
            live.append({"role": "user", "content": 'Protocol result:\n{"error":"read required","executed":false}'})
            live.append({"role": "user", "content": "Harness warning: no tool ran; choose a different action."})
    recorded = prepare_context(live, "last-action-v2")
    assert prepare_context(recorded, "last-action-v2") != recorded  # old exporter lost one turn
    events.append({"type": "model", "content": done, "context_messages": recorded,
                   "prompt_policy": "last-action-v2", "context_contract": "runtime-once-v1"})
    converted = examples(row, events)
    assert len(converted) == 4
    assert converted[-1]["messages"][:-1] == recorded
    assert "grep another" in str(converted[-1]["messages"])
    assert all(rejected not in r["messages"][-1]["content"] for r in converted)
    assert _changed_after_rejection(converted[1]["messages"],
                                    AgentAction(command="grep another src.py"))


def test_export_refuses_ambiguous_historical_context():
    issue = json.dumps({"instance_id": "task-1", "issue": "fix", "allowed_test_command": ""})
    row = {"task_id": "task-1", "run_id": "run-1", "agent_status": "completed",
           "evaluation_verdict": "passed", "messages": [{"role": "user", "content": issue},
           {"role": "assistant", "content": '{"done":true}'}]}
    event = {"type": "model", "content": '{"done":true}',
             "context_messages": [{"role": "system", "content": AgentRuntime._system_prompt()},
                                  {"role": "user", "content": issue}],
             "prompt_policy": "last-action-v2"}
    with pytest.raises(ValueError, match="context contract"):
        examples(row, [{"type": "prompt", "content": issue}, event])
