import json

import pytest

from codeagentbench.training.action_context import compact_context, encode_next_action
from codeagentbench.adapters.model import LocalHFModel
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


def test_receipt_without_action_fails_closed():
    with pytest.raises(ValueError, match="lost"):
        compact_context(history()[:2] + history()[3:])


def test_local_policy_same_for_bound_and_generation():
    model = LocalHFModel.__new__(LocalHFModel)
    model.prompt_policy = "last-action-v2"
    assert model._prepare_messages(history()) == compact_context(history())
    model.prompt_policy = "native"
    assert model._prepare_messages(history()) == history()


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
    events = [{"type": "prompt", "content": issue}, {"type": "model", "content": action}, {"type": "tool", "intent": {"command": "grep target src.py"}, "receipt": {"exit_code": 0, "stdout": "observed", "stderr": "", "timed_out": False}}, {"type": "model", "content": done}]
    converted = examples(row, events)
    assert len(converted) == 2
    assert [m["role"] for m in converted[1]["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert "observed" in converted[1]["messages"][3]["content"]
    events[2]["intent"]["command"] = "different command"
    with pytest.raises(ValueError, match="does not match"):
        examples(row, events)
