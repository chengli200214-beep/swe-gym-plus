"""The replay probe classifies model outputs without executing them."""
import json

from codeagentbench.adapters.action import parse_action
from codeagentbench.adapters.model import ModelResponse
from scripts.replay_aligned_actions import action_kind, replay_records


def test_replay_classifies_without_using_the_target_as_input() -> None:
    target = json.dumps({"command": "", "done": True, "message": "finished"})
    prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    record = {"task_id": "train-task", "run_id": "run", "action_index": 3,
              "messages": prompt + [{"role": "assistant", "content": target}]}

    class RecordingModel:
        def complete(self, messages, *, temperature=0):
            assert messages == prompt
            assert temperature == 0
            return ModelResponse('{"command":"git diff","done":false,"message":"check"}', completion_tokens=10)

    result = replay_records([record], RecordingModel())[0]
    assert (result["expected"], result["actual"], result["exact_action"]) == ("done", "command", False)


def test_action_kind_distinguishes_edit_and_invalid_replay() -> None:
    action = parse_action('{"edit":{"path":"moto/example.py","before":"a","after":"b"},"done":false}')
    assert action_kind(action) == "edit"

    class BrokenModel:
        def complete(self, messages, *, temperature=0):
            return ModelResponse("not an action")

    record = {"task_id": "train-task", "run_id": "run", "action_index": 1,
              "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                           {"role": "assistant", "content": '{"command":"","done":true}'}]}
    result = replay_records([record], BrokenModel())[0]
    assert result["actual"] == "invalid"
    assert not result["exact_action"]
