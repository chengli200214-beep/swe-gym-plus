"""The adapter must send the same prepared input that Runtime records."""
from contextlib import nullcontext
from types import SimpleNamespace
import json

from codeagentbench.adapters.model import LocalHFModel
from codeagentbench.harness.context_history import prepare_context


class Tokens:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, device):
        return self

    def __getitem__(self, key):
        return SimpleNamespace(shape=(3,))


class RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        if kwargs.get("tokenize"):
            return {"input_ids": Tokens(7)}
        return "rendered input"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(7))}

    def decode(self, tokens, **kwargs):
        return '{"done":true}'


def prepared_recovery_input():
    messages = [{"role": "system", "content": "contract"}, {"role": "user", "content": "task"}]
    for index in range(2):
        messages += [{"role": "assistant", "content": json.dumps({"command": f"inspect-{index}"})},
                     {"role": "user", "content": "Tool result:\n" + json.dumps(
                         {"exit_code": 0, "stdout": f"real evidence {index}", "stderr": "", "timed_out": False})}]
    messages.append({"role": "user", "content": "Harness warning: preserve the observed source and change action."})
    return messages


def test_counting_and_generation_use_exact_runtime_input_without_second_compaction():
    model = LocalHFModel.__new__(LocalHFModel)
    model.prompt_policy = "last-action-v2"
    model.max_new_tokens = 32
    model._tokenizer = RecordingTokenizer()
    model._torch = SimpleNamespace(inference_mode=nullcontext)
    model._model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device="cpu")]),
        generate=lambda **kwargs: Tokens(10))
    messages = prepared_recovery_input()
    original = json.dumps(messages)
    assert prepare_context(messages, model.prompt_policy) != messages  # regression trigger
    assert model.request_token_bound(messages) == 39
    assert model.complete(messages).text == '{"done":true}'
    assert model._tokenizer.seen == [messages, messages]
    assert json.dumps(messages) == original
