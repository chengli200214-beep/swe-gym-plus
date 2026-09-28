"""Unit checks for the assisted read-to-edit diagnostic boundary."""
from __future__ import annotations

import json
from collections import UserDict
from pathlib import Path

from codeagentbench.adapters.model import ModelResponse
from scripts.probe_read_to_edit import SCRIPTED_READS, ScriptedPreReadModel, action_kind, summarize_events


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [len(message["content"]) for message in messages]


class FakeModel:
    prompt_policy = "recent-history-v3"
    _tokenizer = FakeTokenizer()

    def request_token_bound(self, messages):
        return 123

    def complete(self, messages, *, temperature=0):
        return ModelResponse('{"done":true}', prompt_tokens=10)


class DictTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = super().apply_chat_template(messages, tokenize=tokenize,
                                          add_generation_prompt=add_generation_prompt)
        return UserDict({"input_ids": [ids]})


def test_scripted_reads_are_declared_and_handoff_hashes_actual_input():
    model = ScriptedPreReadModel(FakeModel())
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "issue"}]
    assert model.request_token_bound(messages) == 123
    for expected in SCRIPTED_READS:
        response = model.complete(messages)
        assert json.loads(response.text) == {"read": expected, "done": False}
        assert response.prompt_tokens == 0
    assert model.complete(messages).text == '{"done":true}'
    assert model.first_real_prompt_sha256
    assert model.first_real_token_ids_sha256
    assert model.first_real_input_tokens == 2
    assert model.complete(messages).text == '{"done":true}'


def test_handoff_accepts_tokenizer_batch_encoding():
    fake = FakeModel()
    fake._tokenizer = DictTokenizer()
    model = ScriptedPreReadModel(fake)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "issue"}]
    for _ in SCRIPTED_READS:
        model.complete(messages)
    assert model.complete(messages).text == '{"done":true}'
    assert model.first_real_input_tokens == 2
    assert model.first_real_prompt_sha256
    assert model.first_real_token_ids_sha256


def test_event_summary_does_not_count_harness_reads_as_model_edits(tmp_path: Path):
    events = []
    for index, read in enumerate(SCRIPTED_READS):
        action = json.dumps({"read": read, "done": False})
        events.extend([
            {"type": "model", "content": action},
            {"type": "tool", "intent": {"action_json": action, "pre_digest": "same"},
             "receipt": {"exit_code": 0, "timed_out": False, "post_digest": "same"}},
        ])
    edit = '{"edit":{"path":"moto/core/utils.py","before":"old","after":"new"},"done":false}'
    events.extend([
        {"type": "model", "content": edit},
        {"type": "tool", "intent": {"action_json": edit, "pre_digest": "before"},
         "receipt": {"exit_code": 0, "timed_out": False, "post_digest": "after"}},
    ])
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    assert summarize_events(path) == {
        "scripted_read_receipts": 2, "model_action_kinds": ["edit"],
        "valid_edit_proposed": True, "edit_executed": True,
    }
    assert action_kind("not json") == "invalid"
