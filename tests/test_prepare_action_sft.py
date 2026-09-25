from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_action_sft import action_examples, prepare


def _row() -> dict:
    return {
        "task_id": "task-1",
        "run_id": "run-1",
        "evaluation_verdict": "passed",
        "messages": [
            {"role": "user", "content": json.dumps({"instance_id": "task-1", "issue": "Fix the function", "allowed_test_command": ""})},
            {"role": "assistant", "content": '<command>rg "function" src</command>'},
            {"role": "tool", "content": "x" * 100 + "useful tail"},
            {"role": "assistant", "content": '{"command":"python -m pytest -q","done":false,"message":"test"}'},
            {"role": "tool", "content": "1 passed"},
            {"role": "assistant", "content": '{"command":"","done":true,"message":"finished"}'},
        ],
    }


def test_action_examples_are_canonical_and_exclude_done() -> None:
    examples = action_examples(_row(), system_prompt="Return JSON", context_chars=12)
    assert len(examples) == 2
    assert [item["action_index"] for item in examples] == [0, 1]
    assert all([message["role"] for message in item["messages"]] == ["system", "user", "assistant"] for item in examples)
    assert json.loads(examples[0]["messages"][-1]["content"])["command"] == 'rg "function" src'
    assert "useful tail" in examples[1]["messages"][1]["content"]
    assert "x" * 10 not in examples[1]["messages"][1]["content"]
    assert all(json.loads(item["messages"][-1]["content"])["done"] is False for item in examples)


def test_prepare_requires_passed_unique_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "private" / "actions.jsonl"
    row = _row()
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    receipt = prepare(source, output)
    assert receipt["distinct_tasks"] == 1
    assert receipt["action_examples"] == 2
    assert receipt["blind_prompt_checked"] is True
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert output.with_suffix(".receipt.json").exists()

    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source task"):
        prepare(source, output)
    row["evaluation_verdict"] = "failed"
    with pytest.raises(ValueError, match="passed independent evaluation"):
        action_examples(row, system_prompt="Return JSON")


def test_action_examples_reject_historical_evaluator_command() -> None:
    row = _row()
    row["messages"][0]["content"] = json.dumps({
        "instance_id": "task-1",
        "issue": "Fix the function",
        "allowed_test_command": "python -m pytest -q tests/hidden_test.py::test_gold",
    })
    with pytest.raises(ValueError, match="exposes an evaluator test command"):
        action_examples(row, system_prompt="Return JSON")


def test_action_examples_reject_unverifiable_prompt() -> None:
    row = _row()
    row["messages"][0]["content"] = "Fix the function"
    with pytest.raises(ValueError, match="not auditable JSON"):
        action_examples(row, system_prompt="Return JSON")


def test_formal_examples_include_only_verified_completion_and_history():
    row = _row()
    row["agent_status"] = "completed"
    examples = action_examples(row, system_prompt="Return JSON", context_chars=4000, include_done=True, history=True)
    assert len(examples) == 3
    assert json.loads(examples[-1]["messages"][-1]["content"])["done"]
    assert "Previous action:" in examples[-1]["messages"][1]["content"]
    assert "1 passed" in examples[-1]["messages"][1]["content"]
    row["agent_status"] = "blocked"
    assert len(action_examples(row, system_prompt="Return JSON", include_done=True)) == 2
