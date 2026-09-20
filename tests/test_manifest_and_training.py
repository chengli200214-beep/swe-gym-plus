from __future__ import annotations

from codeagentbench.adapters.swe_gym import normalize_swe_gym_row
from codeagentbench.models import EvaluationResult, Verdict
from codeagentbench.tasks.manifest import grouped_split, load_manifest
from codeagentbench.tasks.quality import run_controls
from codeagentbench.training.reward import formal_reward
from codeagentbench.training.sft import trajectory_to_sft
from codeagentbench.train_sft import encode_example


def test_swe_gym_row_mapping_and_grouped_split() -> None:
    rows = [
        normalize_swe_gym_row({"instance_id": "a", "repo": "r", "base_commit": "1", "problem_statement": "a", "patch": "gold", "test_patch": "test", "group_id": "pr-1"}, split="train"),
        normalize_swe_gym_row({"instance_id": "b", "repo": "r", "base_commit": "1", "problem_statement": "b", "patch": "gold2", "test_patch": "test2", "group_id": "pr-1"}, split="train"),
    ]
    assigned = grouped_split(rows, smoke=1, dev=0, evaluation=0)
    assert {task.split for task in assigned} == {"train"}


def test_tasks_without_group_id_split_independently() -> None:
    rows = [
        normalize_swe_gym_row(
            {
                "instance_id": name,
                "repo": "r",
                "base_commit": "1",
                "problem_statement": name,
                "patch": "gold",
                "test_patch": "test",
            },
            split="train",
        )
        for name in ("a", "b", "c")
    ]

    assigned = grouped_split(rows, smoke=1, dev=1, evaluation=1)

    assert [task.split for task in assigned].count("smoke") == 1
    assert [task.split for task in assigned].count("dev") == 1
    assert [task.split for task in assigned].count("eval") == 1
    assert rows[0].agent_view().issue == "a"


def test_swe_gym_string_test_lists_are_normalized() -> None:
    task = normalize_swe_gym_row({
        "instance_id": "x",
        "repo": "owner/repo",
        "base_commit": "abc",
        "problem_statement": "issue",
        "FAIL_TO_PASS": "['tests/test_x.py::test_one']",
        "PASS_TO_PASS": "[]",
    })
    assert task.eval_spec.fail_to_pass == ("tests/test_x.py::test_one",)
    assert task.eval_spec.test_command == "python -m pytest -q tests/test_x.py::test_one"


def test_sft_preserves_interaction_roles_and_reward_is_binary() -> None:
    record = trajectory_to_sft([
        {"type": "prompt", "content": "issue"},
        {"type": "model", "content": '{"command":"pytest"}'},
        {"type": "tool", "receipt": {"stdout": "ok", "stderr": ""}},
    ])
    assert [message["role"] for message in record["messages"]] == ["user", "assistant", "tool"]
    assert formal_reward(EvaluationResult(Verdict.PASSED, 0, True, True)) == 1.0
    assert formal_reward(EvaluationResult(Verdict.FAILED, 1, False, False)) == 0.0


def test_sft_truncation_keeps_final_supervised_turn() -> None:
    class TinyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "|".join(f"{m['role']}:{m['content']}" for m in messages)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    messages = [
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": "early"},
        {"role": "tool", "content": "x" * 40},
        {"role": "assistant", "content": "final patch action"},
    ]
    encoded = encode_example(TinyTokenizer(), messages, max_seq_len=32)
    assert encoded is not None
    assert any(label != -100 for label in encoded["labels"][-16:])


def test_quality_controls_admit_demo_task() -> None:
    from pathlib import Path

    task = load_manifest(Path(__file__).parents[1] / "data/manifests/demo.json").tasks[0]
    report = run_controls(task, timeout=60)
    assert report.admitted is True
    assert [item.name for item in report.controls] == ["unfixed", "gold"]
