from __future__ import annotations

from codeagentbench.adapters.swe_gym import normalize_swe_gym_row
from codeagentbench.models import EvaluationResult, Verdict
from codeagentbench.tasks.manifest import grouped_split, load_manifest
from codeagentbench.tasks.quality import run_controls
from codeagentbench.training.reward import formal_reward
from codeagentbench.training.sft import trajectory_to_sft


def test_swe_gym_row_mapping_and_grouped_split() -> None:
    rows = [
        normalize_swe_gym_row({"instance_id": "a", "repo": "r", "base_commit": "1", "problem_statement": "a", "patch": "gold", "test_patch": "test"}, split="train"),
        normalize_swe_gym_row({"instance_id": "b", "repo": "r", "base_commit": "1", "problem_statement": "b", "patch": "gold2", "test_patch": "test2"}, split="train"),
    ]
    assigned = grouped_split(rows, smoke=1, dev=0, evaluation=0)
    assert {task.split for task in assigned} == {"train"}
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


def test_quality_controls_admit_demo_task() -> None:
    from pathlib import Path

    task = load_manifest(Path(__file__).parents[1] / "data/manifests/demo.json").tasks[0]
    report = run_controls(task, timeout=60)
    assert report.admitted is True
    assert [item.name for item in report.controls] == ["unfixed", "gold"]
