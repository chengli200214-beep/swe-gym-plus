import json
import shlex

from codeagentbench.adapters.swe_gym import normalize_swe_gym_row
from scripts.accelerate_campaign import completed_controls, quote_manifest, rejection, seed_ok


def test_shell_sensitive_test_selector_is_one_argument():
    name = "tests/test_x.py::test_parser[value(a b);echo SECRET]"
    task = normalize_swe_gym_row({"instance_id": "x", "FAIL_TO_PASS": [name]})
    assert shlex.split(task.eval_spec.test_command) == ["python", "-m", "pytest", "-q", name]


def test_quote_repair_preserves_frozen_source_and_every_selector():
    names = ["t.py::test_x[a(b)]", "t.py::test_y[foo bar]"]
    source = {"tasks": [{"instance_id": "x", "eval_spec": {
        "fail_to_pass": names[:1], "pass_to_pass": names[1:], "gold_patch": "PRIVATE",
        "test_command": "python -m pytest -q " + " ".join(names)}}]}
    original = json.dumps(source)
    fixed, changed = quote_manifest(source)
    assert json.dumps(source) == original
    assert changed == ["x"]
    assert shlex.split(fixed["tasks"][0]["eval_spec"]["test_command"])[4:] == names
    assert fixed["tasks"][0]["eval_spec"]["gold_patch"] == "PRIVATE"


def test_custom_control_command_is_not_rewritten():
    source = {"tasks": [{"instance_id": "x", "eval_spec": {
        "fail_to_pass": ["t.py::test_a"], "pass_to_pass": [], "test_command": "python custom_runner.py"}}]}
    assert quote_manifest(source) == (source, [])


def test_infrastructure_failure_is_distinct_from_model_or_gold_failure():
    report = {"admitted": False, "controls": [{"name": "gold", "reason": "isolated test unavailable", "exit_code": None}]}
    assert rejection(report) == "isolation_unavailable"


def test_failed_agent_with_passing_patch_is_not_clean_training_seed():
    row = {"task_id": "x", "agent_status": "failed", "evaluation_verdict": "passed"}
    assert not seed_ok(row, ["x"])


def test_completed_seed_requires_blind_prompt_done_and_train_membership():
    row = {"task_id": "x", "agent_status": "completed", "evaluation_verdict": "passed",
           "messages": [{"role": "user", "content": json.dumps({"instance_id": "x"})},
                        {"role": "assistant", "content": '{"done":true}'}]}
    assert seed_ok(row, ["x"])
    assert not seed_ok(row, ["heldout"])
    row["messages"][0]["content"] = json.dumps({"instance_id": "x", "gold_patch": "LEAK"})
    assert not seed_ok(row, ["x"])


def test_visible_test_seed_requires_explicit_matching_train_allowlist():
    command = "python -m pytest -q -x tests/test_public.py"
    row = {"task_id": "x", "agent_status": "completed", "evaluation_verdict": "passed",
           "messages": [{"role": "user", "content": json.dumps({"instance_id": "x", "allowed_test_command": command})},
                        {"role": "assistant", "content": '{"done":true}'}]}
    assert not seed_ok(row, ["x"])
    assert not seed_ok(row, ["x"], visible_test_commands={"x": "python -m pytest -q tests/test_other.py"})
    assert seed_ok(row, ["x"], visible_test_commands={"x": command})
    assert not seed_ok(row, ["heldout"], visible_test_commands={"x": command})
    row["messages"][0]["content"] = json.dumps({"instance_id": "x", "allowed_test_command": command,
                                                  "test_patch": "SECRET"})
    assert not seed_ok(row, ["x"], visible_test_commands={"x": command})


def test_ready_collection_ignores_unwritten_or_incomplete_controls(tmp_path):
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "ready.json").write_text('{"task_id":"ready","admitted":true}')
    (quality / "writing.json").write_text('{"task_id":')
    (quality / "heldout.json").write_text('{"task_id":"heldout","admitted":true}')
    assert set(completed_controls(tmp_path, ["ready", "writing", "missing"])) == {"ready"}
