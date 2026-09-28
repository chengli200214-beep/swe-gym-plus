"""Synthetic protocol fixtures, never experiment or training evidence."""
from copy import deepcopy
import json

import pytest

from codeagentbench import train_sft
from codeagentbench.runtime import AgentRuntime
from codeagentbench.training.readiness import audit_readiness, prompt_hash
from scripts import train_aligned_sft


POLICY = "recent-history-v3"


def fixtures(system="synthetic contract"):
    actions = [{"search": {"path": "src.py", "query": "value"}},
               {"read": {"path": "src.py", "start_line": 1, "end_line": 2}},
               {"edit": {"path": "src.py", "before": "value = 1", "after": "value = 2"}},
               {"done": True}]
    prefix = [{"role": "system", "content": system}, {"role": "user", "content": "fixture task"}]
    records = []
    for index, action in enumerate(actions):
        context = deepcopy(prefix)
        if index == 1:
            context += [{"role": "assistant", "content": json.dumps(actions[0])},
                        {"role": "user", "content": 'Protocol result:\n{"executed":false,"error":"repeat"}'},
                        {"role": "user", "content": "Harness warning: change the action."}]
        records.append({"task_id": "fixture", "run_id": "fixture-only", "action_index": index,
                        "source_evaluation_verdict": "passed", "assistant_only_loss": True,
                        "next_action_only_loss": True, "prompt_policy": POLICY,
                        "messages": context + [{"role": "assistant", "content": json.dumps(action)}]})
    return records


def audit(records):
    return audit_readiness(records, expected_system_prompt="synthetic contract", expected_prompt_policy=POLICY)


def test_minimum_alignment_does_not_claim_authenticity():
    report = audit(fixtures())
    assert report["ready"], report
    assert report["target_kinds"] == {"search": 1, "read": 1, "edit": 1, "done": 1}
    assert report["recovery_examples"] == 1
    assert "not receipt/split/tokenizer" in report["scope"]


def test_legacy_shell_training_cannot_silently_be_called_typed_tool_alignment():
    records = fixtures("historical shell-only system")
    for row in records:
        row["prompt_policy"] = "last-action-v2"
        row["messages"] = row["messages"][:2] + [{"role": "assistant", "content": '{"command":"inspect"}'}]
    report = audit(records)
    assert not report["ready"]
    assert report["errors"]["system_prompt_mismatch"] == 4
    assert report["errors"]["context_policy_mismatch"] == 4
    for key in ("search", "read", "edit", "done"):
        assert report["errors"][f"missing_{key}_targets"] == 1


@pytest.mark.parametrize("field,value,error", [
    ("source_evaluation_verdict", "failed", "source_not_declared_passed"),
    ("next_action_only_loss", False, "next_action_masking_not_declared"),
    ("action_index", True, "invalid_source_identity"),
    ("task_id", [], "invalid_source_identity"),
    ("messages", [{"role": "assistant", "content": "bad"}], "invalid_next_action_messages"),
])
def test_invalid_record_metadata_is_rejected(field, value, error):
    records = fixtures()
    records[0][field] = value
    assert audit(records)["errors"][error] == 1


@pytest.mark.parametrize("change", ["same_action", "only_message_changed", "no_result", "executed", "stale"])
def test_recovery_is_not_credited_for_unchanged_or_unobserved_action(change):
    records = fixtures()
    row = records[1]
    if change in {"same_action", "only_message_changed"}:
        target = json.loads(row["messages"][2]["content"])
        if change == "only_message_changed":
            target["message"] = "different explanation, identical operation"
        row["messages"][-1]["content"] = json.dumps(target)
    elif change == "no_result":
        row["messages"][3]["content"] = "Harness warning: not a rejection receipt"
    elif change == "executed":
        row["messages"][3]["content"] = 'Protocol result:\n{"executed":true}'
    else:
        row["messages"].insert(-1, {"role": "assistant", "content": '{"command":"intervening action"}'})
    assert audit(records)["errors"]["missing_changed_action_after_rejection"] == 1


def test_all_source_runs_need_done_and_unique_action_identities():
    records = fixtures()
    records[0]["run_id"] = "unfinished"
    records.append(deepcopy(records[1]))
    errors = audit(records)["errors"]
    assert errors["source_runs_without_done_target"] == 1
    assert errors["duplicate_action_identity"] == 1


def test_preflight_rejects_before_trainer_or_gpu_and_redacts_text(tmp_path, monkeypatch, capsys):
    data = tmp_path / "actions.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in fixtures("private prompt marker")), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train_file": str(data)}), encoding="utf-8")
    monkeypatch.setattr(train_sft, "main", lambda *args: pytest.fail("trainer reached before readiness"))
    monkeypatch.setattr(train_sft, "_require_gpu", lambda *args: pytest.fail("GPU reached before readiness"))
    assert train_aligned_sft.main(["--config", str(config), "--dry-run"]) == 2
    output = capsys.readouterr().out
    assert "system_prompt_mismatch" in output
    assert "private prompt marker" not in output


def test_cli_override_and_encoder_restore(tmp_path, monkeypatch):
    system = AgentRuntime._system_prompt()
    data = tmp_path / "actions.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in fixtures(system)), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train_file": "not-used.jsonl", "deployment_prompt_policy": POLICY,
                                 "deployment_system_prompt_sha256": prompt_hash(system)}), encoding="utf-8")
    original = train_sft.encode_example
    called = []

    def trainer(argv):
        assert train_sft.encode_example is train_aligned_sft.encode_next_action
        called.append(argv)
        return 0

    monkeypatch.setattr(train_sft, "main", trainer)
    argv = ["--config", str(config), "--train-file", str(data), "--dry-run"]
    assert train_aligned_sft.main(argv) == 0
    assert called == [argv]
    assert train_sft.encode_example is original
