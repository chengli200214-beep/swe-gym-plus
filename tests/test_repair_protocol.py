import json

import httpx
import pytest

from scripts.repair_protocol_next import ProtocolTeacher, retry_candidates, teacher_environment, verify_drained_receipts


def reply(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))


def test_protocol_teacher_has_separate_low_thinking_text_protocol(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: reply(200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    calls = []
    def post(*a, **k):
        calls.append(k["json"])
        return reply(200, {"id": "receipt", "model": "deepseek-v4-pro", "choices": [{"finish_reason": "stop", "message": {"content": '{"done":true}', "reasoning_content": "private thought"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 80, "completion_tokens_details": {"reasoning_tokens": 50}}})
    monkeypatch.setattr(httpx, "post", post)
    original = [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "public issue"}]
    result = ProtocolTeacher().complete(original)
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["response_format"] == {"type": "text"}
    assert calls[0]["max_tokens"] == 16384
    assert original[0]["content"] == "Return JSON"
    assert result.completion_tokens == 80
    out = capsys.readouterr().out
    assert '"finish_reason": "stop"' in out and "private thought" not in out


def test_unknown_teacher_outcome_is_not_replayed(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: reply(200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    calls = []
    def post(*a, **k):
        calls.append(1); raise httpx.ReadTimeout("unknown")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        ProtocolTeacher().complete([{"role": "system", "content": "JSON"}])
    assert len(calls) == 1


def test_retry_pool_excludes_unknown_active_passed_and_heldout(tmp_path):
    source, pro = tmp_path / "source", tmp_path / "pro"
    (source / "quality").mkdir(parents=True); (pro / "outcomes").mkdir(parents=True)
    for task, status in {"ready": "rejected", "unknown": "uncertain", "passed": "rejected", "heldout": "rejected"}.items():
        (source / "quality" / (task + ".json")).write_text('{"admitted":true}')
        (pro / "outcomes" / (task + ".json")).write_text(json.dumps({"status": status}))
    assert retry_candidates(source, pro, ["ready", "unknown", "passed", "active"], {"passed": None}) == ["ready"]


def test_teacher_environment_satisfies_cli_floor_without_contaminating_evaluation():
    safe = {"PATH": "/usr/bin:/bin", "CODEAGENTBENCH_EXECUTOR": "bwrap"}
    api = teacher_environment(safe, "dummy-test-key")
    assert api["DEEPSEEK_MIN_BALANCE_CNY"] == "0"
    assert api["DEEPSEEK_API_KEY"] == "dummy-test-key"
    assert "DEEPSEEK_API_KEY" not in safe and "DEEPSEEK_MIN_BALANCE_CNY" not in safe


def test_high_profile_reaches_the_actual_child_request(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setenv("CODEAGENTBENCH_PROTOCOL_REASONING", "high")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: reply(200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    calls = []
    def post(*a, **k):
        calls.append(k["json"])
        return reply(200, {"choices": [{"message": {"content": '{"done":true}'}}]})
    monkeypatch.setattr(httpx, "post", post)
    ProtocolTeacher().complete([{"role": "system", "content": "JSON"}])
    assert calls[0]["reasoning_effort"] == "high" and calls[0]["response_format"] == {"type": "text"}
    assert teacher_environment({}, "dummy", "high")["CODEAGENTBENCH_PROTOCOL_REASONING"] == "high"


def test_drained_receipts_reject_missing_unknown_or_mismatched_jobs(tmp_path):
    (tmp_path / "logs").mkdir(); (tmp_path / "outcomes").mkdir()
    (tmp_path / "logs/t-protocol-0-rollout.log").write_text("TEACHER_RECEIPT")
    with pytest.raises(ValueError, match="unknown launched"):
        verify_drained_receipts(tmp_path)
    run = tmp_path / "runs/t-protocol-0"; run.mkdir(parents=True)
    (run / "summary.json").write_text('{"task_id":"t","status":"completed"}')
    with pytest.raises(ValueError, match="safely drained"):
        verify_drained_receipts(tmp_path)
    outcome = tmp_path / "outcomes/t.json"
    outcome.write_text('{"run_id":"wrong","status":"rejected"}')
    with pytest.raises(ValueError, match="mismatched"):
        verify_drained_receipts(tmp_path)
    outcome.write_text('{"run_id":"t-protocol-0","status":"rejected"}')
    assert verify_drained_receipts(tmp_path)["launched_jobs"] == 1
