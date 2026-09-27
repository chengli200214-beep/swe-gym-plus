import json

import httpx
import pytest

from scripts.fast_finish import ThinkingTeacher, exhausted_tasks, qualified_paths, snapshot_exports


def response(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))


def test_thinking_teacher_uses_official_pro_and_counts_all_completion_usage(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setenv("DEEPSEEK_MIN_BALANCE_CNY", "0")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: response(200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    requests = []
    def post(*a, **k):
        requests.append((a, k))
        return response(200, {"choices": [{"message": {"content": '{"done":true}', "reasoning_content": "not exported"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 50}})
    monkeypatch.setattr(httpx, "post", post)
    result = ThinkingTeacher().complete([{"role": "user", "content": "fix"}])
    assert requests[0][0][0] == "https://api.deepseek.com/chat/completions"
    assert requests[0][1]["json"]["model"] == "deepseek-v4-pro"
    assert requests[0][1]["json"]["thinking"] == {"type": "enabled"}
    assert result.completion_tokens == 50 and result.estimated_cost_cny is None
    assert "not exported" not in result.text


def test_thinking_teacher_does_not_replay_unknown_paid_outcome(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-test-key")
    monkeypatch.setenv("DEEPSEEK_MIN_BALANCE_CNY", "0")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: response(200, {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]}))
    calls = []
    def post(*a, **k):
        calls.append(1)
        raise httpx.ReadTimeout("unknown")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(RuntimeError, match="outcome unknown"):
        ThinkingTeacher().complete([])
    assert len(calls) == 1


def test_retry_pool_requires_both_exhausted_rejections_and_train_admission(tmp_path):
    (tmp_path / "outcomes").mkdir(); (tmp_path / "quality").mkdir()
    for task, statuses in {"ready": ["rejected", "rejected"], "active": ["rejected"], "unknown": ["rejected", "interrupted"], "heldout": ["rejected", "rejected"]}.items():
        for i, status in enumerate(statuses):
            (tmp_path / "outcomes" / (task + "-fast-" + str(i) + ".json")).write_text(json.dumps({"status": status}))
        (tmp_path / "quality" / (task + ".json")).write_text('{"admitted":true}')
    assert exhausted_tasks(tmp_path, ["ready", "active", "unknown"]) == ["ready"]


def test_union_snapshot_deduplicates_and_rechecks_clean_completion(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    row = {"task_id": "x", "agent_status": "completed", "evaluation_verdict": "passed", "messages": [{"role": "user", "content": '{"instance_id":"x"}'}, {"role": "assistant", "content": '{"done":true}'}]}
    for root in roots:
        (root / "exports").mkdir(parents=True)
        (root / "exports/x.jsonl").write_text(json.dumps(row))
    paths = qualified_paths(roots, ["x"])
    assert len(paths) == 1 and paths["x"] == roots[0] / "exports/x.jsonl"
    exports = snapshot_exports(tmp_path / "snapshot", paths, ["x"])
    assert json.loads((exports / "x.jsonl").read_text()) == row
    assert json.loads((tmp_path / "snapshot/sources.json").read_text())["distinct_tasks"] == 1
