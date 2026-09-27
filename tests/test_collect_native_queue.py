import json

import httpx
import pytest

from scripts.collect_native_queue import NativeTeacher
from codeagentbench.adapters.action import parse_action


def response(message, finish="tool_calls"):
    return httpx.Response(200, json={"id": "r1", "model": "deepseek-v4-pro",
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 100,
                  "completion_tokens_details": {"reasoning_tokens": 70}}},
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))


def native(name, arguments, call_id="call-real"):
    return {"role": "assistant", "content": None, "reasoning_content": "private reasoning",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest.fixture
def setup_teacher(monkeypatch, tmp_path):
    journal = tmp_path / "actions.jsonl"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setenv("CODEAGENTBENCH_NATIVE_JOURNAL", str(journal))
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200,
        json={"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "30"}]},
        request=httpx.Request("GET", "https://api.deepseek.com/user/balance")))
    return journal, [{"role": "system", "content": 'Return exactly JSON: {"command":"...", "done":false, "message":"..."}. '},
                     {"role": "user", "content": '{"instance_id":"public-task","issue":"fix"}'}]


def test_native_preserves_real_receipt_and_reasoning_despite_runtime_compression(setup_teacher, monkeypatch, capsys):
    journal, messages = setup_teacher
    requests = []
    replies = iter([response(native("execute_shell", {"command": "pwd"})),
                    response(native("finish_task", {"message": "tested"}))])
    def post(*a, **k):
        requests.append(json.loads(json.dumps(k["json"]))); return next(replies)
    monkeypatch.setattr(httpx, "post", post)
    teacher = NativeTeacher()
    assert parse_action(teacher.complete(messages).text).command == "pwd"
    journal.write_text(json.dumps({"type": "receipt", "action_id": "a0", "command": "pwd",
        "exit_code": 0, "stdout": "/workspace", "stderr": "", "timed_out": False}) + "\n")
    compressed = [messages[0], messages[1], {"role": "user", "content": "Evidence summary after context compression"}]
    teacher.request_token_bound(compressed)
    assert parse_action(teacher.complete(compressed).text).done
    prior = requests[1]["messages"]
    assert prior[2]["reasoning_content"] == "private reasoning"
    assert prior[3]["role"] == "tool" and prior[3]["tool_call_id"] == "call-real"
    assert json.loads(prior[3]["content"])["stdout"] == "/workspace"
    assert sum(m["role"] == "tool" for m in prior) == 1
    assert "tool_choice" not in requests[0] and requests[0]["reasoning_effort"] == "high"
    assert "private reasoning" not in capsys.readouterr().out


@pytest.mark.parametrize("reply", [
    response({"content": "<tool_calls><result>fake output</result></tool_calls>"}, "stop"),
    response(native("unknown", {"command": "pwd"})),
    response(native("execute_shell", {"command": "pwd", "done": True})),
    response(native("execute_shell", {"command": "pwd"}), "length"),
    response(native("finish_task", {"message": 123})),
])
def test_native_invalid_response_fails_closed_without_mining_commands(setup_teacher, monkeypatch, reply):
    _, messages = setup_teacher
    monkeypatch.setattr(httpx, "post", lambda *a, **k: reply)
    teacher = NativeTeacher(); result = teacher.complete(messages)
    assert result.completion_tokens == 100 and teacher.pending is None
    with pytest.raises(ValueError): parse_action(result.text)


def test_native_refuses_finish_mixed_with_unobserved_commands(setup_teacher, monkeypatch):
    _, messages = setup_teacher
    message = native("execute_shell", {"command": "pwd"})
    message["tool_calls"] += native("finish_task", {"message": "fake done"}, "call-2")["tool_calls"]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: response(message))
    teacher = NativeTeacher()
    with pytest.raises(ValueError): parse_action(teacher.complete(messages).text)
    assert teacher.pending is None


def test_native_refuses_unmatched_receipt_before_next_paid_call(setup_teacher, monkeypatch):
    journal, messages = setup_teacher
    calls = []
    def post(*a, **k): calls.append(1); return response(native("execute_shell", {"command": "pwd"}))
    monkeypatch.setattr(httpx, "post", post)
    teacher = NativeTeacher(); teacher.complete(messages)
    journal.write_text(json.dumps({"type": "receipt", "action_id": "a0", "command": "ls"}) + "\n")
    with pytest.raises(RuntimeError, match="matching real receipt"): teacher.complete(messages)
    assert len(calls) == 1


def test_native_unknown_transport_is_never_replayed(setup_teacher, monkeypatch):
    _, messages = setup_teacher; calls = []
    def post(*a, **k): calls.append(1); raise httpx.ReadTimeout("unknown")
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(RuntimeError, match="outcome unknown"): NativeTeacher().complete(messages)
    assert len(calls) == 1

def test_serial_queue_executes_known_calls_once_and_waits_for_actual_receipts(setup_teacher, monkeypatch):
    journal, messages = setup_teacher
    first = native("execute_shell", {"command": "pwd"}, "call-1")
    first["tool_calls"] += native("execute_shell", {"command": "ls"}, "call-2")["tool_calls"]
    replies = iter([response(first), response(native("finish_task", {"message": "tested"}, "call-3"))])
    requests = []
    def post(*a, **k):
        requests.append(json.loads(json.dumps(k["json"]))); return next(replies)
    monkeypatch.setattr(httpx, "post", post)
    teacher = NativeTeacher()
    first_action = teacher.complete(messages)
    assert parse_action(first_action.text).command == "pwd" and first_action.completion_tokens == 100
    rows = [{"type": "receipt", "action_id": "a0", "command": "pwd",
             "exit_code": 0, "stdout": "/workspace", "stderr": "", "timed_out": False}]
    journal.write_text(json.dumps(rows[0]) + "\n")
    assert teacher.request_token_bound(messages) == 1
    second_action = teacher.complete(messages)
    assert parse_action(second_action.text).command == "ls"
    assert second_action.prompt_tokens == second_action.completion_tokens == 0 and len(requests) == 1
    rows.append({"type": "receipt", "action_id": "a1", "command": "ls",
                 "exit_code": 2, "stdout": "", "stderr": "real failure", "timed_out": False})
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert parse_action(teacher.complete(messages).text).done and len(requests) == 2
    tools = [m for m in requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call-1", "call-2"]
    assert json.loads(tools[1]["content"])["exit_code"] == 2
    assert requests[1]["messages"][2]["reasoning_content"] == "private reasoning"


def test_multi_call_validation_is_atomic(setup_teacher, monkeypatch):
    _, messages = setup_teacher
    message = native("execute_shell", {"command": "pwd"}, "valid")
    message["tool_calls"] += native("execute_shell", {"command": 123}, "invalid")["tool_calls"]
    monkeypatch.setattr(httpx, "post", lambda *a, **k: response(message))
    teacher = NativeTeacher()
    with pytest.raises(ValueError): parse_action(teacher.complete(messages).text)
    assert not teacher.queued and teacher.pending is None and len(teacher.history) == 2
