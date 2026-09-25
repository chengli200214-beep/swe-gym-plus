from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from codeagentbench.adapters.model import DeepSeekModel


def test_deepseek_request_has_bounded_output_and_non_thinking_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def post(url: str, *, headers: dict, json: dict, timeout: float):
        captured.update(url=url, headers=headers, request=json, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": '{"done": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, RequestError=Exception))
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
    response = DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])

    assert captured["request"]["model"] == "deepseek-flash"
    assert captured["request"]["thinking"] == {"type": "disabled"}
    assert captured["request"]["response_format"] == {"type": "json_object"}
    assert captured["request"]["max_tokens"] == 4096
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 4
    assert response.estimated_cost_cny == pytest.approx((12 * 2 + 4 * 8) / 1_000_000)
    assert response.cost_usd == 0.0


def test_deepseek_output_limit_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    model = DeepSeekModel(api_key="test-key")
    assert model.max_output_tokens == 1024
    assert model.thinking == "enabled"


def test_deepseek_retries_one_empty_response_and_counts_both_uses(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict] = []
    contents = ["", '{"command": "pwd", "done": false}']

    def post(url: str, *, headers: dict, json: dict, timeout: float):
        requests.append(json.copy())
        content = contents.pop(0)
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, RequestError=Exception))
    response = DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])

    assert len(requests) == 2
    assert requests[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert "previous response was empty" in requests[1]["messages"][-1]["content"]
    assert response.text == '{"command": "pwd", "done": false}'
    assert response.prompt_tokens == 20
    assert response.completion_tokens == 10
    assert response.estimated_cost_cny == pytest.approx((20 * 2 + 10 * 8) / 1_000_000)


def test_deepseek_counts_cached_input_at_separate_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, *, headers: dict, json: dict, timeout: float):
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": '{"done": true}'}}],
                "usage": {"prompt_tokens": 100, "prompt_cache_hit_tokens": 60,
                          "prompt_cache_miss_tokens": 40, "completion_tokens": 10},
            },
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, RequestError=Exception))
    response = DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])
    assert response.estimated_cost_cny == pytest.approx((60 * 0.04 + 40 * 2 + 10 * 8) / 1_000_000)


def test_deepseek_missing_usage_has_no_cost_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, *, headers: dict, json: dict, timeout: float):
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"done": true}'}}]},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, RequestError=Exception))
    response = DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])
    assert response.estimated_cost_cny is None


def test_deepseek_balance_floor_checks_before_paid_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MIN_BALANCE_CNY", "5.40")
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "64")
    calls: list[str] = []

    def get(url: str, *, headers: dict, timeout: float):
        calls.append("balance")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "35.40"}]},
        )

    def post(url: str, *, headers: dict, json: dict, timeout: float):
        calls.append("paid")
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '{"done": true}'}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=get, post=post, RequestError=Exception))
    DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])
    assert calls == ["balance", "paid"]


def test_deepseek_balance_floor_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MIN_BALANCE_CNY", "5.40")
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "64")
    calls: list[str] = []

    def get(url: str, *, headers: dict, timeout: float):
        calls.append("balance")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "6.00"}]},
        )

    def post(*args, **kwargs):
        calls.append("paid")
        raise AssertionError("paid request should not be sent")

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=get, post=post, RequestError=Exception))
    with pytest.raises(RuntimeError, match="too close"):
        DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])
    assert calls == ["balance"]


def test_deepseek_balance_floor_rejects_large_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MIN_BALANCE_CNY", "5.40")
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "64")
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(RequestError=Exception))
    with pytest.raises(RuntimeError, match="100 KB"):
        DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "x" * 100_000}])


def test_deepseek_stops_after_two_empty_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def post(url: str, *, headers: dict, json: dict, timeout: float):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": None}}], "usage": {"completion_tokens": 3}},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post, RequestError=Exception))
    response = DeepSeekModel(api_key="test-key").complete([{"role": "user", "content": "hi"}])

    assert calls == 2
    assert response.text == ""
    assert response.completion_tokens == 6


@pytest.mark.parametrize("name,value", [("DEEPSEEK_MAX_OUTPUT_TOKENS", "0"), ("DEEPSEEK_THINKING", "maybe")])
def test_deepseek_rejects_invalid_limits(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        DeepSeekModel(api_key="test-key")
