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


def test_deepseek_output_limit_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    model = DeepSeekModel(api_key="test-key")
    assert model.max_output_tokens == 1024
    assert model.thinking == "enabled"


@pytest.mark.parametrize("name,value", [("DEEPSEEK_MAX_OUTPUT_TOKENS", "0"), ("DEEPSEEK_THINKING", "maybe")])
def test_deepseek_rejects_invalid_limits(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        DeepSeekModel(api_key="test-key")
