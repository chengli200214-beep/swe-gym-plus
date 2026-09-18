"""A small JSON action protocol and DeepSeek-compatible model adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        """Return one assistant response."""


class ScriptedModel:
    """Deterministic model for contract tests and local harness demos."""

    def __init__(self, responses: list[str | dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        if not self.responses:
            return ModelResponse(json.dumps({"done": True, "message": "script exhausted"}))
        response = self.responses.pop(0)
        return ModelResponse(json.dumps(response) if isinstance(response, dict) else response, completion_tokens=32)


class DeepSeekModel:
    """OpenAI-compatible DeepSeek adapter with usage captured from the response."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the API baseline")

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DeepSeek adapter requires `pip install codeagentbench[api]`") from exc
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        text = payload["choices"][0]["message"]["content"]
        return ModelResponse(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), float(payload.get("cost_usd", 0.0)))
