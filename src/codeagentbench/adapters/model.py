"""A small JSON action protocol and DeepSeek-compatible model adapter."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
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


class LocalHFModel:
    """Local Transformers model for offline base-vs-adapter comparisons."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        base_model_path: str | Path | None = None,
        max_new_tokens: int = 512,
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LocalHFModel requires the training dependencies") from exc

        model_path = str(model_path)
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("LocalHFModel with a LoRA checkpoint requires peft") from exc
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_path = str(base_model_path or config.get("base_model_name_or_path", ""))
            if not base_path:
                raise ValueError("LoRA checkpoint does not declare a base model")
            base = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
            self._model = PeftModel.from_pretrained(
                base,
                model_path,
                is_trainable=False,
                local_files_only=local_files_only,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self._model.eval()

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return ModelResponse(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
        )


class DeepSeekModel:
    """OpenAI-compatible DeepSeek adapter with usage captured from the response."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.max_output_tokens = int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "4096"))
        if self.max_output_tokens < 1:
            raise ValueError("DEEPSEEK_MAX_OUTPUT_TOKENS must be positive")
        self.thinking = os.getenv("DEEPSEEK_THINKING", "disabled")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DEEPSEEK_THINKING must be enabled or disabled")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the API baseline")

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DeepSeek adapter requires `pip install codeagentbench[api]`") from exc
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "thinking": {"type": self.thinking},
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_output_tokens,
        }
        prompt_tokens = completion_tokens = 0
        cost_usd = 0.0
        for empty_attempt in range(2):
            response = None
            for attempt in range(4):
                try:
                    response = httpx.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=request,
                        timeout=120.0,
                    )
                except httpx.RequestError:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
                    continue
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                    break
                time.sleep(2**attempt)
            assert response is not None
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage", {})
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            cost_usd += float(payload.get("cost_usd", 0.0))
            text = payload["choices"][0]["message"].get("content") or ""
            if text.strip() or empty_attempt:
                return ModelResponse(text, prompt_tokens, completion_tokens, cost_usd)
            request["messages"] = messages + [
                {"role": "user", "content": "The previous response was empty. Return one non-empty JSON action object."}
            ]
        raise AssertionError("unreachable")
