"""A small JSON action protocol and DeepSeek-compatible model adapter."""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    estimated_cost_cny: float | None = None


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
        floor = os.getenv("DEEPSEEK_MIN_BALANCE_CNY")
        try:
            self.min_balance_cny = Decimal(floor) if floor is not None else None
        except InvalidOperation as exc:
            raise ValueError("DEEPSEEK_MIN_BALANCE_CNY must be a nonnegative number") from exc
        if self.min_balance_cny is not None:
            if not self.min_balance_cny.is_finite() or self.min_balance_cny < 0:
                raise ValueError("DEEPSEEK_MIN_BALANCE_CNY must be a nonnegative number")
            if self.base_url != "https://api.deepseek.com" or self.model != "deepseek-flash":
                raise ValueError("CNY balance guard currently supports only the official deepseek-flash API")
            if self.max_output_tokens > 1024:
                raise ValueError("CNY balance guard requires DEEPSEEK_MAX_OUTPUT_TOKENS <= 1024")

    def _check_balance(self, httpx: Any) -> None:
        if self.min_balance_cny is None:
            return
        response = httpx.get(
            f"{self.base_url}/user/balance",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        balances = [entry for entry in payload.get("balance_infos", []) if entry.get("currency") == "CNY"]
        if not payload.get("is_available") or len(balances) != 1:
            raise RuntimeError("DeepSeek CNY balance is unavailable; refusing paid request")
        balance = Decimal(str(balances[0]["total_balance"]))
        if not balance.is_finite() or balance <= self.min_balance_cny + Decimal("1.00"):
            raise RuntimeError("DeepSeek CNY balance is too close to the configured floor")

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
        estimated_cost_cny = 0.0
        cost_estimate_available = True
        for empty_attempt in range(2):
            response = None
            for attempt in range(4):
                if self.min_balance_cny is not None:
                    if len(json.dumps(request, ensure_ascii=False).encode("utf-8")) > 100_000:
                        raise RuntimeError("DeepSeek request exceeds 100 KB guarded input limit")
                    self._check_balance(httpx)
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
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            prompt_tokens += prompt
            completion_tokens += completion
            if self.model == "deepseek-flash":
                if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                    cost_estimate_available = False
                hit = int(usage.get("prompt_cache_hit_tokens", 0))
                miss = int(usage.get("prompt_cache_miss_tokens", prompt - hit))
                if hit < 0 or miss < 0 or hit + miss > prompt:
                    raise ValueError("invalid DeepSeek cache token usage")
                # Conservative peak-hour CNY estimate, not a provider invoice or hard cap.
                estimated_cost_cny += ((prompt - hit) * 2.0 + hit * 0.04 + completion * 8.0) / 1_000_000
            text = payload["choices"][0]["message"].get("content") or ""
            if text.strip() or empty_attempt:
                return ModelResponse(
                    text, prompt_tokens, completion_tokens,
                    estimated_cost_cny=estimated_cost_cny if self.model == "deepseek-flash" and cost_estimate_available else None,
                )
            request["messages"] = messages + [
                {"role": "user", "content": "The previous response was empty. Return one non-empty JSON action object."}
            ]
        raise AssertionError("unreachable")
