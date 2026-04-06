from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import httpx

from src.schemas import TokenUsage


class GeneratorResult:
    def __init__(self, answer: str, token_usage: TokenUsage | None = None, metadata: dict[str, Any] | None = None, latency_seconds: float | None = None) -> None:
        self.answer = answer
        self.token_usage = token_usage or TokenUsage()
        self.metadata = metadata or {}
        self.latency_seconds = latency_seconds


class BaseGenerator(ABC):
    model_name: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        """Generate a single answer."""


class OllamaGeneratorBase(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        endpoint: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        num_ctx: int,
        top_p: float,
        repeat_penalty: float,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
            },
        }
        started = perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
        latency = perf_counter() - started
        answer = str(body.get("response", "") or "").strip()
        usage = TokenUsage(
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            total_tokens=(body.get("prompt_eval_count") or 0) + (body.get("eval_count") or 0) or None,
        )
        metadata = {
            "done_reason": body.get("done_reason"),
            "total_duration": body.get("total_duration"),
            "eval_duration": body.get("eval_duration"),
        }
        return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)
