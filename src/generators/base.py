from __future__ import annotations

import time
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
        retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        max_retry_backoff_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.retries = max(1, int(retries))
        self.retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self.max_retry_backoff_seconds = max(1.0, float(max_retry_backoff_seconds))

    def _retry_delay_seconds(self, attempt_index: int, response: httpx.Response | None = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, min(self.max_retry_backoff_seconds, float(retry_after)))
                except ValueError:
                    pass
        base = self.retry_backoff_seconds * (2 ** attempt_index)
        return min(self.max_retry_backoff_seconds, base)

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        last_exc: Exception | None = None

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt_index in range(self.retries):
                scale = 2 ** attempt_index
                attempt_num_ctx = max(1024, int(self.num_ctx / scale))
                attempt_max_tokens = max(128, int(self.max_tokens / scale))
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": attempt_max_tokens,
                        "num_ctx": attempt_num_ctx,
                        "top_p": self.top_p,
                        "repeat_penalty": self.repeat_penalty,
                    },
                }
                started = perf_counter()
                response: httpx.Response | None = None
                try:
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
                        "attempt": attempt_index + 1,
                        "effective_num_ctx": attempt_num_ctx,
                        "effective_max_tokens": attempt_max_tokens,
                    }
                    return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    if attempt_index >= self.retries - 1:
                        break
                    time.sleep(self._retry_delay_seconds(attempt_index))
                    continue
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code
                    retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                    if not retryable or attempt_index >= self.retries - 1:
                        body_text = ""
                        try:
                            body_text = exc.response.text
                        except Exception:
                            body_text = ""
                        raise RuntimeError(
                            f"Ollama generate failed (status={status_code}) at {self.endpoint}: {body_text or str(exc)}"
                        ) from exc
                    time.sleep(self._retry_delay_seconds(attempt_index, response=exc.response))
                    continue

        if last_exc is not None:
            raise RuntimeError(f"Ollama generate failed after {self.retries} attempts: {last_exc}") from last_exc
        raise RuntimeError("Ollama generate failed without a captured exception.")
