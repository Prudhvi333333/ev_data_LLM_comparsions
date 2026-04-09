from __future__ import annotations

import re
import time
from time import perf_counter
from typing import Any

import httpx

from src.generators.base import BaseGenerator, GeneratorResult
from src.schemas import TokenUsage


class ChatGPTGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        endpoint: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        top_p: float,
        retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        max_retry_backoff_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.api_keys = self._parse_api_keys(api_key)
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.top_p = top_p
        self.retries = max(1, int(retries))
        self.retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
        self.max_retry_backoff_seconds = max(1.0, float(max_retry_backoff_seconds))

    @staticmethod
    def _parse_api_keys(raw: str) -> list[str]:
        return [token.strip() for token in re.split(r"[,\n;]+", str(raw or "")) if token.strip()]

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

    @staticmethod
    def _extract_answer(body: dict[str, Any]) -> str:
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        if not self.api_keys:
            raise RuntimeError("ChatGPT/OpenAI API key is missing.")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        last_exc: Exception | None = None
        total_attempts = max(1, self.retries) * max(1, len(self.api_keys))

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt_index in range(total_attempts):
                api_key = self.api_keys[attempt_index % len(self.api_keys)]
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                started = perf_counter()
                response: httpx.Response | None = None
                try:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    latency = perf_counter() - started

                    usage_raw = body.get("usage") or {}
                    usage = TokenUsage(
                        input_tokens=usage_raw.get("prompt_tokens"),
                        output_tokens=usage_raw.get("completion_tokens"),
                        total_tokens=usage_raw.get("total_tokens"),
                    )
                    answer = self._extract_answer(body)
                    metadata = {
                        "provider": "openai",
                        "model": body.get("model", self.model_name),
                        "response_id": body.get("id"),
                        "finish_reason": ((body.get("choices") or [{}])[0].get("finish_reason")),
                        "attempt": attempt_index + 1,
                    }
                    return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    if attempt_index >= total_attempts - 1:
                        break
                    time.sleep(self._retry_delay_seconds(attempt_index))
                    continue
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    status_code = exc.response.status_code
                    retryable = status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                    if not retryable or attempt_index >= total_attempts - 1:
                        raise
                    time.sleep(self._retry_delay_seconds(attempt_index, response=exc.response))
                    continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI generation failed without a captured exception.")
