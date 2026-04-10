from __future__ import annotations

import re
import time
from time import perf_counter
from typing import Any

from src.generators.base import BaseGenerator, GeneratorResult
from src.schemas import TokenUsage


genai = None
genai_types = None
legacy_genai = None


def _load_new_sdk() -> tuple[Any | None, Any | None]:
    try:  # pragma: no cover - import paths vary by environment
        from google import genai as imported_genai
        from google.genai import types as imported_types

        return imported_genai, imported_types
    except Exception:  # pragma: no cover - optional dependency behavior
        return None, None


def _load_legacy_sdk() -> Any | None:
    try:  # pragma: no cover - fallback SDK
        import google.generativeai as imported_legacy_genai

        return imported_legacy_genai
    except Exception:
        return None


class GeminiGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        api_key: str,
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

    def _generate_with_new_sdk(self, system_prompt: str, user_prompt: str, api_key: str, attempt: int) -> GeneratorResult:
        client = genai.Client(api_key=api_key)
        started = perf_counter()
        response = client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
            ),
        )
        latency = perf_counter() - started
        usage_meta = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
            total_tokens=getattr(usage_meta, "total_token_count", None),
        )
        answer = str(getattr(response, "text", "") or "").strip()
        metadata: dict[str, Any] = {"sdk": "google-genai", "attempt": attempt}
        return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)

    def _generate_with_legacy_sdk(self, system_prompt: str, user_prompt: str, api_key: str, attempt: int) -> GeneratorResult:
        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )
        started = perf_counter()
        response = model.generate_content(
            user_prompt,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_tokens,
            },
            request_options={"timeout": self.timeout_seconds},
        )
        latency = perf_counter() - started
        usage_meta = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", None),
            output_tokens=getattr(usage_meta, "candidates_token_count", None),
            total_tokens=getattr(usage_meta, "total_token_count", None),
        )
        answer = str(getattr(response, "text", "") or "").strip()
        metadata: dict[str, Any] = {"sdk": "google-generativeai", "attempt": attempt}
        return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)

    def _retry_delay_seconds(self, attempt_index: int, exc: Exception) -> float:
        message = str(exc)
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.0, min(self.max_retry_backoff_seconds, float(match.group(1))))
            except ValueError:
                pass
        base = self.retry_backoff_seconds * (2 ** attempt_index)
        return min(self.max_retry_backoff_seconds, base)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        text = str(exc).lower()
        retry_markers = (
            "429",
            "quota exceeded",
            "rate limit",
            "resource exhausted",
            "temporarily unavailable",
            "deadline exceeded",
            "timed out",
            "timeout",
            "connection",
            "internal",
            "503",
            "500",
        )
        return any(marker in text for marker in retry_markers)

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        global genai, genai_types, legacy_genai

        if not self.api_keys:
            raise RuntimeError("Gemini API key is missing.")
        if genai is None or genai_types is None:
            genai, genai_types = _load_new_sdk()
        if legacy_genai is None:
            legacy_genai = _load_legacy_sdk()
        if genai is None or genai_types is None:
            if legacy_genai is None:
                raise RuntimeError(
                    "Neither google-genai nor google-generativeai is installed; cannot call Gemini."
                )

        last_exc: Exception | None = None
        total_attempts = max(1, self.retries) * max(1, len(self.api_keys))
        for attempt_index in range(total_attempts):
            api_key = self.api_keys[attempt_index % len(self.api_keys)]
            try:
                if genai is not None and genai_types is not None:
                    return self._generate_with_new_sdk(system_prompt, user_prompt, api_key=api_key, attempt=attempt_index + 1)
                if legacy_genai is not None:
                    return self._generate_with_legacy_sdk(system_prompt, user_prompt, api_key=api_key, attempt=attempt_index + 1)
                raise RuntimeError("Gemini SDK became unavailable at runtime.")
            except Exception as exc:
                last_exc = exc
                if attempt_index >= total_attempts - 1 or not self._is_retryable(exc):
                    break
                time.sleep(self._retry_delay_seconds(attempt_index, exc))

        if last_exc is not None:
            raise RuntimeError(str(last_exc))
        raise RuntimeError("Gemini generation failed without a captured exception.")
