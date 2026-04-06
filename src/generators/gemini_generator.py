from __future__ import annotations

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
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.top_p = top_p

    def _generate_with_new_sdk(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        client = genai.Client(api_key=self.api_key)
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
        metadata: dict[str, Any] = {"sdk": "google-genai"}
        return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)

    def _generate_with_legacy_sdk(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        legacy_genai.configure(api_key=self.api_key)
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
        metadata: dict[str, Any] = {"sdk": "google-generativeai"}
        return GeneratorResult(answer=answer, token_usage=usage, metadata=metadata, latency_seconds=latency)

    def generate(self, system_prompt: str, user_prompt: str) -> GeneratorResult:
        global genai, genai_types, legacy_genai

        if not self.api_key:
            raise RuntimeError("Gemini API key is missing.")
        if genai is None or genai_types is None:
            genai, genai_types = _load_new_sdk()
        if genai is not None and genai_types is not None:
            return self._generate_with_new_sdk(system_prompt, user_prompt)

        if legacy_genai is None:
            legacy_genai = _load_legacy_sdk()
        if legacy_genai is not None:
            return self._generate_with_legacy_sdk(system_prompt, user_prompt)

        raise RuntimeError(
            "Neither google-genai nor google-generativeai is installed; cannot call Gemini."
        )
