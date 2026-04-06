from __future__ import annotations

from typing import Any

__all__ = ["GeminiGenerator", "GemmaGenerator", "QwenGenerator"]


def __getattr__(name: str) -> Any:
    if name == "GeminiGenerator":
        from src.generators.gemini_generator import GeminiGenerator

        return GeminiGenerator
    if name == "GemmaGenerator":
        from src.generators.gemma_generator import GemmaGenerator

        return GemmaGenerator
    if name == "QwenGenerator":
        from src.generators.qwen_generator import QwenGenerator

        return QwenGenerator
    raise AttributeError(name)
