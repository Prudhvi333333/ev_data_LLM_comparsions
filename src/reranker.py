from __future__ import annotations

from typing import Iterable

from src.schemas import KnowledgeChunk
from src.utils.logger import get_logger


logger = get_logger("reranker")

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency behavior
    CrossEncoder = None


class HybridReranker:
    def __init__(self, model_name: str, allow_remote_downloads: bool) -> None:
        self.model_name = model_name
        self.cross_encoder = None
        if CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(model_name, local_files_only=not allow_remote_downloads)
            except TypeError:
                self.cross_encoder = CrossEncoder(model_name)
            except Exception as exc:
                logger.warning("Cross-encoder unavailable, falling back to lexical reranker: %s", exc)
                self.cross_encoder = None

    @staticmethod
    def _fallback_score(query: str, chunk: KnowledgeChunk, filter_values: set[str]) -> float:
        query_tokens = set(query.lower().split())
        chunk_tokens = set(chunk.text.lower().split())
        overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
        metadata_bonus = 0.0
        if filter_values:
            haystack = " ".join(str(value) for value in chunk.metadata.values()).lower()
            if any(value in haystack for value in filter_values):
                metadata_bonus += 0.25
        return overlap + metadata_bonus

    def rerank(
        self,
        query: str,
        candidates: list[KnowledgeChunk],
        top_k: int,
        filter_values: Iterable[str] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        if not candidates:
            return []
        if self.cross_encoder is not None:
            try:
                scores = self.cross_encoder.predict([(query, chunk.text) for chunk in candidates])
                ordered = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
                return [(chunk, float(score)) for chunk, score in ordered[:top_k]]
            except Exception as exc:
                logger.warning("Cross-encoder rerank failed, falling back to lexical scoring: %s", exc)

        filters = {str(value).lower() for value in (filter_values or []) if str(value).strip()}
        ordered = sorted(
            ((chunk, self._fallback_score(query, chunk, filters)) for chunk in candidates),
            key=lambda item: item[1],
            reverse=True,
        )
        return ordered[:top_k]
