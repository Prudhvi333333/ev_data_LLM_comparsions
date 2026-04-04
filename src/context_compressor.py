from __future__ import annotations

from types import SimpleNamespace

from src.indexer import _token_ngrams, _tokenize


class ContextCompressor:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config

    def compress(self, question: str, chunks: list[str], max_tokens: int = 2000) -> str:
        if not chunks:
            return "RELEVANT KNOWLEDGE BASE EXCERPTS:\n"

        query_tokens = _tokenize(question)
        query_token_set = set(query_tokens)
        query_ngram_set = set(_token_ngrams(query_tokens))

        def relevance_score(chunk: str) -> float:
            chunk_tokens = _tokenize(chunk)
            if not query_token_set:
                return 0.0
            chunk_token_set = set(chunk_tokens)
            chunk_ngram_set = set(_token_ngrams(chunk_tokens))
            token_overlap = len(query_token_set & chunk_token_set) / len(query_token_set)
            ngram_overlap = len(query_ngram_set & chunk_ngram_set) / max(1, len(query_ngram_set))
            return 0.7 * token_overlap + 0.3 * ngram_overlap

        ranked_chunks = sorted(chunks, key=relevance_score, reverse=True)
        selected_chunks: list[str] = []
        estimated_tokens = 0.0

        for chunk in ranked_chunks:
            chunk_tokens = len(chunk.split()) * 1.3
            if selected_chunks and estimated_tokens + chunk_tokens > max_tokens:
                continue
            selected_chunks.append(chunk)
            estimated_tokens += chunk_tokens

        if not selected_chunks:
            selected_chunks = [ranked_chunks[0]]

        return "RELEVANT KNOWLEDGE BASE EXCERPTS:\n" + "\n---\n".join(selected_chunks)
