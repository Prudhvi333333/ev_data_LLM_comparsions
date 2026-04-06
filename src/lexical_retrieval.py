from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import pickle
import string

from src.schemas import KnowledgeChunk
from src.utils.files import ensure_directory
from src.utils.logger import get_logger


logger = get_logger("lexical_retrieval")

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency behavior
    BM25Okapi = None


_TRANSLATION_TABLE = str.maketrans({char: " " for char in string.punctuation})
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "for",
    "from",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in str(text).lower().translate(_TRANSLATION_TABLE).split()
        if token and token not in _STOPWORDS
    ]


class SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avg_doc_len = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        for doc in corpus_tokens:
            tf: dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in tf:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.num_docs = len(corpus_tokens)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc_idx, tf in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[doc_idx]
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
                score += idf * ((freq * (self.k1 + 1)) / denom)
            scores.append(score)
        return scores


class LexicalIndex:
    def __init__(self, chunks: list[KnowledgeChunk], bm25: Any) -> None:
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def build_or_load(cls, chunks: list[KnowledgeChunk], index_dir: Path) -> "LexicalIndex":
        ensure_directory(index_dir)
        path = index_dir / "lexical.pkl"
        if path.exists():
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            loaded_chunks = [KnowledgeChunk.model_validate(row) for row in payload["chunks"]]
            logger.info("Loaded lexical index from %s", path)
            return cls(loaded_chunks, payload["bm25"])

        corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens) if BM25Okapi is not None else SimpleBM25(corpus_tokens)
        with path.open("wb") as handle:
            pickle.dump({"chunks": [chunk.model_dump() for chunk in chunks], "bm25": bm25}, handle)
        logger.info("Built lexical index at %s", path)
        return cls(chunks, bm25)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ordered = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[tuple[KnowledgeChunk, float]] = []
        for idx, score in ordered:
            chunk = self.chunks[idx]
            if allowed_chunk_ids is not None and chunk.chunk_id not in allowed_chunk_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results
