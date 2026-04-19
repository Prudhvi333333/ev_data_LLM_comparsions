from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import hashlib
import math
import string

import numpy as np

from src.utils.logger import get_logger


logger = get_logger("embeddings")

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency behavior
    SentenceTransformer = None


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


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in str(text).lower().translate(_TRANSLATION_TABLE).split()
        if token and token not in _STOPWORDS
    ]


def _token_ngrams(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    ngrams = list(tokens)
    ngrams.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return ngrams


class BaseEmbedder(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Prepare any corpus-dependent state."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed documents as float32 numpy arrays."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""

    def dump_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, payload: dict[str, Any]) -> None:
        del payload


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, allow_remote_downloads: bool) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model_name = model_name
        self.name = f"sentence-transformer::{model_name}"
        try:
            self.model = SentenceTransformer(model_name, local_files_only=not allow_remote_downloads)
        except TypeError:
            self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        del texts

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension
        self.idf_by_bucket = np.ones(self.dimension, dtype=np.float32)
        self.name = f"hashing-tfidf::{dimension}"

    def fit(self, texts: list[str]) -> None:
        doc_freq = np.zeros(self.dimension, dtype=np.float32)
        total_docs = max(1, len(texts))
        for text in texts:
            seen = set()
            for token in _token_ngrams(_tokenize(text)):
                seen.add(self._bucket_for(token))
            for bucket in seen:
                doc_freq[bucket] += 1
        self.idf_by_bucket = np.array(
            [math.log((1 + total_docs) / (1 + freq)) + 1.0 for freq in doc_freq],
            dtype=np.float32,
        )

    def _bucket_for(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _token_ngrams(_tokenize(text)):
            vector[self._bucket_for(token)] += self.idf_by_bucket[self._bucket_for(token)]
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def dump_state(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "idf_by_bucket": self.idf_by_bucket.tolist(),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        self.dimension = int(payload["dimension"])
        self.idf_by_bucket = np.asarray(payload["idf_by_bucket"], dtype=np.float32)
        self.name = f"hashing-tfidf::{self.dimension}"


def build_embedder(
    model_name: str,
    allow_remote_downloads: bool,
    allow_hash_fallback: bool,
    dimension: int,
) -> BaseEmbedder:
    if SentenceTransformer is not None:
        try:
            logger.info("Loading embedding model %s", model_name)
            return SentenceTransformerEmbedder(model_name, allow_remote_downloads)
        except Exception as exc:
            logger.warning("Falling back from embedding model %s: %s", model_name, exc)
    if not allow_hash_fallback:
        raise RuntimeError(
            "Failed to load sentence-transformers embedder and hash fallback is disabled."
        )
    logger.warning("Using local hashing TF-IDF embedder fallback.")
    return HashingEmbedder(dimension=dimension)


def save_embedder_state(path: Path, embedder: BaseEmbedder) -> None:
    from src.utils.files import write_json

    payload = {
        "name": embedder.name,
        "state": embedder.dump_state(),
    }
    write_json(path, payload)


def load_embedder_state(path: Path, embedder: BaseEmbedder) -> BaseEmbedder:
    from src.utils.files import read_json

    payload = read_json(path)
    embedder.load_state(payload.get("state", {}))
    return embedder
