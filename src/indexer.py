from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hashlib
import math
import pickle
import string

from src.utils.logger import get_logger


logger = get_logger("indexer")

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
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
    "show",
    "the",
    "their",
    "to",
    "what",
    "which",
    "with",
}


def _tokenize(text: str) -> list[str]:
    tokens = [
        token
        for token in text.lower().translate(_TRANSLATION_TABLE).split()
        if token and token not in _STOPWORDS
    ]
    return tokens


def _token_ngrams(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    ngrams = list(tokens)
    ngrams.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return ngrams


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _field_matches(actual_value: Any, expected: Any) -> bool:
    actual = str(actual_value or "").lower()
    if isinstance(expected, dict):
        if "$in" in expected:
            return any(str(option).lower() in actual for option in expected["$in"])
        if "$eq" in expected:
            return str(expected["$eq"]).lower() in actual
    return str(expected).lower() in actual


def _matches_local_filter(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches_local_filter(metadata, clause) for clause in where["$and"])
    if "$or" in where:
        return any(_matches_local_filter(metadata, clause) for clause in where["$or"])
    for key, expected in where.items():
        if not _field_matches(metadata.get(key, ""), expected):
            return False
    return True


def _to_chroma_where_filter(where: dict[str, Any] | None) -> dict[str, Any] | None:
    if not where:
        return None
    if any(key.startswith("$") for key in where):
        return where
    if len(where) == 1:
        key, value = next(iter(where.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in where.items()]}


class _LocalTfidfEmbeddingModel:
    def __init__(self, dimension: int = 4096) -> None:
        self.dimension = dimension
        self.idf_by_bucket = [1.0] * self.dimension

    def fit(self, texts: list[str]) -> None:
        doc_freq = [0] * self.dimension
        total_docs = max(1, len(texts))

        for text in texts:
            seen_buckets = set()
            for token in _token_ngrams(_tokenize(text)):
                bucket = self._bucket_for(token)
                seen_buckets.add(bucket)
            for bucket in seen_buckets:
                doc_freq[bucket] += 1

        self.idf_by_bucket = [
            math.log((1 + total_docs) / (1 + freq)) + 1.0
            for freq in doc_freq
        ]

    def encode(self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True) -> list[list[float]]:
        del batch_size, show_progress_bar
        embeddings = [self._embed_one(text) for text in texts]
        if not normalize_embeddings:
            return embeddings
        normalized: list[list[float]] = []
        for embedding in embeddings:
            norm = math.sqrt(sum(value * value for value in embedding)) or 1.0
            normalized.append([value / norm for value in embedding])
        return normalized

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _token_ngrams(_tokenize(text)):
            bucket = self._bucket_for(token)
            vector[bucket] += self.idf_by_bucket[bucket]
        return vector

    def _bucket_for(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension


class _SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = {}
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avg_doc_len = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
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


@dataclass
class SemanticCandidate:
    id: str
    text: str
    metadata: dict[str, Any]
    semantic_score: float


class KBIndexer:
    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.collection_name = "kb_collection"
        self.chroma_path = Path(config.paths.chroma)
        self.bm25_path = self.chroma_path / "bm25.pkl"
        self.local_store_path = self.chroma_path / "local_vector_store.pkl"
        self.documents: list[dict[str, Any]] = []
        self.doc_ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []
        self.tokenized_documents: list[list[str]] = []
        self.bm25 = None

        self.embedding_model = self._load_embedding_model()
        self.chroma_client = None
        self.collection = None
        self._init_vector_store()

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            logger.warning("sentence-transformers unavailable; using local TF-IDF hash embeddings.")
            return _LocalTfidfEmbeddingModel()
        try:
            return SentenceTransformer(self.config.retrieval.embedding_model)
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s); using local TF-IDF hash embeddings.",
                self.config.retrieval.embedding_model,
                exc,
            )
            return _LocalTfidfEmbeddingModel()

    def _init_vector_store(self) -> None:
        if chromadb is None:
            logger.warning("chromadb unavailable; using local pickle vector store.")
            return
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
        except Exception as exc:
            logger.warning("Failed to initialize ChromaDB (%s); using local pickle vector store.", exc)
            self.chroma_client = None
            self.collection = None

    @property
    def use_chroma(self) -> bool:
        return self.chroma_client is not None and self.collection is not None

    def _embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [list(map(float, emb)) for emb in self.embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)]

    def _save_local_store(self) -> None:
        payload = {
            "doc_ids": self.doc_ids,
            "texts": self.texts,
            "metadatas": self.metadatas,
            "embeddings": self.embeddings,
        }
        with self.local_store_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def _load_local_store(self) -> None:
        if not self.local_store_path.exists():
            return
        with self.local_store_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.doc_ids = list(payload.get("doc_ids", []))
        self.texts = list(payload.get("texts", []))
        self.metadatas = list(payload.get("metadatas", []))
        self.embeddings = list(payload.get("embeddings", []))
        self.documents = [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(self.doc_ids, self.texts, self.metadatas)
        ]
        self.tokenized_documents = [_tokenize(text) for text in self.texts]

    def _load_bm25(self) -> None:
        if not self.bm25_path.exists():
            return
        with self.bm25_path.open("rb") as handle:
            self.bm25 = pickle.load(handle)

    def _load_chroma_docs(self) -> None:
        if not self.use_chroma:
            return
        fetched = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids = list(fetched.get("ids", []))
        self.texts = list(fetched.get("documents", []))
        self.metadatas = list(fetched.get("metadatas", []))
        self.documents = [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(self.doc_ids, self.texts, self.metadatas)
        ]
        self.tokenized_documents = [_tokenize(text) for text in self.texts]

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.documents = list(documents)
        self.doc_ids = [doc["id"] for doc in self.documents]
        self.texts = [doc["text"] for doc in self.documents]
        self.metadatas = [_normalize_metadata(doc["metadata"]) for doc in self.documents]
        self.tokenized_documents = [_tokenize(text) for text in self.texts]

        if isinstance(self.embedding_model, _LocalTfidfEmbeddingModel):
            self.embedding_model.fit(self.texts)

        self.embeddings = []
        batch_size = 32
        for start in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[start : start + batch_size]
            logger.info("Embedding docs %s-%s", start + 1, start + len(batch_texts))
            self.embeddings.extend(self._embed_texts(batch_texts, batch_size=batch_size))

        if self.use_chroma:
            for start in range(0, len(self.texts), batch_size):
                end = start + batch_size
                self.collection.upsert(
                    ids=self.doc_ids[start:end],
                    embeddings=self.embeddings[start:end],
                    documents=self.texts[start:end],
                    metadatas=self.metadatas[start:end],
                )
        else:
            self._save_local_store()

        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.tokenized_documents)
        else:
            logger.warning("rank-bm25 unavailable; using local BM25 implementation.")
            self.bm25 = _SimpleBM25(self.tokenized_documents)

        with self.bm25_path.open("wb") as handle:
            pickle.dump(self.bm25, handle)

        logger.info("Indexed %s documents", len(self.documents))

    def is_indexed(self) -> bool:
        if self.use_chroma:
            try:
                return self.collection.count() > 0
            except Exception:
                return False
        return self.local_store_path.exists()

    def load_existing(self) -> None:
        if self.use_chroma:
            self._load_chroma_docs()
        else:
            self._load_local_store()
        self._load_bm25()

    def semantic_search(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[SemanticCandidate]:
        query_embedding = self._embed_texts([query], batch_size=1)[0]

        if self.use_chroma:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(max(1, n_results), max(1, len(self.doc_ids) or n_results)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = _to_chroma_where_filter(where)
            fetched = self.collection.query(**query_kwargs)
            ids = fetched.get("ids", [[]])[0]
            documents = fetched.get("documents", [[]])[0]
            metadatas = fetched.get("metadatas", [[]])[0]
            distances = fetched.get("distances", [[]])[0]
            return [
                SemanticCandidate(
                    id=str(doc_id),
                    text=text,
                    metadata=metadata,
                    semantic_score=max(0.0, 1.0 - float(distance or 0.0)),
                )
                for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
            ]

        candidates: list[SemanticCandidate] = []
        for doc_id, text, metadata, embedding in zip(
            self.doc_ids, self.texts, self.metadatas, self.embeddings
        ):
            if where and not _matches_local_filter(metadata, where):
                continue
            candidates.append(
                SemanticCandidate(
                    id=doc_id,
                    text=text,
                    metadata=metadata,
                    semantic_score=_cosine_similarity(query_embedding, embedding),
                )
            )

        candidates.sort(key=lambda item: item.semantic_score, reverse=True)
        return candidates[:n_results]


def get_or_build_index(
    config: SimpleNamespace,
    documents: list[dict[str, Any]],
    force_reindex: bool = False,
) -> KBIndexer:
    indexer = KBIndexer(config)

    if force_reindex:
        if indexer.local_store_path.exists():
            indexer.local_store_path.unlink()
        if indexer.bm25_path.exists():
            indexer.bm25_path.unlink()

    if indexer.is_indexed() and not force_reindex:
        indexer.load_existing()
        if not indexer.documents:
            indexer.build_index(documents)
    else:
        indexer.build_index(documents)

    if indexer.bm25 is None and indexer.bm25_path.exists():
        indexer.load_existing()

    return indexer
