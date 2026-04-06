from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from src.chunking import build_chunks
from src.embeddings import BaseEmbedder, HashingEmbedder, build_embedder, load_embedder_state, save_embedder_state
from src.schemas import KnowledgeChunk
from src.utils.files import ensure_directory, read_jsonl, stable_hash_dict, write_json, write_jsonl
from src.utils.logger import get_logger


logger = get_logger("vectorstore")


class LocalVectorStore:
    def __init__(
        self,
        index_dir: Path,
        chunks: list[KnowledgeChunk],
        vectors: np.ndarray,
        embedder: BaseEmbedder,
        manifest: dict[str, Any],
    ) -> None:
        self.index_dir = index_dir
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder = embedder
        self.manifest = manifest
        self._chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def build_or_load(
        cls,
        kb_frame,
        corpus_hash: str,
        chunking_hash: str,
        index_root: Path,
        embedding_model: str,
        allow_remote_downloads: bool,
        allow_hash_fallback: bool,
        dense_dimension: int,
    ) -> "LocalVectorStore":
        chunks = build_chunks(kb_frame, corpus_hash)
        embedder = build_embedder(
            embedding_model,
            allow_remote_downloads=allow_remote_downloads,
            allow_hash_fallback=allow_hash_fallback,
            dimension=dense_dimension,
        )
        embedder_slug = embedding_model.replace("/", "__").replace(":", "_")
        index_dir = ensure_directory(index_root / corpus_hash / embedder_slug)
        manifest_path = index_dir / "manifest.json"
        chunks_path = index_dir / "chunks.jsonl"
        vectors_path = index_dir / "vectors.npy"
        embedder_state_path = index_dir / "embedder_state.json"

        expected_signature = stable_hash_dict(
            {
                "corpus_hash": corpus_hash,
                "chunking_hash": chunking_hash,
                "chunk_count": len(chunks),
            }
        )

        if manifest_path.exists() and chunks_path.exists() and vectors_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == expected_signature and manifest.get("embedding_backend") == embedder.name:
                logger.info("Loading vector index from %s", index_dir)
                loaded_chunks = [KnowledgeChunk.model_validate(row) for row in read_jsonl(chunks_path)]
                vectors = np.load(vectors_path)
                if embedder_state_path.exists():
                    try:
                        load_embedder_state(embedder_state_path, embedder)
                    except Exception as exc:
                        logger.warning("Failed to load embedder state: %s", exc)
                return cls(index_dir=index_dir, chunks=loaded_chunks, vectors=vectors, embedder=embedder, manifest=manifest)

        logger.info("Building vector index at %s", index_dir)
        texts = [chunk.text for chunk in chunks]
        embedder.fit(texts)
        vectors = embedder.embed_texts(texts)
        manifest = {
            "signature": expected_signature,
            "corpus_hash": corpus_hash,
            "chunking_hash": chunking_hash,
            "chunk_count": len(chunks),
            "embedding_backend": embedder.name,
        }
        write_jsonl(chunks_path, [chunk.model_dump() for chunk in chunks])
        np.save(vectors_path, vectors)
        save_embedder_state(embedder_state_path, embedder)
        write_json(manifest_path, manifest)
        return cls(index_dir=index_dir, chunks=chunks, vectors=vectors, embedder=embedder, manifest=manifest)

    def search(
        self,
        query: str,
        top_k: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        if top_k <= 0 or not self.chunks:
            return []
        query_vector = self.embedder.embed_query(query).astype(np.float32)
        scores = self.vectors @ query_vector
        order = np.argsort(scores)[::-1]
        results: list[tuple[KnowledgeChunk, float]] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            if allowed_chunk_ids is not None and chunk.chunk_id not in allowed_chunk_ids:
                continue
            results.append((chunk, float(scores[int(idx)])))
            if len(results) >= top_k:
                break
        return results

    @property
    def embedding_backend(self) -> str:
        return str(self.manifest.get("embedding_backend", self.embedder.name))
