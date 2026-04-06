from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.lexical_retrieval import LexicalIndex
from src.query_router import StructuredQueryPlan
from src.reranker import HybridReranker
from src.schemas import RetrievedChunkArtifact
from src.vectorstore import LocalVectorStore


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunkArtifact]
    diagnostics: dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        lexical_index: LexicalIndex,
        reranker: HybridReranker,
        top_k_dense: int,
        top_k_sparse: int,
        top_k_fused: int,
        top_k_reranked: int,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.reranker = reranker
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_fused = top_k_fused
        self.top_k_reranked = top_k_reranked
        self.rrf_k = rrf_k

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], filter_spec: dict[str, Any]) -> bool:
        field = filter_spec["field"]
        op = filter_spec["op"]
        value = filter_spec["value"]
        actual_raw = metadata.get(field, "")
        actual = str(actual_raw).lower()
        if op == "eq":
            return actual == str(value).lower()
        if op == "contains":
            return str(value).lower() in actual
        if op == "in":
            return actual in {str(item).lower() for item in value}
        if op == "gt":
            return float(actual_raw or 0.0) > float(value)
        if op == "gte":
            return float(actual_raw or 0.0) >= float(value)
        if op == "lt":
            return float(actual_raw or 0.0) < float(value)
        if op == "lte":
            return float(actual_raw or 0.0) <= float(value)
        return True

    def _allowed_chunk_ids(self, plan: StructuredQueryPlan) -> set[str] | None:
        if plan.support_record_ids:
            allowed = {
                chunk.chunk_id
                for chunk in self.vector_store.chunks
                if str(chunk.metadata.get("record_id", "")) in {str(record_id) for record_id in plan.support_record_ids}
            }
            return allowed or None

        allowed: set[str] = set()
        for chunk in self.vector_store.chunks:
            if all(self._metadata_matches(chunk.metadata, filter_spec) for filter_spec in plan.filters):
                if plan.supplier_only:
                    is_supplier = bool(chunk.metadata.get("is_supplier"))
                    classification = str(chunk.metadata.get("classification_method", "")).lower()
                    supplier_type = str(chunk.metadata.get("supplier_or_affiliation_type", "")).lower()
                    if not is_supplier and classification != "supplier" and "automotive supply chain participant" not in supplier_type:
                        continue
                allowed.add(chunk.chunk_id)
        return allowed or None

    def _fuse(
        self,
        dense_hits: list[tuple[Any, float]],
        sparse_hits: list[tuple[Any, float]],
    ) -> list[tuple[Any, float, dict[str, int]]]:
        scores: dict[str, float] = {}
        provenance: dict[str, dict[str, int]] = {}
        chunk_map: dict[str, Any] = {}

        for rank, (chunk, _) in enumerate(dense_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            provenance.setdefault(chunk.chunk_id, {})["dense"] = rank

        for rank, (chunk, _) in enumerate(sparse_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            provenance.setdefault(chunk.chunk_id, {})["sparse"] = rank

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            (chunk_map[chunk_id], score, provenance.get(chunk_id, {}))
            for chunk_id, score in ordered[: self.top_k_fused]
        ]

    def retrieve(self, question: str, plan: StructuredQueryPlan) -> RetrievalResult:
        if not plan.use_retrieval_support:
            diagnostics = {
                "allowed_chunk_count": len(plan.support_record_ids),
                "filter_fallback_applied": False,
                "dense_hit_count": 0,
                "sparse_hit_count": 0,
                "fused_hit_count": 0,
                "reranked_hit_count": 0,
                "dense_scores": {},
                "sparse_scores": {},
                "rerank_scores": {},
                "retrieval_skipped": True,
            }
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        query_text = plan.retrieval_query or question
        allowed_chunk_ids = self._allowed_chunk_ids(plan)
        dense_hits = self.vector_store.search(query_text, self.top_k_dense, allowed_chunk_ids=allowed_chunk_ids)
        sparse_hits = self.lexical_index.search(query_text, self.top_k_sparse, allowed_chunk_ids=allowed_chunk_ids)

        if not dense_hits and not sparse_hits and allowed_chunk_ids is not None:
            dense_hits = self.vector_store.search(query_text, self.top_k_dense, allowed_chunk_ids=None)
            sparse_hits = self.lexical_index.search(query_text, self.top_k_sparse, allowed_chunk_ids=None)
            allowed_chunk_ids = None
            filter_fallback = True
        else:
            filter_fallback = False

        fused = self._fuse(dense_hits, sparse_hits)
        reranked = self.reranker.rerank(
            query=query_text,
            candidates=[chunk for chunk, _, _ in fused],
            top_k=self.top_k_reranked,
            filter_values=[str(spec["value"]) for spec in plan.filters],
        )
        rerank_scores = {chunk.chunk_id: score for chunk, score in reranked}

        artifacts: list[RetrievedChunkArtifact] = []
        for reranked_rank, (chunk, rerank_score) in enumerate(reranked, start=1):
            provenance = next((p for c, _, p in fused if c.chunk_id == chunk.chunk_id), {})
            artifacts.append(
                RetrievedChunkArtifact(
                    chunk_id=chunk.chunk_id,
                    score=float(rerank_score),
                    source="reranked",
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_rank=provenance.get("dense"),
                    sparse_rank=provenance.get("sparse"),
                    fused_rank=next((idx for idx, (c, _, _) in enumerate(fused, start=1) if c.chunk_id == chunk.chunk_id), None),
                    reranked_rank=reranked_rank,
                )
            )

        diagnostics = {
            "allowed_chunk_count": len(allowed_chunk_ids or []),
            "filter_fallback_applied": filter_fallback,
            "dense_hit_count": len(dense_hits),
            "sparse_hit_count": len(sparse_hits),
            "fused_hit_count": len(fused),
            "reranked_hit_count": len(artifacts),
            "retrieval_skipped": False,
            "query_text": query_text,
            "dense_scores": {chunk.chunk_id: score for chunk, score in dense_hits[:10]},
            "sparse_scores": {chunk.chunk_id: score for chunk, score in sparse_hits[:10]},
            "rerank_scores": rerank_scores,
        }
        return RetrievalResult(chunks=artifacts, diagnostics=diagnostics)
