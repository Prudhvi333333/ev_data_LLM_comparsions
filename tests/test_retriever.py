from __future__ import annotations

from pathlib import Path

import numpy as np

from src.embeddings import HashingEmbedder
from src.lexical_retrieval import LexicalIndex
from src.query_router import StructuredQueryPlan
from src.reranker import HybridReranker
from src.retriever import HybridRetriever
from src.schemas import KnowledgeChunk
from src.vectorstore import LocalVectorStore


def test_hybrid_retriever_respects_structured_filters(tmp_path: Path) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="chunk-1",
            record_id="1",
            text="Company: Alpha Battery\nCategory: Tier 1\nCounty: Troup County\nEV Supply Chain Role: Battery Pack\nEmployment: 700",
            metadata={
                "company": "Alpha Battery",
                "category": "Tier 1",
                "category_normalized": "tier 1",
                "county": "Troup County",
                "record_id": "1",
                "classification_method": "Supplier",
                "supplier_or_affiliation_type": "Automotive supply chain participant",
                "is_supplier": True,
            },
        ),
        KnowledgeChunk(
            chunk_id="chunk-2",
            record_id="2",
            text="Company: Beta Thermal\nCategory: Tier 1/2\nCounty: Cobb County\nEV Supply Chain Role: Thermal Management\nEmployment: 100",
            metadata={
                "company": "Beta Thermal",
                "category": "Tier 1/2",
                "category_normalized": "tier 1 2",
                "county": "Cobb County",
                "record_id": "2",
                "classification_method": "Supplier",
                "supplier_or_affiliation_type": "Automotive supply chain participant",
                "is_supplier": True,
            },
        ),
    ]
    embedder = HashingEmbedder(dimension=128)
    embedder.fit([chunk.text for chunk in chunks])
    vectors = embedder.embed_texts([chunk.text for chunk in chunks]).astype(np.float32)
    vector_store = LocalVectorStore(tmp_path, chunks, vectors, embedder, manifest={"embedding_backend": embedder.name})
    lexical_index = LexicalIndex.build_or_load(chunks, tmp_path)
    reranker = HybridReranker("missing-cross-encoder", allow_remote_downloads=False)
    retriever = HybridRetriever(
        vector_store=vector_store,
        lexical_index=lexical_index,
        reranker=reranker,
        top_k_dense=4,
        top_k_sparse=4,
        top_k_fused=4,
        top_k_reranked=2,
    )
    plan = StructuredQueryPlan(
        route_name="filtered_list",
        question_type="list_records",
        filters=[{"field": "category_normalized", "op": "eq", "value": "tier 1"}],
        supplier_only=True,
        requires_structured_context=True,
        group_by=[],
        aggregate=None,
        aggregate_field=None,
        sort_by=None,
        sort_desc=True,
        top_n=None,
        requested_fields=["company"],
        dedupe_by="company",
        answer_schema=["company"],
        use_retrieval_support=True,
        support_record_ids=["1"],
    )

    result = retriever.retrieve("battery pack supplier in troup county", plan)
    assert result.chunks
    assert result.chunks[0].metadata["company"] == "Alpha Battery"
    assert all(chunk.metadata["category"] == "Tier 1" for chunk in result.chunks)
