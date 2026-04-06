from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class KnowledgeChunk(BaseModel):
    chunk_id: str
    record_id: str
    text: str
    metadata: dict[str, Any]


class RetrievedChunkArtifact(BaseModel):
    chunk_id: str
    score: float
    source: Literal["dense", "sparse", "fused", "reranked"]
    text: str
    metadata: dict[str, Any]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    reranked_rank: int | None = None


class StructuredOpArtifact(BaseModel):
    name: str
    plan_summary: str
    filters: list[dict[str, Any]] = Field(default_factory=list)
    result_rows: list[dict[str, Any]] = Field(default_factory=list)
    rendered_text: str


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GenerationRecord(BaseModel):
    question_id: str
    question: str
    reference_answer: str
    pipeline_name: str
    model_name: str
    rag_enabled: bool
    prompt_version: str
    system_prompt: str
    user_prompt: str
    retrieved_context_exact: str = ""
    retrieved_chunks: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_metadata: list[dict[str, Any]] = Field(default_factory=list)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    structured_ops_used: list[str] = Field(default_factory=list)
    structured_ops_outputs: list[dict[str, Any]] = Field(default_factory=list)
    final_context_presented_to_model: str = ""
    query_route_name: str | None = None
    ontology_buckets: list[str] = Field(default_factory=list)
    generated_answer: str = ""
    raw_generated_answer: str | None = None
    answer_sanitized: bool = False
    answer_sanitizer_notes: list[str] = Field(default_factory=list)
    answer_status: Literal["success", "failed", "skipped"] = "failed"
    error_message: str | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    attempt_count: int = 1
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_tabular_row(self) -> dict[str, Any]:
        payload = self.model_dump()
        for key in (
            "retrieved_chunks",
            "retrieved_chunk_ids",
            "retrieved_metadata",
            "reranked_chunk_ids",
            "structured_ops_used",
            "structured_ops_outputs",
            "ontology_buckets",
            "answer_sanitizer_notes",
            "retrieval_diagnostics",
            "generation_metadata",
        ):
            payload[key] = json.dumps(payload[key], ensure_ascii=True)
        return payload


class GenerationRunManifest(BaseModel):
    manifest_version: str = "1.0"
    pipeline_name: str
    model_key: str
    model_name: str
    rag_enabled: bool
    prompt_version: str
    run_key: str
    output_dir: str
    question_count_target: int | None = None
    started_at_utc: str = Field(default_factory=utc_now_iso)
    completed_at_utc: str | None = None
    config_hash: str
    kb_hash: str
    questions_hash: str
    corpus_hash: str
    chunking_hash: str
    index_backend: str
    embedding_backend: str
    repo_commit: str | None = None
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    question_id: str
    pipeline_name: str
    generation_output_path: str
    evaluation_status: Literal["success", "failed", "not_applicable"] = "failed"
    faithfulness: float | None = None
    response_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    error_message: str | None = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)
    metric_payload: dict[str, Any] = Field(default_factory=dict)

    def to_tabular_row(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["metric_payload"] = json.dumps(payload["metric_payload"], ensure_ascii=True)
        return payload


class EvaluationRunManifest(BaseModel):
    manifest_version: str = "1.0"
    pipeline_name: str
    input_path: str
    output_dir: str
    run_key: str
    question_count_target: int | None = None
    started_at_utc: str = Field(default_factory=utc_now_iso)
    completed_at_utc: str | None = None
    metrics: list[str] = Field(default_factory=list)
    config_hash: str
    generation_manifest_hash: str | None = None
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)


class AccuracyEvaluationRecord(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    pipeline_name: str
    judge_model: str
    generation_output_path: str
    label: Literal["correct", "partially_correct", "incorrect"] | None = None
    answer_correctness_score: float | None = None
    reason: str | None = None
    evaluation_status: Literal["success", "failed", "not_applicable"] = "failed"
    error_message: str | None = None
    timestamp_utc: str = Field(default_factory=utc_now_iso)

    def to_tabular_row(self) -> dict[str, Any]:
        return self.model_dump()


class AccuracyEvaluationRunManifest(BaseModel):
    manifest_version: str = "1.0"
    pipeline_name: str
    input_path: str
    output_dir: str
    run_key: str
    question_count_target: int | None = None
    started_at_utc: str = Field(default_factory=utc_now_iso)
    completed_at_utc: str | None = None
    judge_model: str
    config_hash: str
    generation_manifest_hash: str | None = None
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)
