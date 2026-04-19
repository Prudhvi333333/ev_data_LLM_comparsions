from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.answer_sanitizer import sanitize_answer
from src.generators.base import BaseGenerator
from src.prompts import rag_system_prompt, rag_user_prompt
from src.retriever import HybridRetriever
from src.schemas import GenerationRecord
from src.structured_ops import StructuredOpsEngine
from src.structured_renderers import render_structured_answer


def _count_tokens_rough(text: str) -> int:
    return len(str(text).split())


def _truncate_context(blocks: list[str], max_tokens: int) -> str:
    kept: list[str] = []
    used_tokens = 0
    for block in blocks:
        block_tokens = _count_tokens_rough(block)
        if kept and used_tokens + block_tokens > max_tokens:
            break
        kept.append(block)
        used_tokens += block_tokens
    return "\n\n---\n\n".join(kept).strip()


@dataclass
class RAGPipeline:
    pipeline_name: str
    model_name: str
    generator: BaseGenerator
    retriever: HybridRetriever
    structured_ops: StructuredOpsEngine
    prompt_version: str
    rag_not_found_text: str
    max_context_tokens: int

    def run_question(self, question_row: dict[str, Any], attempt_count: int = 1) -> GenerationRecord:
        question = str(question_row["question"])
        reference_answer = str(question_row["reference_answer"])
        question_id = str(question_row["question_id"])
        plan, structured_artifacts = self.structured_ops.execute(question)
        retrieval = self.retriever.retrieve(question, plan)

        raw_chunks = retrieval.chunks
        context_blocks = [artifact.rendered_text for artifact in structured_artifacts]
        context_blocks.extend(chunk.text for chunk in raw_chunks)
        exact_context = _truncate_context(context_blocks, self.max_context_tokens)

        system_prompt = rag_system_prompt(self.rag_not_found_text)
        user_prompt = rag_user_prompt(question, exact_context, plan.answer_schema or plan.requested_fields)

        base_payload = dict(
            question_id=question_id,
            question=question,
            reference_answer=reference_answer,
            pipeline_name=self.pipeline_name,
            model_name=self.model_name,
            rag_enabled=True,
            prompt_version=self.prompt_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_context_exact=exact_context,
            retrieved_chunks=[chunk.text for chunk in raw_chunks],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in raw_chunks],
            retrieved_metadata=[chunk.metadata for chunk in raw_chunks],
            reranked_chunk_ids=[chunk.chunk_id for chunk in raw_chunks],
            structured_ops_used=[artifact.name for artifact in structured_artifacts],
            structured_ops_outputs=[artifact.model_dump() for artifact in structured_artifacts],
            final_context_presented_to_model=exact_context,
            query_route_name=plan.route_name,
            ontology_buckets=plan.ontology_buckets,
            retrieval_diagnostics=retrieval.diagnostics,
            attempt_count=attempt_count,
        )

        structured_answer = render_structured_answer(
            plan=plan,
            artifact=structured_artifacts[0] if structured_artifacts else None,
            not_found_text=self.rag_not_found_text,
        )
        if structured_answer is not None:
            sanitized = sanitize_answer(structured_answer, self.rag_not_found_text)
            return GenerationRecord(
                **base_payload,
                generated_answer=sanitized.answer,
                raw_generated_answer=structured_answer,
                answer_sanitized=sanitized.changed,
                answer_sanitizer_notes=sanitized.notes,
                answer_status="success",
                latency_seconds=0.0,
                generation_metadata={"structured_renderer_used": True, "route_notes": plan.route_notes},
            )

        try:
            result = self.generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            return GenerationRecord(
                **base_payload,
                answer_status="failed",
                error_message=str(exc),
            )

        sanitized = sanitize_answer(result.answer or self.rag_not_found_text, self.rag_not_found_text)
        return GenerationRecord(
            **base_payload,
            generated_answer=sanitized.answer,
            raw_generated_answer=result.answer,
            answer_sanitized=sanitized.changed,
            answer_sanitizer_notes=sanitized.notes,
            answer_status="success",
            latency_seconds=result.latency_seconds,
            input_tokens=result.token_usage.input_tokens,
            output_tokens=result.token_usage.output_tokens,
            total_tokens=result.token_usage.total_tokens,
            generation_metadata={**result.metadata, "route_notes": plan.route_notes},
        )
