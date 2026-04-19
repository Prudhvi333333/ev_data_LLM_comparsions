from __future__ import annotations

from src.evaluation.dataset_builder import build_context_list, build_single_turn_payload
from src.schemas import GenerationRecord


def test_dataset_builder_prefers_exact_final_context_when_available() -> None:
    record = GenerationRecord(
        question_id="8",
        question="Which county has the highest total employment among Tier 1 suppliers only?",
        reference_answer="Troup County has 2,435 employees.",
        pipeline_name="qwen_rag",
        model_name="qwen2.5:14b",
        rag_enabled=True,
        prompt_version="benchmark_v2",
        system_prompt="system",
        user_prompt="user",
        retrieved_context_exact="context",
        retrieved_chunks=["chunk-a", "chunk-b"],
        structured_ops_outputs=[{"rendered_text": "structured-a"}],
        generated_answer="Troup County has 2,435 employees.",
        answer_status="success",
    )

    contexts = build_context_list(record)
    assert contexts == ["context"]

    payload = build_single_turn_payload(record)
    assert payload["user_input"] == record.question
    assert payload["response"] == record.generated_answer
    assert payload["reference"] == record.reference_answer
    assert payload["retrieved_contexts"] == contexts


def test_dataset_builder_reconstructs_context_from_saved_artifacts_when_needed() -> None:
    record = GenerationRecord(
        question_id="9",
        question="List battery pack suppliers.",
        reference_answer="Reference",
        pipeline_name="qwen_rag",
        model_name="qwen2.5:14b",
        rag_enabled=True,
        prompt_version="benchmark_v2",
        system_prompt="system",
        user_prompt="user",
        retrieved_chunks=["chunk-a", "chunk-b"],
        structured_ops_outputs=[{"rendered_text": "structured-a"}],
        generated_answer="Answer",
        answer_status="success",
    )

    assert build_context_list(record) == ["structured-a", "chunk-a", "chunk-b"]
