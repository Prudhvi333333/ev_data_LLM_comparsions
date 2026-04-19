from __future__ import annotations

from typing import Any

from src.schemas import GenerationRecord


def build_context_list(record: GenerationRecord) -> list[str]:
    if record.final_context_presented_to_model:
        return [record.final_context_presented_to_model]
    if record.retrieved_context_exact:
        return [record.retrieved_context_exact]
    contexts: list[str] = []
    for artifact in record.structured_ops_outputs:
        rendered = artifact.get("rendered_text") if isinstance(artifact, dict) else None
        if rendered:
            contexts.append(str(rendered))
    for chunk_text in record.retrieved_chunks:
        if chunk_text:
            contexts.append(str(chunk_text))
    return contexts


def build_single_turn_payload(record: GenerationRecord) -> dict[str, Any]:
    return {
        "user_input": record.question,
        "response": record.generated_answer,
        "reference": record.reference_answer,
        "retrieved_contexts": build_context_list(record),
    }


def build_evaluation_dataset(records: list[GenerationRecord]) -> Any:
    try:
        from ragas import EvaluationDataset
    except Exception as exc:  # pragma: no cover - import guarded at runtime
        raise RuntimeError("ragas is not installed. Run `pip install -r requirements.txt`.") from exc

    rows = [build_single_turn_payload(record) for record in records]
    if hasattr(EvaluationDataset, "from_list"):
        return EvaluationDataset.from_list(rows)

    try:
        from ragas import SingleTurnSample
    except Exception as exc:  # pragma: no cover - compatibility fallback
        raise RuntimeError("Installed ragas version does not expose EvaluationDataset.from_list.") from exc

    samples = [SingleTurnSample(**row) for row in rows]
    return EvaluationDataset(samples=samples)
