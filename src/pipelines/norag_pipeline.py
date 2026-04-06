from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.answer_sanitizer import sanitize_answer
from src.generators.base import BaseGenerator
from src.prompts import norag_system_prompt, norag_user_prompt
from src.schemas import GenerationRecord


@dataclass
class NoRAGPipeline:
    pipeline_name: str
    model_name: str
    generator: BaseGenerator
    prompt_version: str
    not_found_text: str

    def run_question(self, question_row: dict[str, Any], attempt_count: int = 1) -> GenerationRecord:
        question = str(question_row["question"])
        reference_answer = str(question_row["reference_answer"])
        question_id = str(question_row["question_id"])

        system_prompt = norag_system_prompt(self.not_found_text)
        user_prompt = norag_user_prompt(question)
        base_payload = dict(
            question_id=question_id,
            question=question,
            reference_answer=reference_answer,
            pipeline_name=self.pipeline_name,
            model_name=self.model_name,
            rag_enabled=False,
            prompt_version=self.prompt_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attempt_count=attempt_count,
        )

        try:
            result = self.generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            return GenerationRecord(
                **base_payload,
                answer_status="failed",
                error_message=str(exc),
            )

        sanitized = sanitize_answer(result.answer or self.not_found_text, self.not_found_text)
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
            generation_metadata=result.metadata,
        )
