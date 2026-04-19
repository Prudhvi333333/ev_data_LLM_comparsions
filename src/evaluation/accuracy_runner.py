from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.config_models import BenchmarkConfig
from src.schemas import AccuracyEvaluationRecord, GenerationRecord
from src.utils.files import ensure_directory, read_jsonl, write_json


JUDGE_SYSTEM_PROMPT = (
    "You are a strict benchmark evaluator.\n"
    "Compare a model-generated answer against a human-validated gold answer.\n"
    "Judge factual alignment and completeness, not writing style.\n"
    "Treat concise structured answers fairly.\n"
    "If the core entities, mappings, counts, top items, or locations match the gold answer, do not lower the score just because the answer is brief, lacks narrative framing, or includes extra harmless metadata.\n"
    "Return JSON only."
)

JUDGE_USER_TEMPLATE = """Question:
{question}

Human-validated gold answer:
{gold_answer}

Model-generated answer:
{generated_answer}

Return JSON only:
{{
  "answer_correctness_score": 0.0_to_1.0,
  "reason": "short explanation"
}}

Rules:
- Accept paraphrases.
- Ignore wording/style differences.
- Do not penalize concise or table-like answers when the factual content matches.
- Do not penalize extra harmless metadata such as tier, OEM, employment, location, or facility fields when the core answer is correct.
- If a short answer gives the exact count, exact top entity, or exact location/company tuple requested, that can still be correct.
- If a list answer contains all required gold entities or mappings, it can be correct even if it omits the gold answer's narrative preamble.
- Score guidance:
  - 1.00 = fully correct and complete
  - 0.85-0.95 = essentially correct with only minor omissions
  - 0.60-0.84 = substantially correct but incomplete
  - 0.40-0.59 = partly correct with important missing pieces
  - 0.10-0.39 = weak overlap / limited correctness
  - 0.00 = wrong, off-topic, hallucinated, or false abstention
- For list questions, materially complete coverage should score high.
- For count/ranking questions, exact result must be correct to score high.
- Do not reward verbosity.
- Return JSON only.
"""


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
    return cleaned.strip()


def label_from_score(score: float) -> str:
    if score >= 0.85:
        return "correct"
    if score >= 0.40:
        return "partially_correct"
    return "incorrect"


def parse_judge_payload(raw_text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(raw_text)
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        payload = {}

    score = payload.get("answer_correctness_score", payload.get("accuracy_score"))
    if score is None:
        score_match = re.search(
            r"(?:answer_correctness_score|accuracy_score|score)\s*[:=]\s*([01](?:\.\d+)?)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if score_match:
            score = score_match.group(1)

    if score is None:
        label_match = re.search(
            r"(?:label)\s*[:=]\s*['\"]?([a-z_ ]+)",
            cleaned,
            flags=re.IGNORECASE,
        )
        raw_label = str(payload.get("label", "")).strip().lower()
        if not raw_label and label_match:
            raw_label = str(label_match.group(1)).strip().lower()
        if raw_label == "partially correct":
            raw_label = "partially_correct"
        if raw_label == "partial":
            raw_label = "partially_correct"
        if not raw_label:
            lowered = cleaned.lower()
            if "partially_correct" in lowered or "partially correct" in lowered:
                raw_label = "partially_correct"
            elif "incorrect" in lowered:
                raw_label = "incorrect"
            elif "correct" in lowered:
                raw_label = "correct"
        if raw_label == "correct":
            normalized_score = 1.0
        elif raw_label == "partially_correct":
            normalized_score = 0.5
        elif raw_label == "incorrect":
            normalized_score = 0.0
        else:
            raise ValueError("Judge response missing answer_correctness_score.")
    else:
        normalized_score = max(0.0, min(1.0, float(score)))

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        reason_match = re.search(r"(?:reason)\s*[:=]\s*['\"]?(.+)", cleaned, flags=re.IGNORECASE)
        if reason_match:
            reason = str(reason_match.group(1)).strip(" \"'")
    if not reason:
        reason = "No reason provided by judge."

    return {
        "label": label_from_score(normalized_score),
        "answer_correctness_score": normalized_score,
        "reason": reason,
    }


def summarize_accuracy_records(records: list[AccuracyEvaluationRecord]) -> dict[str, Any]:
    success_rows = [row for row in records if row.evaluation_status == "success"]
    correct_count = sum(row.label == "correct" for row in success_rows)
    partially_correct_count = sum(row.label == "partially_correct" for row in success_rows)
    incorrect_count = sum(row.label == "incorrect" for row in success_rows)
    mean_answer_correctness_score = (
        sum(float(row.answer_correctness_score or 0.0) for row in success_rows) / len(success_rows) if success_rows else 0.0
    )
    return {
        "pipeline_name": success_rows[0].pipeline_name if success_rows else (records[0].pipeline_name if records else ""),
        "total_rows": len(records),
        "rows_evaluated": len(success_rows),
        "mean_answer_correctness_score": mean_answer_correctness_score,
        "correct_count": correct_count,
        "partially_correct_count": partially_correct_count,
        "incorrect_count": incorrect_count,
        "failed_evaluation_rows": sum(row.evaluation_status != "success" for row in records),
    }


def write_accuracy_summary(output_dir: Path, records: list[AccuracyEvaluationRecord]) -> dict[str, Any]:
    summary = summarize_accuracy_records(records)
    summary_frame = pd.DataFrame([summary])
    ensure_directory(output_dir)
    summary_frame.to_excel(output_dir / "answer_correctness_summary.xlsx", index=False, sheet_name="summary")
    write_json(output_dir / "answer_correctness_summary.json", summary)
    return summary


def build_accuracy_comparison(input_paths: list[Path], output_path: Path) -> Path:
    if len(input_paths) < 2:
        raise RuntimeError("accuracy comparison expects at least two answer_correctness_rows.jsonl inputs.")

    runs: list[tuple[str, list[AccuracyEvaluationRecord]]] = []
    for path in input_paths:
        rows = [AccuracyEvaluationRecord.model_validate(row) for row in read_jsonl(path)]
        if not rows:
            raise RuntimeError(f"No accuracy rows found in {path}")
        runs.append((rows[0].pipeline_name, rows))

    pipeline_maps = {name: {row.question_id: row for row in rows} for name, rows in runs}
    question_ids = sorted({qid for pipeline_map in pipeline_maps.values() for qid in pipeline_map}, key=lambda item: int(item))

    comparison_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        sample_row = next((pipeline_map.get(question_id) for pipeline_map in pipeline_maps.values() if pipeline_map.get(question_id)), None)
        row_payload: dict[str, Any] = {
            "question_id": question_id,
            "question": sample_row.question if sample_row else "",
        }
        best_score = -1.0
        winners: list[str] = []
        notes_parts: list[str] = []
        for pipeline_name, pipeline_map in pipeline_maps.items():
            row = pipeline_map.get(question_id)
            score = float(row.answer_correctness_score or 0.0) if row else 0.0
            row_payload[f"{pipeline_name}_score"] = score if row else None
            row_payload[f"{pipeline_name}_label"] = row.label if row else None
            if row:
                notes_parts.append(f"{pipeline_name}: {row.reason}")
            if score > best_score:
                best_score = score
                winners = [pipeline_name]
            elif score == best_score:
                winners.append(pipeline_name)
        row_payload["best_pipeline"] = winners[0] if len(winners) == 1 else "tie"
        row_payload["notes"] = " | ".join(notes_parts)
        comparison_rows.append(row_payload)

    aggregate_rows = []
    for pipeline_name, rows in runs:
        summary = summarize_accuracy_records(rows)
        aggregate_rows.append(
            {
                "pipeline_name": pipeline_name,
                "total_rows": summary["total_rows"],
                "mean_answer_correctness_score": summary["mean_answer_correctness_score"],
                "correct_count": summary["correct_count"],
                "partially_correct_count": summary["partially_correct_count"],
                "incorrect_count": summary["incorrect_count"],
            }
        )

    ensure_directory(output_path.parent)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        pd.DataFrame(aggregate_rows).to_excel(writer, index=False, sheet_name="aggregate_summary")
        pd.DataFrame(comparison_rows).to_excel(writer, index=False, sheet_name="per_question_comparison")
        rag_vs_norag_rows = []
        family_groups: dict[str, dict[str, float]] = {}
        for row in aggregate_rows:
            pipeline_name = str(row["pipeline_name"])
            family = pipeline_name.split("_")[0]
            mode = "rag" if pipeline_name.endswith("_rag") else "norag"
            family_groups.setdefault(family, {})[mode] = float(row["mean_answer_correctness_score"])
        for family, scores in sorted(family_groups.items()):
            rag_score = scores.get("rag")
            norag_score = scores.get("norag")
            rag_vs_norag_rows.append(
                {
                    "model_family": family,
                    "rag_score": rag_score,
                    "norag_score": norag_score,
                    "rag_minus_norag": (rag_score - norag_score) if rag_score is not None and norag_score is not None else None,
                }
            )
        pd.DataFrame(rag_vs_norag_rows).to_excel(writer, index=False, sheet_name="rag_vs_norag_summary")
    pd.DataFrame(comparison_rows).to_csv(output_path.with_suffix(".csv"), index=False)
    write_json(
        output_path.with_name(f"{output_path.stem}_summary.json"),
        {
            "aggregate_summary": aggregate_rows,
            "rag_vs_norag_summary": rag_vs_norag_rows,
        },
    )
    return output_path


class AccuracyRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.endpoint = config.accuracy_evaluation.endpoint
        self.judge_model = config.accuracy_evaluation.judge_model
        self.fallback_judge_models = list(config.accuracy_evaluation.fallback_judge_models)
        self.temperature = config.accuracy_evaluation.temperature
        self.max_tokens = config.accuracy_evaluation.max_tokens
        self.timeout_seconds = config.accuracy_evaluation.timeout_seconds
        self.retries = config.accuracy_evaluation.retries
        self.retry_backoff_seconds = max(0.1, float(config.accuracy_evaluation.retry_backoff_seconds))
        self.max_retry_backoff_seconds = max(1.0, float(config.accuracy_evaluation.max_retry_backoff_seconds))
        self.min_seconds_between_requests = max(0.0, float(config.accuracy_evaluation.min_seconds_between_requests))
        self.fail_open_with_zero = bool(config.accuracy_evaluation.fail_open_with_zero)
        self._last_request_ts = 0.0
        self.judge_models: list[str] = []
        for model_name in [self.judge_model, *self.fallback_judge_models]:
            model_name = str(model_name or "").strip()
            if model_name and model_name not in self.judge_models:
                self.judge_models.append(model_name)

    def _call_judge(self, question: str, gold_answer: str, generated_answer: str) -> str:
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"
        base_payload = {
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for model_name in self.judge_models:
                if self.min_seconds_between_requests > 0:
                    elapsed = time.monotonic() - self._last_request_ts
                    if elapsed < self.min_seconds_between_requests:
                        time.sleep(self.min_seconds_between_requests - elapsed)
                payload = dict(base_payload)
                payload["model"] = model_name
                self._last_request_ts = time.monotonic()
                try:
                    response = client.post(self.endpoint, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    raw = str(body.get("response", "") or "").strip()
                    if raw:
                        return raw
                    last_error = RuntimeError(f"Judge model '{model_name}' returned an empty response payload.")
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code == 404:
                        last_error = RuntimeError(
                            f"Judge model '{model_name}' not found at {self.endpoint}: {exc.response.text}"
                        )
                        continue
                    raise RuntimeError(
                        f"Judge API error for model '{model_name}' (status={status_code}): {exc.response.text}"
                    ) from exc
                except Exception as exc:
                    last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No judge model configured.")

    def evaluate_record(
        self,
        record: GenerationRecord,
        gold_answer: str,
        input_path: Path,
    ) -> AccuracyEvaluationRecord:
        if record.answer_status != "success":
            if self.fail_open_with_zero:
                reason = f"Generation failed ({record.answer_status}); assigned score 0.0 to avoid null metrics."
                if record.error_message:
                    reason = f"{reason} Error: {record.error_message}"
                return AccuracyEvaluationRecord(
                    question_id=record.question_id,
                    question=record.question,
                    gold_answer=gold_answer,
                    generated_answer=record.generated_answer,
                    pipeline_name=record.pipeline_name,
                    judge_model=self.judge_model,
                    generation_output_path=str(input_path),
                    evaluation_status="success",
                    label="incorrect",
                    answer_correctness_score=0.0,
                    reason=reason,
                    error_message=f"generation_status={record.answer_status}",
                )
            return AccuracyEvaluationRecord(
                question_id=record.question_id,
                question=record.question,
                gold_answer=gold_answer,
                generated_answer=record.generated_answer,
                pipeline_name=record.pipeline_name,
                judge_model=self.judge_model,
                generation_output_path=str(input_path),
                evaluation_status="not_applicable",
                error_message=f"generation_status={record.answer_status}",
            )

        last_error: str | None = None
        for attempt in range(1, self.retries + 1):
            try:
                raw_text = self._call_judge(
                    question=record.question,
                    gold_answer=gold_answer,
                    generated_answer=record.generated_answer,
                )
                parsed = parse_judge_payload(raw_text)
                return AccuracyEvaluationRecord(
                    question_id=record.question_id,
                    question=record.question,
                    gold_answer=gold_answer,
                    generated_answer=record.generated_answer,
                    pipeline_name=record.pipeline_name,
                    judge_model=self.judge_model,
                    generation_output_path=str(input_path),
                    evaluation_status="success",
                    label=parsed["label"],
                    answer_correctness_score=parsed["answer_correctness_score"],
                    reason=parsed["reason"],
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    delay = min(self.max_retry_backoff_seconds, self.retry_backoff_seconds * (2 ** (attempt - 1)))
                    time.sleep(delay)

        if self.fail_open_with_zero:
            reason = "Judge failed after retries; assigned score 0.0 to avoid null metrics."
            if last_error:
                reason = f"{reason} Error: {last_error}"
            return AccuracyEvaluationRecord(
                question_id=record.question_id,
                question=record.question,
                gold_answer=gold_answer,
                generated_answer=record.generated_answer,
                pipeline_name=record.pipeline_name,
                judge_model=self.judge_model,
                generation_output_path=str(input_path),
                evaluation_status="success",
                label="incorrect",
                answer_correctness_score=0.0,
                reason=reason,
                error_message=last_error,
            )

        return AccuracyEvaluationRecord(
            question_id=record.question_id,
            question=record.question,
            gold_answer=gold_answer,
            generated_answer=record.generated_answer,
            pipeline_name=record.pipeline_name,
            judge_model=self.judge_model,
            generation_output_path=str(input_path),
            evaluation_status="failed",
            error_message=last_error,
        )
