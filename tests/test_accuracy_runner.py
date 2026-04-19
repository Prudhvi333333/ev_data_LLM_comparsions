from __future__ import annotations

from src.evaluation.accuracy_runner import parse_judge_payload, summarize_accuracy_records
from src.schemas import AccuracyEvaluationRecord


def test_parse_judge_payload_strips_code_fences() -> None:
    raw = """```json
    {"answer_correctness_score":0.72,"reason":"Misses two companies."}
    ```"""
    parsed = parse_judge_payload(raw)
    assert parsed["label"] == "partially_correct"
    assert parsed["answer_correctness_score"] == 0.72
    assert parsed["reason"] == "Misses two companies."


def test_parse_judge_payload_normalizes_score_from_label() -> None:
    raw = '{"label":"incorrect","reason":"Wrong count."}'
    parsed = parse_judge_payload(raw)
    assert parsed["label"] == "incorrect"
    assert parsed["answer_correctness_score"] == 0.0


def test_summarize_accuracy_records_counts_weighted_mean() -> None:
    rows = [
        AccuracyEvaluationRecord(
            question_id="1",
            question="q1",
            gold_answer="g1",
            generated_answer="a1",
            pipeline_name="qwen_rag",
            judge_model="llama3.1:8b",
            generation_output_path="x",
            evaluation_status="success",
            label="correct",
            answer_correctness_score=1.0,
            reason="ok",
        ),
        AccuracyEvaluationRecord(
            question_id="2",
            question="q2",
            gold_answer="g2",
            generated_answer="a2",
            pipeline_name="qwen_rag",
            judge_model="llama3.1:8b",
            generation_output_path="x",
            evaluation_status="success",
            label="partially_correct",
            answer_correctness_score=0.5,
            reason="partial",
        ),
        AccuracyEvaluationRecord(
            question_id="3",
            question="q3",
            gold_answer="g3",
            generated_answer="a3",
            pipeline_name="qwen_rag",
            judge_model="llama3.1:8b",
            generation_output_path="x",
            evaluation_status="failed",
            error_message="judge_timeout",
        ),
    ]
    summary = summarize_accuracy_records(rows)
    assert summary["correct_count"] == 1
    assert summary["partially_correct_count"] == 1
    assert summary["incorrect_count"] == 0
    assert summary["rows_evaluated"] == 2
    assert summary["failed_evaluation_rows"] == 1
    assert summary["mean_answer_correctness_score"] == 0.75
