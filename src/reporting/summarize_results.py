from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.schemas import EvaluationRecord, GenerationRecord
from src.utils.files import ensure_directory, summarize_status_counts, write_json, write_table_outputs


def summarize_generation(records: list[GenerationRecord]) -> dict:
    successful = [record for record in records if record.answer_status == "success"]
    return {
        "total_questions": len(records),
        "success_rate": (len(successful) / len(records)) if records else 0.0,
        "status_counts": summarize_status_counts([record.model_dump() for record in records], "answer_status"),
        "mean_latency_seconds": sum((record.latency_seconds or 0.0) for record in successful) / len(successful) if successful else 0.0,
        "rag_enabled": records[0].rag_enabled if records else None,
        "pipeline_name": records[0].pipeline_name if records else None,
    }


def summarize_evaluation(records: list[EvaluationRecord]) -> dict:
    successful = [record for record in records if record.evaluation_status == "success"]
    def _mean(field: str) -> float | None:
        values = [getattr(record, field) for record in successful if getattr(record, field) is not None]
        return (sum(values) / len(values)) if values else None

    return {
        "total_rows": len(records),
        "status_counts": summarize_status_counts([record.model_dump() for record in records], "evaluation_status"),
        "mean_faithfulness": _mean("faithfulness"),
        "mean_response_relevancy": _mean("response_relevancy"),
        "mean_context_precision": _mean("context_precision"),
        "mean_context_recall": _mean("context_recall"),
        "mean_answer_correctness": _mean("answer_correctness"),
        "pipeline_name": records[0].pipeline_name if records else None,
    }


def write_generation_reports(records: list[GenerationRecord], output_dir: Path) -> None:
    ensure_directory(output_dir)
    summary = summarize_generation(records)
    write_json(output_dir / "generation_summary.json", summary)
    pd.DataFrame([record.model_dump() for record in records]).to_csv(output_dir / "generation_rows.csv", index=False)
    error_rows = [record.model_dump() for record in records if record.answer_status != "success"]
    pd.DataFrame(error_rows).to_csv(output_dir / "generation_error_analysis.csv", index=False)
    rag_rows = [record.model_dump() for record in records if record.rag_enabled]
    if rag_rows:
        retrieval_rows = []
        for row in rag_rows:
            retrieval_rows.append(
                {
                    "question_id": row["question_id"],
                    "pipeline_name": row["pipeline_name"],
                    "retrieved_chunk_count": len(row["retrieved_chunk_ids"]),
                    "structured_ops_used": ",".join(row["structured_ops_used"]),
                    "filter_fallback_applied": row["retrieval_diagnostics"].get("filter_fallback_applied"),
                }
            )
        pd.DataFrame(retrieval_rows).to_csv(output_dir / "retrieval_diagnostics.csv", index=False)


def write_evaluation_reports(records: list[EvaluationRecord], output_dir: Path) -> None:
    ensure_directory(output_dir)
    summary = summarize_evaluation(records)
    write_json(output_dir / "evaluation_summary.json", summary)
    pd.DataFrame([record.model_dump() for record in records]).to_csv(output_dir / "evaluation_rows.csv", index=False)
    metric_rows = [record.model_dump() for record in records if record.evaluation_status == "success"]
    pd.DataFrame(metric_rows).to_csv(output_dir / "evaluation_metric_rows.csv", index=False)


def _read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_generation_run_catalog(generation_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not generation_root.exists():
        return rows
    for manifest_path in sorted(generation_root.glob("*/*/run_manifest.json")):
        manifest = _read_json_if_present(manifest_path)
        summary = _read_json_if_present(manifest_path.parent / "generation_summary.json")
        status_counts = summary.get("status_counts", {})
        rows.append(
            {
                "pipeline_name": manifest.get("pipeline_name"),
                "model_name": manifest.get("model_name"),
                "rag_enabled": manifest.get("rag_enabled"),
                "run_key": manifest.get("run_key"),
                "question_count_target": manifest.get("question_count_target"),
                "total_questions_materialized": summary.get("total_questions"),
                "success_rate": summary.get("success_rate"),
                "success_count": status_counts.get("success", 0),
                "failed_count": status_counts.get("failed", 0),
                "started_at_utc": manifest.get("started_at_utc"),
                "completed_at_utc": manifest.get("completed_at_utc"),
                "output_dir": str(manifest_path.parent),
            }
        )
    return rows


def collect_evaluation_run_catalog(evaluation_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not evaluation_root.exists():
        return rows
    for manifest_path in sorted(evaluation_root.glob("*/*/evaluation_manifest.json")):
        manifest = _read_json_if_present(manifest_path)
        summary = _read_json_if_present(manifest_path.parent / "evaluation_summary.json")
        status_counts = summary.get("status_counts", {})
        rows.append(
            {
                "pipeline_name": manifest.get("pipeline_name"),
                "run_key": manifest.get("run_key"),
                "question_count_target": manifest.get("question_count_target"),
                "total_rows_materialized": summary.get("total_rows"),
                "success_count": status_counts.get("success", 0),
                "failed_count": status_counts.get("failed", 0),
                "not_applicable_count": status_counts.get("not_applicable", 0),
                "mean_faithfulness": summary.get("mean_faithfulness"),
                "mean_response_relevancy": summary.get("mean_response_relevancy"),
                "mean_context_precision": summary.get("mean_context_precision"),
                "mean_context_recall": summary.get("mean_context_recall"),
                "mean_answer_correctness": summary.get("mean_answer_correctness"),
                "started_at_utc": manifest.get("started_at_utc"),
                "completed_at_utc": manifest.get("completed_at_utc"),
                "output_dir": str(manifest_path.parent),
            }
        )
    return rows


def write_run_catalogs(root: Path, output_dir: Path) -> tuple[Path, Path]:
    ensure_directory(output_dir)
    generation_rows = collect_generation_run_catalog(root / "outputs" / "generation")
    evaluation_rows = collect_evaluation_run_catalog(root / "outputs" / "evaluation")
    generation_path = output_dir / "generation_run_catalog"
    evaluation_path = output_dir / "evaluation_run_catalog"
    write_table_outputs(generation_path, generation_rows)
    write_table_outputs(evaluation_path, evaluation_rows)
    return generation_path.with_suffix(".csv"), evaluation_path.with_suffix(".csv")
