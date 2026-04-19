from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import pandas as pd

from src.schemas import EvaluationRecord, GenerationRecord
from src.utils.files import ensure_directory


def _load_generation_rows(paths: Iterable[Path]) -> list[GenerationRecord]:
    rows: list[GenerationRecord] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(GenerationRecord.model_validate(json.loads(line)))
    return rows


def _load_evaluation_rows(paths: Iterable[Path]) -> list[EvaluationRecord]:
    rows: list[EvaluationRecord] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(EvaluationRecord.model_validate(json.loads(line)))
    return rows


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_counts_from_manifests(
    paths: list[Path],
    manifest_filename: str,
    pipeline_counts: dict[str, int],
) -> dict[str, int]:
    expected_counts: dict[str, int] = {}
    for path in paths:
        manifest = _load_manifest(path.parent / manifest_filename)
        pipeline_name = manifest.get("pipeline_name")
        target = manifest.get("question_count_target")
        if pipeline_name and target is not None:
            expected_counts[str(pipeline_name)] = int(target)

    fallback_target = max(pipeline_counts.values(), default=0)
    for pipeline_name, actual_count in pipeline_counts.items():
        expected_counts.setdefault(pipeline_name, max(actual_count, fallback_target))
    return expected_counts


def merge_generation_runs(generation_paths: list[Path], output_dir: Path, force: bool = False) -> Path:
    ensure_directory(output_dir)
    rows = _load_generation_rows(generation_paths)
    if not rows:
        raise RuntimeError("No generation rows were provided for merge.")
    frame = pd.DataFrame(
        [
            {
                "question_id": row.question_id,
                "question": row.question,
                "reference_answer": row.reference_answer,
                "pipeline_name": row.pipeline_name,
                "generated_answer": row.generated_answer,
                "answer_status": row.answer_status,
                "latency_seconds": row.latency_seconds,
            }
            for row in rows
        ]
    )

    pipeline_counts = frame.groupby("pipeline_name")["question_id"].nunique().to_dict()
    expected_counts = _expected_counts_from_manifests(generation_paths, "run_manifest.json", pipeline_counts)
    incomplete = {
        name: {"actual": count, "expected": expected_counts[name]}
        for name, count in pipeline_counts.items()
        if count < expected_counts[name]
    }
    if incomplete and not force:
        raise RuntimeError(f"Refusing to merge incomplete runs without --force: {incomplete}")

    merged = frame.pivot_table(
        index=["question_id", "question", "reference_answer"],
        columns="pipeline_name",
        values=["generated_answer", "answer_status", "latency_seconds"],
        aggfunc="first",
    )
    merged.columns = ["__".join(str(part) for part in column if part) for column in merged.columns]
    merged = merged.reset_index().sort_values("question_id")
    summary = (
        frame.groupby("pipeline_name")
        .agg(
            questions=("question_id", "nunique"),
            success_rate=("answer_status", lambda values: sum(value == "success" for value in values) / max(1, len(values))),
            mean_latency_seconds=("latency_seconds", "mean"),
        )
        .reset_index()
    )
    output_path = output_dir / "merged_generation_runs.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        merged.to_excel(writer, index=False, sheet_name="generation_comparison")
        summary.to_excel(writer, index=False, sheet_name="generation_summary")
    merged.to_csv(output_dir / "merged_generation_runs.csv", index=False)
    summary.to_csv(output_dir / "generation_summary.csv", index=False)
    return output_path


def merge_evaluation_runs(evaluation_paths: list[Path], output_dir: Path, force: bool = False) -> Path:
    ensure_directory(output_dir)
    rows = _load_evaluation_rows(evaluation_paths)
    if not rows:
        raise RuntimeError("No evaluation rows were provided for merge.")
    frame = pd.DataFrame(
        [
            {
                "question_id": row.question_id,
                "pipeline_name": row.pipeline_name,
                "evaluation_status": row.evaluation_status,
                "faithfulness": row.faithfulness,
                "response_relevancy": row.response_relevancy,
                "context_precision": row.context_precision,
                "context_recall": row.context_recall,
                "answer_correctness": row.answer_correctness,
            }
            for row in rows
        ]
    )

    pipeline_counts = frame.groupby("pipeline_name")["question_id"].nunique().to_dict()
    expected_counts = _expected_counts_from_manifests(evaluation_paths, "evaluation_manifest.json", pipeline_counts)
    incomplete = {
        name: {"actual": count, "expected": expected_counts[name]}
        for name, count in pipeline_counts.items()
        if count < expected_counts[name]
    }
    if incomplete and not force:
        raise RuntimeError(f"Refusing to merge incomplete evaluation runs without --force: {incomplete}")

    summary = (
        frame[frame["evaluation_status"] == "success"]
        .groupby("pipeline_name")
        .agg(
            faithfulness=("faithfulness", "mean"),
            response_relevancy=("response_relevancy", "mean"),
            context_precision=("context_precision", "mean"),
            context_recall=("context_recall", "mean"),
            answer_correctness=("answer_correctness", "mean"),
        )
        .reset_index()
    )
    output_path = output_dir / "merged_evaluation_runs.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        frame.sort_values(["pipeline_name", "question_id"]).to_excel(writer, index=False, sheet_name="evaluation_rows")
        summary.to_excel(writer, index=False, sheet_name="aggregate_metrics_summary")
    frame.to_csv(output_dir / "merged_evaluation_runs.csv", index=False)
    summary.to_csv(output_dir / "aggregate_metrics_summary.csv", index=False)
    return output_path
