from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.context_compressor import ContextCompressor
from src.evaluator import RAGASEvaluator
from src.generator import get_generator
from src.indexer import get_or_build_index
from src.kb_loader import load_kb
from src.reporter import build_report
from src.retriever import HybridRetriever
from src.summary_reporter import build_cross_pipeline_summary
from src.utils.async_helpers import create_timeout_guard
from src.utils.config_loader import get_pipeline_id, load_config
from src.utils.logger import get_logger


logger = get_logger("main")

PIPELINES = [
    ("qwen", "rag"),
    ("qwen", "norag"),
    ("gemma", "rag"),
    ("gemma", "norag"),
    ("gemini", "rag"),
    ("gemini", "norag"),
]


def _extract_row_value(row: pd.Series, names: list[str], default: str = "") -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if pd.notna(value):
                return value
    return default


def _question_id_value(row: pd.Series) -> int:
    value = _extract_row_value(row, ["Num", "Question_ID", "Question ID"], default=0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_questions_df(config) -> pd.DataFrame:
    questions_df = pd.read_excel(config.paths.questions).fillna("")
    for column in questions_df.columns:
        questions_df[column] = questions_df[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return questions_df


def _progress_path(config, pipeline_id: str) -> Path:
    return Path(config.paths.progress) / f"{pipeline_id}_progress.jsonl"


def _load_progress_entries(progress_file: Path) -> dict[str, dict[str, Any]]:
    if not progress_file.exists():
        return {}
    loaded: dict[str, dict[str, Any]] = {}
    with progress_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            loaded[str(payload.get("question_id"))] = payload
    return loaded


async def _append_progress_entry(
    progress_file: Path,
    result: dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    async with lock:
        with progress_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def _count_timeouts(results: list[dict[str, Any]]) -> int:
    timeout_count = 0
    for result in results:
        if result.get("generated_answer") in {"TIMEOUT", "GENERATION_TIMEOUT"}:
            timeout_count += 1
        elif result.get("faithfulness_reason") == "timeout":
            timeout_count += 1
    return timeout_count


def _zero_metric_payload(reason: str) -> dict[str, Any]:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "faithfulness_reason": reason,
        "answer_relevancy_reason": reason,
        "context_precision_reason": reason,
        "context_recall_reason": reason,
        "answer_correctness_reason": reason,
        "final_score": 0.0,
    }


async def _process_question_row(
    row: pd.Series,
    mode: str,
    generator,
    retriever: HybridRetriever,
    compressor: ContextCompressor,
    evaluator: RAGASEvaluator,
    generation_semaphore: asyncio.Semaphore,
    eval_semaphore: asyncio.Semaphore,
    config,
    progress_file: Path,
    progress_lock: asyncio.Lock,
) -> dict[str, Any]:
    question_id = _question_id_value(row)
    category = str(_extract_row_value(row, ["Use Case Category", "Category"]))
    question = str(_extract_row_value(row, ["Question"]))
    golden = str(
        _extract_row_value(
            row,
            ["Human Validated Answers", "Human validated answers", "Golden_Answer"],
        )
    )

    try:
        if mode == "rag":
            docs = retriever.retrieve(question)
            context = compressor.compress(
                question,
                [doc["text"] for doc in docs],
                max_tokens=int(config.retrieval.max_context_tokens),
            )
        else:
            docs = []
            context = ""

        async with generation_semaphore:
            answer = await create_timeout_guard(
                generator.generate(question, context if mode == "rag" else None),
                timeout_sec=float(config.generation.timeout_seconds),
                fallback_value="GENERATION_TIMEOUT",
            )

        base_result = {
            "question_id": question_id,
            "category": category,
            "question": question,
            "golden_answer": golden,
            "generated_answer": answer,
            "retrieved_context": context if mode == "rag" else "",
        }
    except Exception as exc:
        logger.error("Retrieval/generation failed for question %s: %s", question_id, exc)
        result = {
            "question_id": question_id,
            "category": category,
            "question": question,
            "golden_answer": golden,
            "generated_answer": "PIPELINE_ERROR",
            "retrieved_context": "",
            **_zero_metric_payload("pipeline_error"),
        }
    else:
        try:
            async with eval_semaphore:
                evaluation = await evaluator.evaluate_row(question, golden, answer, context)
            result = {**base_result, **evaluation}
        except Exception as exc:
            logger.error("Evaluation failed for question %s: %s", question_id, exc)
            result = {
                **base_result,
                **_zero_metric_payload(f"evaluation_error: {exc}"),
            }

    await _append_progress_entry(progress_file, result, progress_lock)
    return result


async def run_pipeline(
    model_key: str,
    mode: str,
    config,
    documents: list[dict[str, Any]],
    retriever: HybridRetriever,
    questions_df: pd.DataFrame,
    evaluator: RAGASEvaluator,
    resume: bool = True,
) -> list[dict[str, Any]]:
    del documents

    pipeline_id = get_pipeline_id(model_key, mode)
    progress_file = _progress_path(config, pipeline_id)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        completed_rows = _load_progress_entries(progress_file)
    else:
        completed_rows = {}
        if progress_file.exists():
            progress_file.unlink()

    pending_rows = [
        row
        for _, row in questions_df.iterrows()
        if str(_question_id_value(row)) not in completed_rows
    ]

    generator = get_generator(model_key, config)
    compressor = ContextCompressor(config)
    generation_parallelism = int(config.concurrency.generation_semaphore)
    if model_key in {"qwen", "gemma"}:
        generation_parallelism = 1
    generation_semaphore = asyncio.Semaphore(max(1, generation_parallelism))
    eval_parallelism = int(config.concurrency.evaluation_semaphore)
    if str(getattr(evaluator, "provider", "")).lower() == "ollama":
        eval_parallelism = 1
    eval_semaphore = asyncio.Semaphore(max(1, eval_parallelism))
    progress_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            _process_question_row(
                row,
                mode,
                generator,
                retriever,
                compressor,
                evaluator,
                generation_semaphore,
                eval_semaphore,
                config,
                progress_file,
                progress_lock,
            )
        )
        for row in pending_rows
    ]

    results = list(completed_rows.values())
    progress_bar = tqdm(
        total=len(questions_df),
        initial=len(results),
        desc=f"Pipeline: {pipeline_id}",
    )

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        progress_bar.update(1)
    progress_bar.close()

    results.sort(key=lambda item: int(item.get("question_id") or 0))
    return results


def _parse_requested_pipelines(raw_values: list[str]) -> list[tuple[str, str]]:
    if not raw_values or raw_values == ["all"]:
        return list(PIPELINES)

    requested_ids: list[str] = []
    for value in raw_values:
        requested_ids.extend(part.strip() for part in value.split(",") if part.strip())

    pipeline_map = {get_pipeline_id(model_key, mode): (model_key, mode) for model_key, mode in PIPELINES}
    unknown = [pipeline_id for pipeline_id in requested_ids if pipeline_id not in pipeline_map]
    if unknown:
        valid = ", ".join(sorted(pipeline_map))
        raise ValueError(f"Unknown pipeline(s): {', '.join(unknown)}. Valid options: {valid}")

    return [pipeline_map[pipeline_id] for pipeline_id in requested_ids]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GNEM RAG vs No-RAG evaluation pipelines.")
    parser.add_argument(
        "--pipeline",
        nargs="*",
        default=["all"],
        help="Pipeline IDs to run, e.g. qwen_rag gemma_norag or 'all'.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from outputs/progress JSONL checkpoints.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuilding the KB index before running pipelines.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions from the questions workbook.",
    )
    parser.add_argument(
        "--question-id",
        nargs="*",
        default=None,
        help="Run only specific question IDs, e.g. --question-id 30 or --question-id 27 30.",
    )
    return parser


def _parse_question_ids(raw_values: list[str] | None) -> set[int]:
    if not raw_values:
        return set()

    parsed_ids: set[int] = set()
    for value in raw_values:
        for part in value.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            parsed_ids.add(int(stripped))
    return parsed_ids


async def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config("config/config.yaml")

    documents = load_kb(config)
    indexer = get_or_build_index(config, documents, force_reindex=args.force_reindex)
    retriever = HybridRetriever(indexer, config)
    questions_df = _load_questions_df(config)
    question_ids = _parse_question_ids(args.question_id)
    if question_ids:
        questions_df = questions_df[
            questions_df.apply(_question_id_value, axis=1).isin(question_ids)
        ]
    if args.limit is not None:
        questions_df = questions_df.head(max(0, args.limit))
    if questions_df.empty:
        raise ValueError("No questions matched the requested --question-id / --limit filters.")
    evaluator = RAGASEvaluator(config)

    report_paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_key, mode in _parse_requested_pipelines(args.pipeline):
        pipeline_id = get_pipeline_id(model_key, mode)
        started = perf_counter()
        logger.info("Running pipeline %s", pipeline_id)

        results = await run_pipeline(
            model_key,
            mode,
            config,
            documents,
            retriever,
            questions_df,
            evaluator,
            resume=args.resume,
        )
        report_paths[pipeline_id] = build_report(results, model_key, mode, config)

        final_scores = [float(row.get("final_score", 0.0) or 0.0) for row in results]
        summary_rows.append(
            {
                "pipeline": pipeline_id,
                "questions_answered": len(results),
                "timeouts": _count_timeouts(results),
                "mean_final_score": mean(final_scores) if final_scores else 0.0,
                "elapsed_seconds": perf_counter() - started,
            }
        )
        logger.info("Report saved: %s", report_paths[pipeline_id])

    comparison_path = build_cross_pipeline_summary(report_paths, config)
    logger.info("Final comparison saved: %s", comparison_path)

    logger.info("Final pipeline summary")
    for row in summary_rows:
        logger.info(
            "%s | questions_answered=%s | timeouts=%s | mean_final_score=%.4f | elapsed=%.2fs",
            row["pipeline"],
            row["questions_answered"],
            row["timeouts"],
            row["mean_final_score"],
            row["elapsed_seconds"],
        )


if __name__ == "__main__":
    asyncio.run(main())
