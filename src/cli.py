from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import subprocess

from src.archive_results import archive_active_results
from src.config_models import PIPELINE_REGISTRY
from src.evaluation.accuracy_runner import AccuracyRunner, build_accuracy_comparison, write_accuracy_summary
from src.evaluation.ragas_runner import RagasRunner
from src.ontology import DomainOntology
from src.ingestion import load_knowledge_base, load_questions
from src.lexical_retrieval import LexicalIndex
from src.pipelines.norag_pipeline import NoRAGPipeline
from src.pipelines.rag_pipeline import RAGPipeline
from src.reporting.merge_runs import merge_evaluation_runs, merge_generation_runs
from src.reporting.summarize_results import (
    collect_evaluation_run_catalog,
    collect_generation_run_catalog,
    write_evaluation_reports,
    write_generation_reports,
    write_run_catalogs,
)
from src.reranker import HybridReranker
from src.schemas import (
    AccuracyEvaluationRecord,
    AccuracyEvaluationRunManifest,
    EvaluationRecord,
    EvaluationRunManifest,
    GenerationRecord,
    GenerationRunManifest,
    utc_now_iso,
)
from src.structured_ops import StructuredOpsEngine
from src.utils.config_loader import config_hash, list_pipelines, load_config, model_config_for_pipeline, relative_to_repo, resolve_pipeline
from src.utils.files import ensure_directory, read_json, read_jsonl, stable_hash_dict, write_json, write_jsonl, write_table_outputs
from src.utils.logger import configure_logging, get_logger
from src.vectorstore import LocalVectorStore
from src.retriever import HybridRetriever


logger = get_logger("cli")


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _settings_snapshot(config: Any) -> dict[str, Any]:
    snapshot = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    if "api_keys" in snapshot:
        snapshot["api_keys"] = {key: "***redacted***" for key in snapshot["api_keys"]}
    return snapshot


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _canonical_generation_output_dir(repo_root: Path, generation_root: Path, pipeline_name: str, run_key: str) -> Path:
    return generation_root / pipeline_name / run_key


def _canonical_evaluation_output_dir(repo_root: Path, evaluation_root: Path, pipeline_name: str, run_key: str) -> Path:
    return evaluation_root / pipeline_name / run_key


def _canonical_accuracy_output_dir(repo_root: Path, evaluation_root: Path, pipeline_name: str, run_key: str) -> Path:
    return evaluation_root / pipeline_name / run_key


def _load_generation_records(path: Path) -> dict[str, GenerationRecord]:
    records: dict[str, GenerationRecord] = {}
    for row in read_jsonl(path):
        record = GenerationRecord.model_validate(row)
        records[record.question_id] = record
    return records


def _load_evaluation_records(path: Path) -> dict[str, EvaluationRecord]:
    records: dict[str, EvaluationRecord] = {}
    for row in read_jsonl(path):
        record = EvaluationRecord.model_validate(row)
        records[record.question_id] = record
    return records


def _load_accuracy_records(path: Path) -> dict[str, AccuracyEvaluationRecord]:
    records: dict[str, AccuracyEvaluationRecord] = {}
    for row in read_jsonl(path):
        record = AccuracyEvaluationRecord.model_validate(row)
        records[record.question_id] = record
    return records


def _materialize_generation_run(output_dir: Path, records: dict[str, GenerationRecord]) -> None:
    sorted_records = [records[key] for key in sorted(records, key=lambda item: int(item))]
    write_jsonl(output_dir / "responses.jsonl", [record.model_dump() for record in sorted_records])
    write_table_outputs(output_dir / "responses", [record.to_tabular_row() for record in sorted_records])
    write_generation_reports(sorted_records, output_dir)


def _materialize_evaluation_run(output_dir: Path, records: dict[str, EvaluationRecord]) -> None:
    sorted_records = [records[key] for key in sorted(records, key=lambda item: int(item))]
    write_jsonl(output_dir / "evaluation_rows.jsonl", [record.model_dump() for record in sorted_records])
    write_table_outputs(output_dir / "evaluation_rows", [record.to_tabular_row() for record in sorted_records])
    write_evaluation_reports(sorted_records, output_dir)


def _materialize_accuracy_run(output_dir: Path, records: dict[str, AccuracyEvaluationRecord]) -> None:
    sorted_records = [records[key] for key in sorted(records, key=lambda item: int(item))]
    write_jsonl(output_dir / "answer_correctness_rows.jsonl", [record.model_dump() for record in sorted_records])
    write_table_outputs(output_dir / "answer_correctness_rows", [record.to_tabular_row() for record in sorted_records])
    summary = write_accuracy_summary(output_dir, sorted_records)
    write_json(output_dir / "answer_correctness_summary.json", summary)


def _build_generator(model_key: str, model_config: Any, gemini_api_key: str, chatgpt_api_key: str) -> Any:
    endpoint = str(model_config.endpoint or "http://127.0.0.1:11434/api/generate")
    if model_config.provider == "ollama":
        from src.generators.qwen_generator import QwenGenerator

        return QwenGenerator(
            model_name=model_config.model_name,
            endpoint=endpoint,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout_seconds=model_config.timeout_seconds,
            num_ctx=model_config.num_ctx,
            top_p=model_config.top_p,
            repeat_penalty=model_config.repeat_penalty,
            retries=model_config.retries,
            retry_backoff_seconds=model_config.retry_backoff_seconds,
            max_retry_backoff_seconds=model_config.max_retry_backoff_seconds,
        )
    if model_config.provider == "gemini":
        from src.generators.gemini_generator import GeminiGenerator

        return GeminiGenerator(
            model_name=model_config.model_name,
            api_key=gemini_api_key,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout_seconds=model_config.timeout_seconds,
            top_p=model_config.top_p,
            retries=model_config.retries,
            retry_backoff_seconds=model_config.retry_backoff_seconds,
            max_retry_backoff_seconds=model_config.max_retry_backoff_seconds,
        )
    if model_config.provider == "openai":
        from src.generators.chatgpt_generator import ChatGPTGenerator

        return ChatGPTGenerator(
            model_name=model_config.model_name,
            api_key=chatgpt_api_key,
            endpoint=str(model_config.endpoint or "https://api.openai.com/v1/chat/completions"),
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout_seconds=model_config.timeout_seconds,
            top_p=model_config.top_p,
            retries=model_config.retries,
            retry_backoff_seconds=model_config.retry_backoff_seconds,
            max_retry_backoff_seconds=model_config.max_retry_backoff_seconds,
        )
    raise RuntimeError(f"Unsupported model provider for key '{model_key}': {model_config.provider}")


def _prepare_questions(question_frame, limit: int | None, question_ids: list[str] | None):
    working = question_frame.copy()
    if question_ids:
        wanted = {str(item) for item in question_ids}
        working = working[working["question_id"].astype(str).isin(wanted)]
    if limit is not None:
        working = working.head(max(0, limit))
    return working


def _resolve_question_subset(
    subset_name: str | None,
    config: Any,
    repo_root: Path,
) -> list[str] | None:
    if not subset_name:
        return None
    subset_path = relative_to_repo(getattr(config.question_subsets, subset_name), repo_root)
    payload = read_json(subset_path)
    if isinstance(payload, dict):
        question_ids = payload.get("question_ids", [])
    else:
        question_ids = payload
    return [str(item) for item in question_ids]


def run_generate(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    config = load_config(args.config, repo_root=repo_root)
    model_info = model_config_for_pipeline(config, args.pipeline)
    pipeline_name = model_info["pipeline_name"]
    model_key = model_info["model_key"]
    rag_enabled = model_info["rag_enabled"]
    model_config = model_info["model_config"]

    ontology = DomainOntology.load(relative_to_repo(config.paths.ontology, repo_root))
    kb_path = relative_to_repo(config.paths.kb, repo_root)
    question_path = relative_to_repo(config.paths.questions, repo_root)
    kb_frame, kb_fingerprint = load_knowledge_base(kb_path, ontology=ontology)
    question_frame, question_fingerprint = load_questions(question_path)
    subset_question_ids = _resolve_question_subset(args.subset, config, repo_root)
    if subset_question_ids:
        args.question_id = (args.question_id or []) + subset_question_ids
    question_frame = _prepare_questions(question_frame, args.limit, args.question_id)

    corpus_hash = stable_hash_dict({"kb_sha256": kb_fingerprint.sha256, "rows": kb_fingerprint.row_count})
    ontology_hash = stable_hash_dict({"ontology_path": config.paths.ontology, "ontology_version": ontology.version})
    chunking_hash = stable_hash_dict(
        {
            "field_order": "v3",
            "chunk_overlap": config.retrieval.chunk_overlap,
            "ontology_hash": ontology_hash,
        }
    )
    cfg_hash = config_hash(config)
    selected_question_ids = sorted(str(value) for value in question_frame["question_id"].astype(str).tolist())
    run_key = stable_hash_dict(
        {
            "pipeline": pipeline_name,
            "config_hash": cfg_hash,
            "kb_hash": kb_fingerprint.sha256,
            "questions_hash": question_fingerprint.sha256,
            "prompt_version": config.prompts.version,
            "subset": args.subset or "all",
            "limit": args.limit,
            "selected_question_ids": selected_question_ids,
        }
    )[:16]

    generation_root = relative_to_repo(config.paths.generation, repo_root)
    output_dir = ensure_directory(_canonical_generation_output_dir(repo_root, generation_root, pipeline_name, run_key))
    run_name = f"generate_{pipeline_name}_{_timestamp_slug()}"
    log_path = configure_logging(relative_to_repo(config.paths.logs, repo_root), run_name)
    logger.info("Starting generation run for %s", pipeline_name)

    manifest = GenerationRunManifest(
        pipeline_name=pipeline_name,
        model_key=model_key,
        model_name=model_config.model_name,
        rag_enabled=rag_enabled,
        prompt_version=config.prompts.version,
        run_key=run_key,
        output_dir=str(output_dir),
        question_count_target=len(question_frame),
        config_hash=cfg_hash,
        kb_hash=kb_fingerprint.sha256,
        questions_hash=question_fingerprint.sha256,
        corpus_hash=corpus_hash,
        chunking_hash=chunking_hash,
        index_backend="local-vectorstore",
        embedding_backend="pending",
        repo_commit=_git_commit(repo_root),
        settings_snapshot=_settings_snapshot(config),
    )

    responses_path = output_dir / "responses.jsonl"
    existing = _load_generation_records(responses_path) if (responses_path.exists() and args.resume) else {}

    generator = _build_generator(model_key, model_config, config.api_keys.gemini, config.api_keys.chatgpt)
    if rag_enabled:
        vector_store = LocalVectorStore.build_or_load(
            kb_frame=kb_frame,
            corpus_hash=corpus_hash,
            chunking_hash=chunking_hash,
            index_root=relative_to_repo(config.paths.indexes, repo_root),
            embedding_model=config.retrieval.embedding_model,
            allow_remote_downloads=config.retrieval.allow_remote_model_downloads,
            allow_hash_fallback=config.retrieval.allow_hash_embedding_fallback,
            dense_dimension=config.retrieval.dense_dimension,
        )
        lexical_index = LexicalIndex.build_or_load(vector_store.chunks, vector_store.index_dir)
        reranker = HybridReranker(
            model_name=config.retrieval.reranker_model,
            allow_remote_downloads=config.retrieval.allow_remote_model_downloads,
        )
        retriever = HybridRetriever(
            vector_store=vector_store,
            lexical_index=lexical_index,
            reranker=reranker,
            top_k_dense=config.retrieval.top_k_dense,
            top_k_sparse=config.retrieval.top_k_sparse,
            top_k_fused=config.retrieval.top_k_fused,
            top_k_reranked=config.retrieval.top_k_reranked,
            rrf_k=config.retrieval.rrf_k,
        )
        structured_ops = StructuredOpsEngine(
            kb_frame=kb_frame,
            ontology=ontology,
            top_n_default=config.structured_ops.top_n_default,
            max_evidence_rows=config.structured_ops.max_evidence_rows,
            ontology_match_mode=config.structured_ops.ontology_match_mode,
        )
        pipeline = RAGPipeline(
            pipeline_name=pipeline_name,
            model_name=model_config.model_name,
            generator=generator,
            retriever=retriever,
            structured_ops=structured_ops,
            prompt_version=config.prompts.version,
            rag_not_found_text=config.prompts.rag_not_found_text,
            max_context_tokens=config.retrieval.max_context_tokens,
        )
        manifest.embedding_backend = vector_store.embedding_backend
    else:
        pipeline = NoRAGPipeline(
            pipeline_name=pipeline_name,
            model_name=model_config.model_name,
            generator=generator,
            prompt_version=config.prompts.version,
            not_found_text=config.prompts.norag_not_found_text,
        )
        manifest.embedding_backend = "not_applicable"

    for _, question_row in question_frame.iterrows():
        question_id = str(question_row["question_id"])
        existing_record = existing.get(question_id)
        if args.resume and existing_record and existing_record.answer_status == "success":
            logger.info("Skipping completed question %s", question_id)
            continue
        attempt_count = (existing_record.attempt_count if existing_record else 0) + 1
        try:
            record = pipeline.run_question(question_row.to_dict(), attempt_count=attempt_count)
        except Exception as exc:
            logger.exception("Generation failed for question %s", question_id)
            record = GenerationRecord(
                question_id=question_id,
                question=str(question_row["question"]),
                reference_answer=str(question_row["reference_answer"]),
                pipeline_name=pipeline_name,
                model_name=model_config.model_name,
                rag_enabled=rag_enabled,
                prompt_version=config.prompts.version,
                system_prompt="",
                user_prompt="",
                answer_status="failed",
                error_message=str(exc),
                attempt_count=attempt_count,
            )
            if config.runtime.fail_fast:
                existing[question_id] = record
                _materialize_generation_run(output_dir, existing)
                manifest.completed_at_utc = utc_now_iso()
                write_json(output_dir / "run_manifest.json", manifest.model_dump())
                raise
        existing[question_id] = record
        _materialize_generation_run(output_dir, existing)

    manifest.completed_at_utc = utc_now_iso()
    write_json(output_dir / "run_manifest.json", manifest.model_dump())
    logger.info("Generation completed for %s. Outputs saved to %s", pipeline_name, output_dir)
    logger.info("Log file: %s", log_path)


def _resolve_generation_input(path_or_dir: str, repo_root: Path) -> Path:
    candidate = relative_to_repo(path_or_dir, repo_root)
    if candidate.is_dir():
        jsonl_path = candidate / "responses.jsonl"
        if jsonl_path.exists():
            return jsonl_path
    return candidate


def run_evaluate(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    config = load_config(args.config, repo_root=repo_root)
    input_path = _resolve_generation_input(args.input, repo_root)
    if not input_path.exists():
        raise RuntimeError(f"Generation input not found: {input_path}")

    generation_records = _load_generation_records(input_path)
    if not generation_records:
        raise RuntimeError(f"No generation rows found in {input_path}")
    sample_record = next(iter(generation_records.values()))
    pipeline_name = sample_record.pipeline_name

    cfg_hash = config_hash(config)
    run_key = stable_hash_dict(
        {"pipeline": pipeline_name, "input_path": str(input_path), "config_hash": cfg_hash, "metrics": config.evaluation.enabled_metrics}
    )[:16]
    evaluation_root = relative_to_repo(config.paths.evaluation, repo_root)
    output_dir = ensure_directory(_canonical_evaluation_output_dir(repo_root, evaluation_root, pipeline_name, run_key))
    run_name = f"evaluate_{pipeline_name}_{_timestamp_slug()}"
    log_path = configure_logging(relative_to_repo(config.paths.logs, repo_root), run_name)
    logger.info("Starting evaluation for %s", pipeline_name)

    existing = _load_evaluation_records(output_dir / "evaluation_rows.jsonl") if (output_dir / "evaluation_rows.jsonl").exists() and args.resume else {}
    runner = RagasRunner(config)

    generation_manifest_path = input_path.parent / "run_manifest.json"
    generation_manifest_hash = None
    if generation_manifest_path.exists():
        generation_manifest_hash = stable_hash_dict(read_json(generation_manifest_path))
    manifest = EvaluationRunManifest(
        pipeline_name=pipeline_name,
        input_path=str(input_path),
        output_dir=str(output_dir),
        run_key=run_key,
        question_count_target=len(generation_records),
        metrics=list(runner.metrics),
        config_hash=cfg_hash,
        generation_manifest_hash=generation_manifest_hash,
        settings_snapshot=_settings_snapshot(config),
    )

    for question_id in sorted(generation_records, key=lambda value: int(value)):
        if args.resume and question_id in existing and existing[question_id].evaluation_status == "success":
            logger.info("Skipping completed evaluation for question %s", question_id)
            continue
        try:
            evaluation_record = runner.evaluate_record(generation_records[question_id], input_path)
        except Exception as exc:
            logger.exception("Evaluation failed for question %s", question_id)
            evaluation_record = EvaluationRecord(
                question_id=question_id,
                pipeline_name=pipeline_name,
                generation_output_path=str(input_path),
                evaluation_status="failed",
                error_message=str(exc),
            )
            if config.runtime.fail_fast:
                existing[question_id] = evaluation_record
                _materialize_evaluation_run(output_dir, existing)
                manifest.completed_at_utc = utc_now_iso()
                write_json(output_dir / "evaluation_manifest.json", manifest.model_dump())
                raise
        existing[question_id] = evaluation_record
        _materialize_evaluation_run(output_dir, existing)

    manifest.completed_at_utc = utc_now_iso()
    write_json(output_dir / "evaluation_manifest.json", manifest.model_dump())
    logger.info("Evaluation completed for %s. Outputs saved to %s", pipeline_name, output_dir)
    logger.info("Log file: %s", log_path)


def run_evaluate_accuracy(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    config = load_config(args.config, repo_root=repo_root)
    input_path = _resolve_generation_input(args.input, repo_root)
    if not input_path.exists():
        raise RuntimeError(f"Generation input not found: {input_path}")

    generation_records = _load_generation_records(input_path)
    if not generation_records:
        raise RuntimeError(f"No generation rows found in {input_path}")
    sample_record = next(iter(generation_records.values()))
    pipeline_name = sample_record.pipeline_name

    question_path = relative_to_repo(config.paths.questions, repo_root)
    question_frame, _question_fingerprint = load_questions(question_path)
    gold_answers = {
        str(row["question_id"]): str(row["reference_answer"])
        for _, row in question_frame.iterrows()
    }

    cfg_hash = config_hash(config)
    run_key = stable_hash_dict(
        {
            "pipeline": pipeline_name,
            "input_path": str(input_path),
            "judge_model": config.accuracy_evaluation.judge_model,
            "config_hash": cfg_hash,
            "task": "accuracy_evaluation",
        }
    )[:16]
    accuracy_root = relative_to_repo(config.paths.accuracy_evaluation, repo_root)
    output_dir = ensure_directory(_canonical_accuracy_output_dir(repo_root, accuracy_root, pipeline_name, run_key))
    run_name = f"evaluate_accuracy_{pipeline_name}_{_timestamp_slug()}"
    log_path = configure_logging(relative_to_repo(config.paths.logs, repo_root), run_name)
    logger.info("Starting direct accuracy evaluation for %s", pipeline_name)

    existing = _load_accuracy_records(output_dir / "answer_correctness_rows.jsonl") if (output_dir / "answer_correctness_rows.jsonl").exists() and args.resume else {}
    runner = AccuracyRunner(config)

    generation_manifest_path = input_path.parent / "run_manifest.json"
    generation_manifest_hash = None
    if generation_manifest_path.exists():
        generation_manifest_hash = stable_hash_dict(read_json(generation_manifest_path))
    manifest = AccuracyEvaluationRunManifest(
        pipeline_name=pipeline_name,
        input_path=str(input_path),
        output_dir=str(output_dir),
        run_key=run_key,
        question_count_target=len(generation_records),
        judge_model=config.accuracy_evaluation.judge_model,
        config_hash=cfg_hash,
        generation_manifest_hash=generation_manifest_hash,
        settings_snapshot=_settings_snapshot(config),
    )

    for question_id in sorted(generation_records, key=lambda value: int(value)):
        if args.resume and question_id in existing and existing[question_id].evaluation_status == "success":
            logger.info("Skipping completed accuracy evaluation for question %s", question_id)
            continue
        gold_answer = gold_answers.get(question_id, generation_records[question_id].reference_answer)
        try:
            accuracy_record = runner.evaluate_record(generation_records[question_id], gold_answer, input_path)
        except Exception as exc:
            logger.exception("Accuracy evaluation failed for question %s", question_id)
            accuracy_record = AccuracyEvaluationRecord(
                question_id=question_id,
                question=generation_records[question_id].question,
                gold_answer=gold_answer,
                generated_answer=generation_records[question_id].generated_answer,
                pipeline_name=pipeline_name,
                judge_model=config.accuracy_evaluation.judge_model,
                generation_output_path=str(input_path),
                evaluation_status="failed",
                error_message=str(exc),
            )
            if config.runtime.fail_fast:
                existing[question_id] = accuracy_record
                _materialize_accuracy_run(output_dir, existing)
                manifest.completed_at_utc = utc_now_iso()
                write_json(output_dir / "accuracy_manifest.json", manifest.model_dump())
                raise
        existing[question_id] = accuracy_record
        _materialize_accuracy_run(output_dir, existing)

    manifest.completed_at_utc = utc_now_iso()
    write_json(output_dir / "accuracy_manifest.json", manifest.model_dump())
    logger.info("Direct accuracy evaluation completed for %s. Outputs saved to %s", pipeline_name, output_dir)
    logger.info("Log file: %s", log_path)


def run_compare_accuracy(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    input_paths = [relative_to_repo(path, repo_root) for path in args.inputs]
    output_path = relative_to_repo(args.output, repo_root)
    build_accuracy_comparison(input_paths, output_path)
    print(f"Wrote accuracy comparison workbook to:\n  {output_path}")


def run_merge_runs(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    output_dir = ensure_directory(relative_to_repo(args.output_dir, repo_root))
    generation_paths = [_resolve_generation_input(path, repo_root) for path in args.generation_inputs]
    merge_generation_runs(generation_paths, output_dir, force=args.force)
    if args.evaluation_inputs:
        evaluation_paths = [relative_to_repo(path, repo_root) for path in args.evaluation_inputs]
        merge_evaluation_runs(evaluation_paths, output_dir, force=args.force)


def run_summarize_results(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    root = relative_to_repo(args.root, repo_root)
    output_dir = ensure_directory(relative_to_repo(args.output_dir, repo_root))
    generation_catalog_path, evaluation_catalog_path = write_run_catalogs(root, output_dir)
    generation_runs = collect_generation_run_catalog(root / "outputs" / "generation")
    evaluation_runs = collect_evaluation_run_catalog(root / "outputs" / "evaluation")

    print("Generation runs:")
    if not generation_runs:
        print("  none found")
    else:
        for row in generation_runs:
            print(
                "  "
                f"{row['pipeline_name']} "
                f"run_key={row['run_key']} "
                f"target={row['question_count_target']} "
                f"materialized={row['total_questions_materialized']} "
                f"success={row['success_count']} "
                f"failed={row['failed_count']} "
                f"path={row['output_dir']}"
            )

    print("\nEvaluation runs:")
    if not evaluation_runs:
        print("  none found")
    else:
        for row in evaluation_runs:
            print(
                "  "
                f"{row['pipeline_name']} "
                f"run_key={row['run_key']} "
                f"target={row['question_count_target']} "
                f"materialized={row['total_rows_materialized']} "
                f"success={row['success_count']} "
                f"failed={row['failed_count']} "
                f"n/a={row['not_applicable_count']} "
                f"path={row['output_dir']}"
            )

    print(f"\nWrote catalogs to:\n  {generation_catalog_path}\n  {evaluation_catalog_path}")


def run_archive_results(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    config = load_config(args.config, repo_root=repo_root)
    archive_dir = archive_active_results(config=config, repo_root=repo_root)
    print(f"Archived active result artifacts to:\n  {archive_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trustworthy domain QA benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Run response generation only.")
    generate.add_argument("--config", default="config/benchmark.yaml")
    generate.add_argument("--pipeline", required=True, choices=list_pipelines())
    generate.add_argument("--limit", type=int, default=None)
    generate.add_argument("--question-id", nargs="*", default=None)
    generate.add_argument("--subset", choices=["regression_v3"], default=None)
    generate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(func=run_generate)

    evaluate = subparsers.add_parser("evaluate", help="Run official ragas evaluation from saved generation outputs.")
    evaluate.add_argument("--config", default="config/benchmark.yaml")
    evaluate.add_argument("--input", required=True, help="Path to responses.jsonl or its parent directory.")
    evaluate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    evaluate.set_defaults(func=run_evaluate)

    evaluate_accuracy = subparsers.add_parser("evaluate-accuracy", help="Run local judge accuracy evaluation against gold answers.")
    evaluate_accuracy.add_argument("--config", default="config/benchmark.yaml")
    evaluate_accuracy.add_argument("--input", required=True, help="Path to responses.jsonl or its parent directory.")
    evaluate_accuracy.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    evaluate_accuracy.set_defaults(func=run_evaluate_accuracy)

    compare_accuracy = subparsers.add_parser("compare-accuracy", help="Build a combined answer-correctness comparison workbook.")
    compare_accuracy.add_argument("--inputs", nargs="+", required=True, help="answer_correctness_rows.jsonl paths.")
    compare_accuracy.add_argument("--output", default="outputs/reports/final_pipeline_comparison.xlsx")
    compare_accuracy.set_defaults(func=run_compare_accuracy)

    merge = subparsers.add_parser("merge-runs", help="Merge completed pipeline artifacts.")
    merge.add_argument("--generation-inputs", nargs="+", required=True)
    merge.add_argument("--evaluation-inputs", nargs="*", default=None)
    merge.add_argument("--output-dir", default="outputs/reports")
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=run_merge_runs)

    summarize = subparsers.add_parser("summarize-results", help="List available generation/evaluation runs.")
    summarize.add_argument("--root", default=".")
    summarize.add_argument("--output-dir", default="outputs/reports/run_catalogs")
    summarize.set_defaults(func=run_summarize_results)

    archive = subparsers.add_parser("archive-results", help="Move active benchmark artifacts into an unused archive folder.")
    archive.add_argument("--config", default="config/benchmark.yaml")
    archive.set_defaults(func=run_archive_results)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
