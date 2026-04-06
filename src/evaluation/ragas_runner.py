from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_models import BenchmarkConfig
from src.evaluation.dataset_builder import build_evaluation_dataset
from src.schemas import EvaluationRecord, GenerationRecord


def _import_metric_from_candidates(metric_module: Any, candidates: list[str]) -> Any | None:
    for candidate in candidates:
        if hasattr(metric_module, candidate):
            value = getattr(metric_module, candidate)
            try:
                return value() if isinstance(value, type) else value
            except TypeError:
                return value
    return None


class RagasRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.ragas = self._import_ragas()
        self.metrics = self._build_metrics()
        self.llm = self._build_llm()
        self.embeddings = self._build_embeddings()
        self.run_config = self._build_run_config()

    @staticmethod
    def _import_ragas() -> Any:
        try:
            import ragas
            return ragas
        except Exception as exc:  # pragma: no cover - import guarded at runtime
            raise RuntimeError(
                "Official ragas is not installed. Install dependencies from requirements.txt."
            ) from exc

    def _build_metrics(self) -> dict[str, Any]:
        metric_module = __import__("ragas.metrics", fromlist=["dummy"])
        candidates = {
            "faithfulness": ["Faithfulness", "faithfulness"],
            "response_relevancy": ["ResponseRelevancy", "response_relevancy", "answer_relevancy"],
            "context_precision": ["ContextPrecision", "LLMContextPrecisionWithoutReference", "context_precision"],
            "context_recall": ["ContextRecall", "LLMContextRecall", "context_recall"],
            "answer_correctness": ["AnswerCorrectness", "answer_correctness"],
        }
        selected: dict[str, Any] = {}
        for metric_name in self.config.evaluation.enabled_metrics:
            metric = _import_metric_from_candidates(metric_module, candidates.get(metric_name, [metric_name]))
            if metric is not None:
                selected[metric_name] = metric
            elif not self.config.evaluation.allow_optional_metrics:
                raise RuntimeError(f"Ragas metric '{metric_name}' is unavailable in this installation.")
        return selected

    def _build_llm(self) -> Any:
        provider = self.config.evaluation.provider
        if provider == "gemini":
            if not self.config.api_keys.gemini:
                raise RuntimeError("Gemini API key is missing for evaluation.")
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.config.evaluation.llm_model,
                google_api_key=self.config.api_keys.gemini,
                temperature=0,
            )
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            base_url = str(self.config.evaluation.endpoint or "http://127.0.0.1:11434").replace("/api/generate", "")
            return ChatOllama(
                model=self.config.evaluation.llm_model,
                base_url=base_url,
                temperature=0,
            )
        raise RuntimeError(f"Unsupported evaluation provider: {provider}")

    def _build_embeddings(self) -> Any:
        provider = self.config.evaluation.embeddings_provider
        if provider == "gemini":
            if not self.config.api_keys.gemini:
                raise RuntimeError("Gemini API key is missing for evaluation embeddings.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=self.config.evaluation.embeddings_model,
                google_api_key=self.config.api_keys.gemini,
            )
        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            base_url = str(self.config.evaluation.endpoint or "http://127.0.0.1:11434").replace("/api/generate", "")
            return OllamaEmbeddings(
                model=self.config.evaluation.embeddings_model,
                base_url=base_url,
            )
        if provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except Exception:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.config.evaluation.embeddings_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        raise RuntimeError(f"Unsupported evaluation embeddings provider: {provider}")

    def _build_run_config(self) -> Any:
        try:
            from ragas import RunConfig

            return RunConfig(
                timeout=self.config.evaluation.timeout_seconds,
                max_retries=self.config.evaluation.retries,
            )
        except Exception:
            return None

    @staticmethod
    def _normalize_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        if "answer_relevancy" in normalized and "response_relevancy" not in normalized:
            normalized["response_relevancy"] = normalized["answer_relevancy"]
        return normalized

    def evaluate_record(self, record: GenerationRecord, input_path: Path) -> EvaluationRecord:
        if record.answer_status != "success":
            return EvaluationRecord(
                question_id=record.question_id,
                pipeline_name=record.pipeline_name,
                generation_output_path=str(input_path),
                evaluation_status="not_applicable",
                error_message=f"generation_status={record.answer_status}",
            )

        active_metrics = dict(self.metrics)
        if not record.rag_enabled:
            active_metrics = {
                key: value
                for key, value in active_metrics.items()
                if key in {"response_relevancy", "answer_correctness"}
            }

        if not active_metrics:
            return EvaluationRecord(
                question_id=record.question_id,
                pipeline_name=record.pipeline_name,
                generation_output_path=str(input_path),
                evaluation_status="not_applicable",
                error_message="no_applicable_metrics",
            )

        dataset = build_evaluation_dataset([record])
        result = self.ragas.evaluate(
            dataset=dataset,
            metrics=list(active_metrics.values()),
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=self.run_config,
            raise_exceptions=self.config.evaluation.raise_exceptions,
            show_progress=False,
            batch_size=self.config.evaluation.batch_size,
            allow_nest_asyncio=False,
        )

        if hasattr(result, "to_pandas"):
            raw_payload = result.to_pandas().iloc[0].to_dict()
        elif hasattr(result, "scores"):
            raw_payload = dict(result.scores[0])
        else:
            raw_payload = dict(result)
        normalized = self._normalize_metric_payload(raw_payload)

        return EvaluationRecord(
            question_id=record.question_id,
            pipeline_name=record.pipeline_name,
            generation_output_path=str(input_path),
            evaluation_status="success",
            faithfulness=normalized.get("faithfulness"),
            response_relevancy=normalized.get("response_relevancy"),
            context_precision=normalized.get("context_precision"),
            context_recall=normalized.get("context_recall"),
            answer_correctness=normalized.get("answer_correctness"),
            metric_payload=normalized,
        )
