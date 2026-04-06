from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    kb: str = "data/kb/GNEM - Auto Landscape Lat Long Updated.xlsx"
    questions: str = "data/Human validated 50 questions.xlsx"
    ontology: str = "config/ontology_v3.yaml"
    output_root: str = "outputs"
    generation: str = "outputs/generation"
    evaluation: str = "outputs/evaluation"
    accuracy_evaluation: str = "outputs/accuracy_evaluation"
    reports: str = "outputs/reports"
    manifests: str = "outputs/manifests"
    indexes: str = "outputs/indexes"
    logs: str = "outputs/logs"
    unused: str = "unused"
    regression_subset: str = "data/regression/v3_regression_subset.json"


class ModelConfig(BaseModel):
    provider: Literal["ollama", "gemini"]
    model_name: str
    endpoint: str | None = None
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_seconds: float = 180.0
    num_ctx: int = 4096
    top_p: float = 0.9
    repeat_penalty: float = 1.05


class RetrievalConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    allow_remote_model_downloads: bool = False
    allow_hash_embedding_fallback: bool = True
    dense_dimension: int = 2048
    top_k_dense: int = 24
    top_k_sparse: int = 24
    top_k_fused: int = 18
    top_k_reranked: int = 8
    max_context_tokens: int = 1800
    rrf_k: int = 60
    chunk_overlap: int = 0


class StructuredOpsConfig(BaseModel):
    enabled: bool = True
    top_n_default: int = 5
    max_evidence_rows: int = 25
    supplier_only_default: bool = True
    employment_field: str = "employment"
    ontology_match_mode: Literal["any", "all"] = "any"


class PromptConfig(BaseModel):
    version: str = "v3"
    rag_not_found_text: str = "Not found in provided context."
    norag_not_found_text: str = (
        "I cannot determine the exact dataset-specific answer from closed-book knowledge alone."
    )
    answer_style: Literal["plain_text"] = "plain_text"


class QuestionSubsetConfig(BaseModel):
    regression_v3: str = "data/regression/v3_regression_subset.json"


class EvaluationConfig(BaseModel):
    enabled_metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "response_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]
    )
    provider: Literal["gemini", "ollama"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    embeddings_provider: Literal["gemini", "ollama", "huggingface"] = "gemini"
    embeddings_model: str = "models/text-embedding-004"
    endpoint: str | None = None
    timeout_seconds: float = 180.0
    retries: int = 2
    raise_exceptions: bool = False
    batch_size: int = 1
    allow_optional_metrics: bool = True


class AccuracyEvaluationConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    judge_model: str = "llama3.1:8b"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    temperature: float = 0.0
    max_tokens: int = 220
    timeout_seconds: float = 180.0
    retries: int = 3


class RuntimeConfig(BaseModel):
    resume: bool = True
    write_excel: bool = True
    write_csv: bool = True
    fail_fast: bool = False
    random_seed: int = 7


class ApiKeysConfig(BaseModel):
    gemini: str = ""


class BenchmarkConfig(BaseModel):
    project_name: str = "benchmark_v3"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    structured_ops: StructuredOpsConfig = Field(default_factory=StructuredOpsConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    accuracy_evaluation: AccuracyEvaluationConfig = Field(default_factory=AccuracyEvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    question_subsets: QuestionSubsetConfig = Field(default_factory=QuestionSubsetConfig)

    @model_validator(mode="after")
    def validate_models(self) -> "BenchmarkConfig":
        required = {"qwen", "gemma", "gemini"}
        missing = sorted(required - set(self.models))
        if missing:
            raise ValueError(f"Missing model configs: {', '.join(missing)}")
        return self

    def ensure_directories(self, repo_root: Path) -> None:
        for raw_path in (
            self.paths.generation,
            self.paths.evaluation,
            self.paths.accuracy_evaluation,
            self.paths.reports,
            self.paths.manifests,
            self.paths.indexes,
            self.paths.logs,
            self.paths.unused,
        ):
            (repo_root / raw_path).mkdir(parents=True, exist_ok=True)


PIPELINE_REGISTRY: dict[str, tuple[str, bool]] = {
    "qwen_rag": ("qwen", True),
    "qwen_norag": ("qwen", False),
    "gemma_rag": ("gemma", True),
    "gemma_norag": ("gemma", False),
    "gemini_rag": ("gemini", True),
    "gemini_norag": ("gemini", False),
}
