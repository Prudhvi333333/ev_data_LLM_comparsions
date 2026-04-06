from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

from dotenv import load_dotenv
from pydantic import ValidationError
import yaml

from src.config_models import BenchmarkConfig, PIPELINE_REGISTRY
from src.utils.files import stable_hash_dict


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value.strip())
        if match:
            return os.environ.get(match.group(1), "")
        return os.path.expandvars(value)
    return value


def load_config(path: str | Path = "config/benchmark.yaml", repo_root: Path | None = None) -> BenchmarkConfig:
    repo_root = (repo_root or Path.cwd()).resolve()
    load_dotenv(repo_root / ".env")

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw = _expand_env_vars(raw)
        config = BenchmarkConfig.model_validate(raw)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid benchmark config: {exc}") from exc

    config.ensure_directories(repo_root)
    return config


def resolve_pipeline(pipeline_name: str) -> tuple[str, bool]:
    normalized = pipeline_name.strip().lower()
    if normalized not in PIPELINE_REGISTRY:
        valid = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ConfigError(f"Unknown pipeline '{pipeline_name}'. Valid pipelines: {valid}")
    return PIPELINE_REGISTRY[normalized]


def list_pipelines() -> list[str]:
    return sorted(PIPELINE_REGISTRY)


def config_hash(config: BenchmarkConfig) -> str:
    snapshot = config.model_dump()
    if "api_keys" in snapshot:
        snapshot["api_keys"] = {key: "***redacted***" for key in snapshot["api_keys"]}
    return stable_hash_dict(snapshot)


def relative_to_repo(path: str | Path, repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def model_config_for_pipeline(config: BenchmarkConfig, pipeline_name: str) -> dict[str, Any]:
    model_key, rag_enabled = resolve_pipeline(pipeline_name)
    return {
        "pipeline_name": pipeline_name,
        "model_key": model_key,
        "rag_enabled": rag_enabled,
        "model_config": config.models[model_key],
    }
