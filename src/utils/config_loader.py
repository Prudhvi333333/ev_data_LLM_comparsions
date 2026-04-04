from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when config.yaml is malformed or incomplete."""


_REQUIRED_PATHS = [
    ("models", "qwen"),
    ("models", "gemma"),
    ("models", "gemini"),
    ("api_keys", "gemini"),
    ("api_keys", "openrouter"),
    ("paths", "kb"),
    ("paths", "questions"),
    ("paths", "output"),
    ("paths", "chroma"),
    ("retrieval", "top_k"),
    ("retrieval", "candidate_pool"),
    ("retrieval", "embedding_model"),
    ("generation", "temperature"),
    ("generation", "max_tokens"),
    ("generation", "timeout_seconds"),
    ("evaluation", "provider"),
    ("evaluation", "judge_model"),
    ("evaluation", "weights"),
    ("concurrency", "generation_semaphore"),
    ("concurrency", "evaluation_semaphore"),
]


def _dict_to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(item) for item in value]
    return value


def _namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    return value


def _require_key(config_dict: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = config_dict
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise ConfigError(f"Missing required config key: {dotted}")
        current = current[key]


def _validate_api_keys(config_dict: dict[str, Any]) -> None:
    gemini_key = str(config_dict["api_keys"].get("gemini", "")).strip()
    if not gemini_key:
        raise ConfigError("Config api_keys.gemini is empty. Set it in config/config.yaml.")

    evaluation_provider = str(
        config_dict.get("evaluation", {}).get("provider", "openrouter")
    ).strip().lower()
    if evaluation_provider == "openrouter":
        openrouter_key = str(config_dict["api_keys"].get("openrouter", "")).strip()
        if not openrouter_key:
            raise ConfigError(
                "Config api_keys.openrouter is empty. Set it in config/config.yaml."
            )


def _ensure_output_dirs(config_dict: dict[str, Any]) -> None:
    for key_name in ("output", "chroma", "logs", "progress"):
        raw_path = config_dict["paths"].get(key_name)
        if raw_path:
            Path(raw_path).mkdir(parents=True, exist_ok=True)

    for file_key in ("kb", "questions"):
        data_path = Path(config_dict["paths"][file_key])
        data_path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> SimpleNamespace:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}

    if not isinstance(config_dict, dict):
        raise ConfigError("Config root must be a YAML mapping.")

    for required in _REQUIRED_PATHS:
        _require_key(config_dict, required)

    _validate_api_keys(config_dict)
    _ensure_output_dirs(config_dict)
    return _dict_to_namespace(config_dict)


def get_pipeline_id(model_key: str, mode: str) -> str:
    normalized_mode = mode.strip().lower().replace("-", "")
    return f"{model_key.strip().lower()}_{normalized_mode}"


def to_plain_dict(value: Any) -> Any:
    return _namespace_to_dict(value)
