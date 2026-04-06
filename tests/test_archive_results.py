from __future__ import annotations

from pathlib import Path

from src.archive_results import archive_active_results
from src.config_models import BenchmarkConfig, ModelConfig


def test_archive_results_moves_active_outputs_and_recreates_directories(tmp_path: Path) -> None:
    for relative_path in (
        "outputs/generation/sample/file.txt",
        "outputs/evaluation/sample/file.txt",
        "outputs/reports/sample/file.txt",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")

    config = BenchmarkConfig(
        models={
            "qwen": ModelConfig(provider="ollama", model_name="qwen2.5:14b"),
            "gemma": ModelConfig(provider="ollama", model_name="gemma3:27b"),
            "gemini": ModelConfig(provider="gemini", model_name="gemini-2.5-flash"),
        }
    )
    config.ensure_directories(tmp_path)

    archive_root = archive_active_results(config=config, repo_root=tmp_path)

    assert (archive_root / "outputs" / "generation" / "sample" / "file.txt").exists()
    assert (archive_root / "outputs" / "evaluation" / "sample" / "file.txt").exists()
    assert (archive_root / "outputs" / "reports" / "sample" / "file.txt").exists()
    assert (tmp_path / "outputs" / "generation").exists()
    assert not any((tmp_path / "outputs" / "generation").iterdir())
