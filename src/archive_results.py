from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from src.config_models import BenchmarkConfig
from src.utils.files import ensure_directory, write_json


def archive_active_results(config: BenchmarkConfig, repo_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_root = ensure_directory(repo_root / config.paths.unused / f"legacy_results_{timestamp}")

    source_paths = {
        "generation": repo_root / config.paths.generation,
        "evaluation": repo_root / config.paths.evaluation,
        "accuracy_evaluation": repo_root / config.paths.accuracy_evaluation,
        "reports": repo_root / config.paths.reports,
        "manifests": repo_root / config.paths.manifests,
        "logs": repo_root / config.paths.logs,
        "indexes": repo_root / config.paths.indexes,
    }

    moved: dict[str, str] = {}
    for key, source in source_paths.items():
        if not source.exists():
            continue
        if source.is_dir() and not any(source.iterdir()):
            continue
        destination = archive_root / "outputs" / key
        ensure_directory(destination.parent)
        shutil.move(str(source), str(destination))
        moved[key] = str(destination)
        ensure_directory(source)

    write_json(
        archive_root / "archive_manifest.json",
        {
            "archive_root": str(archive_root),
            "moved": moved,
        },
    )
    return archive_root
