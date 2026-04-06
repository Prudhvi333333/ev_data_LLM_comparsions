from __future__ import annotations

from pathlib import Path
import logging


_LOGGERS: dict[str, logging.Logger] = {}
_CONFIGURED = False
_LOG_FILE: Path | None = None


def configure_logging(log_dir: Path, run_name: str) -> Path:
    global _CONFIGURED, _LOG_FILE

    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = log_dir / f"{run_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    _LOGGERS.clear()
    _CONFIGURED = True
    return _LOG_FILE


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging(Path("outputs/logs"), "bootstrap")
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.propagate = True
    _LOGGERS[name] = logger
    return logger


def current_log_file() -> Path | None:
    return _LOG_FILE
