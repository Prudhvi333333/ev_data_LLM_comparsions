"""RAG evaluation framework package."""
"""Benchmark framework for trustworthy domain QA comparisons."""

__all__ = ["__version__"]

__version__ = "2.0.0"
from pathlib import Path
import sys

_VENDOR_DIR = Path(__file__).resolve().parents[1] / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
