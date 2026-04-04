from __future__ import annotations

import os
from urllib.parse import urlparse


def build_ollama_generate_url(configured_endpoint: str | None = None) -> str:
    host_value = str(os.getenv("OLLAMA_HOST") or "").strip()
    if host_value:
        if "://" not in host_value:
            host_value = f"http://{host_value}"
        parsed = urlparse(host_value)
        base_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return f"{base_url}/api/generate"

    fallback = str(configured_endpoint or "http://localhost:11434/api/generate").strip()
    if fallback.endswith("/api/generate"):
        return fallback
    if "://" not in fallback:
        fallback = f"http://{fallback}"
    return f"{fallback.rstrip('/')}/api/generate"
