#!/usr/bin/env bash
set -euo pipefail

python main.py generate --pipeline qwen_rag
python main.py generate --pipeline qwen_norag
python main.py generate --pipeline gemma_rag
python main.py generate --pipeline gemma_norag
python main.py generate --pipeline gemini_rag
python main.py generate --pipeline gemini_norag
