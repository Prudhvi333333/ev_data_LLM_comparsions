#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

row_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

wait_or_resume() {
  local input_path="$1"
  local rows_path="$2"
  local process_pattern="$3"

  while [[ "$(row_count "$rows_path")" -lt 50 ]]; do
    if pgrep -f "$process_pattern" >/dev/null; then
      sleep 60
    else
      ./.venv/bin/python main.py evaluate --input "$input_path" --resume
    fi
  done
}

wait_or_resume \
  "outputs/generation/qwen_rag/ef879e20e3908884" \
  "outputs/evaluation/qwen_rag/ba333e7f9f88b35b/evaluation_rows.jsonl" \
  "main.py evaluate --input outputs/generation/qwen_rag/ef879e20e3908884 --resume"

wait_or_resume \
  "outputs/generation/gemma_norag/140a8b54788bc6f1" \
  "outputs/evaluation/gemma_norag/32ee677103fc2e46/evaluation_rows.jsonl" \
  "main.py evaluate --input outputs/generation/gemma_norag/140a8b54788bc6f1 --resume"

wait_or_resume \
  "outputs/generation/gemma_rag/7cafd038ee245e57" \
  "outputs/evaluation/gemma_rag/6c0a840469d71812/evaluation_rows.jsonl" \
  "main.py evaluate --input outputs/generation/gemma_rag/7cafd038ee245e57 --resume"
