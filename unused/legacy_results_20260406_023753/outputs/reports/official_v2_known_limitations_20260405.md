# Official Benchmark V2 Known Limitations

This note applies to the frozen official local comparison set:

- `qwen_norag`
- `qwen_rag`
- `gemma_norag`
- `gemma_rag`

## Freeze Point

Prompt version `v2` was frozen before the official local runs. After that freeze point, the benchmark configuration was not changed in:

- prompts
- retrieval settings
- chunking
- structured-op logic
- evaluation setup

Earlier Qwen runs are pilot/debug runs and must not be mixed into the official `v2` leaderboard.

## Issue Buckets

### Model-Behavior Bucket

- `qwen_norag` and `gemma_norag` are intentionally conservative closed-book baselines and should be interpreted as weak anti-hallucination baselines, not knowledge-complete systems.
- `qwen_rag` shows more false negatives and more cases of failing to fully use available evidence.
- `gemma_rag` shows more prompt/instruction leakage on hard rows, including raw JSON-style or malformed answer spillover.

### Pipeline Bucket

- `Q13`: retrieval slice problem for OEM-linked supplier-network questions.
- `Q35`: confirmed structured-op counting bug; the deterministic count is over-broad and therefore not model-specific.
- alias / ontology gaps remain for battery-material, enclosure-style, and harness-related questions such as `Q14`, `Q15`, `Q17`, and `Q24`.

These pipeline-bucket failures should not be treated as clean evidence that one model is better than another.

## Evaluation Note

Official evaluation uses the frozen `v2` setup with official `ragas` and a local `llama3.1:8b` judge. During evaluation, the local judge can occasionally return parser-unfriendly text around JSON payloads, which may require resumed evaluation passes. This is an evaluation robustness limitation, not a prompt/retrieval change, and should be documented separately from generation quality.
