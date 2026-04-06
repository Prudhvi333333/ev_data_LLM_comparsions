# Official Benchmark V2 Note

This note covers the frozen Prompt V2 local benchmark comparison set for:

- `qwen_norag`
- `qwen_rag`
- `gemma_norag`
- `gemma_rag`

## Freeze Status

- Prompt V2 was frozen before the official comparison runs.
- Retrieval settings, chunking, structured-op logic, and evaluation setup were kept fixed during the official four-run comparison.
- Earlier Prompt V1 / pilot outputs are **not** part of this official comparison set.

## Official Run Artifacts

Generation runs:

- `qwen_norag`: `outputs/generation/qwen_norag/196f13be46797fa1`
- `qwen_rag`: `outputs/generation/qwen_rag/ef879e20e3908884`
- `gemma_norag`: `outputs/generation/gemma_norag/140a8b54788bc6f1`
- `gemma_rag`: `outputs/generation/gemma_rag/7cafd038ee245e57`

Evaluation runs:

- `qwen_norag`: `outputs/evaluation/qwen_norag/7caf82ea01f070e8`
- `qwen_rag`: `outputs/evaluation/qwen_rag/ba333e7f9f88b35b`
- `gemma_norag`: `outputs/evaluation/gemma_norag/32ee677103fc2e46`
- `gemma_rag`: `outputs/evaluation/gemma_rag/6c0a840469d71812`

Merged comparison outputs:

- `outputs/reports/official_v2_comparison/merged_generation_runs.xlsx`
- `outputs/reports/official_v2_comparison/merged_evaluation_runs.xlsx`
- `outputs/reports/official_v2_comparison/per_question_comparison.csv`
- `outputs/reports/official_v2_comparison/aggregate_metrics_summary.csv`
- `outputs/reports/official_v2_comparison/aggregate_metrics_summary_clean.csv`

Audit files:

- `outputs/reports/qwen_v2_audit_20260405.md`
- `outputs/reports/gemma_v2_audit_20260405.md`

## Aggregate Metrics

Official aggregate metrics from `aggregate_metrics_summary.csv`:

| pipeline | faithfulness | response_relevancy | context_precision | context_recall | answer_correctness |
|---|---:|---:|---:|---:|---:|
| `qwen_norag` | n/a | 0.4520 | n/a | n/a | 0.4731 |
| `qwen_rag` | 0.7051 | 0.6163 | 0.6200 | 0.5859 | 0.4757 |
| `gemma_norag` | n/a | 0.2469 | n/a | n/a | 0.4854 |
| `gemma_rag` | 0.6149 | 0.5163 | 0.6200 | 0.5685 | 0.4563 |

Interpretation:

- `qwen_rag` is the strongest local RAG pipeline in this frozen V2 comparison on the available aggregate metrics.
- `gemma_rag` is competitive but trails `qwen_rag` on faithfulness, response relevancy, context recall, and answer correctness. Context precision is effectively tied.
- `qwen_norag` is the stronger no-RAG baseline on response relevancy.
- `gemma_norag` is slightly higher on answer correctness than `qwen_norag`, but it is much less relevant on average.

## Model-Behavior Bucket

These are findings that look more like model behavior than shared pipeline logic:

- `qwen_rag` is more prone to false negatives or over-conservative `not found` style failures on some hard questions.
- `gemma_rag` is more prone to prompt/instruction leakage on hard rows, including raw JSON-style spill or malformed answer fragments.
- The no-RAG baselines are intentionally conservative. They should be interpreted as weak closed-book baselines, not knowledge-complete systems.

## Pipeline Bucket

These are known issues that should **not** be over-interpreted as model superiority:

- `Q13` and `Q15`: retrieval-slice / structured-filter mismatch
- `Q14` and `Q24`: alias / ontology gap for battery-material and enclosure-style questions
- `Q35`: confirmed structured-op counting bug

These are tagged in `per_question_comparison.csv` under `known_issue_bucket`.

## Practical Reading of V2

- Prompt V2 was the correct freeze point: it fixed the earlier structured-evidence obedience problem on questions like `Q8`.
- The remaining errors are now much easier to separate into:
  - model-specific answer behavior
  - shared pipeline/data-selection issues
- That separation is exactly what this benchmark was intended to make visible.

## Next-Version Backlog

These are appropriate V3 targets, but they were intentionally **not** changed during the official V2 comparison:

- sanitize final answers to prevent prompt/JSON leakage
- tighten retrieval slice selection for OEM-linked and narrow lookup questions
- improve alias dictionaries / ontology coverage for battery materials, enclosure, and harness-related terms
- fix the deterministic counting logic behind `Q35`
