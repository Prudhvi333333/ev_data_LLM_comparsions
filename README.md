# Trustworthy Domain QA Benchmark

This repository benchmarks six QA pipelines over the Georgia EV supply-chain knowledge base:

1. `qwen_rag`
2. `qwen_norag`
3. `gemma_rag`
4. `gemma_norag`
5. `gemini_rag`
6. `gemini_norag`

The benchmark is intentionally split into two phases:

- Phase 1: generation only
- Phase 2: evaluation only with official `ragas`

That split makes it much easier to debug whether quality differences come from retrieval, prompts, structured operations, model capability, or evaluation settings.

The active default configuration now targets the structured-first `v3` benchmark architecture. The official frozen `v2` local benchmark is preserved as an archived baseline and its artifacts were moved under `unused/legacy_results_*`.

## Data

Canonical data files:

- Knowledge base: [data/kb/GNEM - Auto Landscape Lat Long Updated.xlsx](data/kb/GNEM%20-%20Auto%20Landscape%20Lat%20Long%20Updated.xlsx)
- Questions + human validated answers: [data/Human validated 50 questions.xlsx](data/Human%20validated%2050%20questions.xlsx)

Legacy run artifacts were archived under `unused/`.
The active `outputs/` directories are intentionally kept clean for fresh `v3` runs.

## Repository Layout

```text
config/
data/
docs/
outputs/
scripts/
src/
tests/
unused/
```

Key modules:

- `src/ingestion.py`
- `src/chunking.py`
- `src/ontology.py`
- `src/query_router.py`
- `src/embeddings.py`
- `src/vectorstore.py`
- `src/lexical_retrieval.py`
- `src/retriever.py`
- `src/reranker.py`
- `src/structured_ops.py`
- `src/answer_sanitizer.py`
- `src/prompts.py`
- `src/generators/`
- `src/pipelines/`
- `src/evaluation/`
- `src/reporting/`
- `src/cli.py`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional but recommended:

- Pull local Ollama models used for generation.
- Set `GEMINI_API_KEY` in `.env` if you want Gemini generation or Gemini-backed RAGAS evaluation.

## Generation Phase

Run one pipeline at a time:

```bash
python main.py generate --pipeline qwen_rag
python main.py generate --pipeline qwen_norag
python main.py generate --pipeline gemma_rag
python main.py generate --pipeline gemma_norag
python main.py generate --pipeline gemini_rag
python main.py generate --pipeline gemini_norag
```

Useful options:

```bash
python main.py generate --pipeline qwen_rag --limit 10
python main.py generate --pipeline qwen_rag --question-id 8 9 10
python main.py generate --pipeline qwen_rag --subset regression_v3
python main.py generate --pipeline qwen_rag --no-resume
```

Generation outputs are written to:

```text
outputs/generation/<pipeline>/<run_key>/
  responses.jsonl
  responses.csv
  responses.xlsx
  run_manifest.json
  generation_summary.json
  generation_error_analysis.csv
  retrieval_diagnostics.csv    # RAG pipelines
```

## Evaluation Phase

Evaluate a saved generation run later without regenerating or re-retrieving:

```bash
python main.py evaluate --input outputs/generation/qwen_rag/<run_key>
python main.py evaluate --input outputs/generation/qwen_norag/<run_key>/responses.jsonl
```

Evaluation outputs are written to:

```text
outputs/evaluation/<pipeline>/<run_key>/
  evaluation_rows.jsonl
  evaluation_rows.csv
  evaluation_rows.xlsx
  evaluation_manifest.json
  evaluation_summary.json
```

## Merge And Compare

Merge completed generation and evaluation runs:

```bash
python main.py merge-runs \
  --generation-inputs \
    outputs/generation/qwen_rag/<run_key>/responses.jsonl \
    outputs/generation/qwen_norag/<run_key>/responses.jsonl \
    outputs/generation/gemma_rag/<run_key>/responses.jsonl \
    outputs/generation/gemma_norag/<run_key>/responses.jsonl \
    outputs/generation/gemini_rag/<run_key>/responses.jsonl \
    outputs/generation/gemini_norag/<run_key>/responses.jsonl \
  --evaluation-inputs \
    outputs/evaluation/qwen_rag/<run_key>/evaluation_rows.jsonl \
    outputs/evaluation/qwen_norag/<run_key>/evaluation_rows.jsonl \
    outputs/evaluation/gemma_rag/<run_key>/evaluation_rows.jsonl \
    outputs/evaluation/gemma_norag/<run_key>/evaluation_rows.jsonl \
    outputs/evaluation/gemini_rag/<run_key>/evaluation_rows.jsonl \
    outputs/evaluation/gemini_norag/<run_key>/evaluation_rows.jsonl
```

This creates:

- per-question comparison tables
- aggregate metric summary
- per-pipeline success-rate and latency summaries

The merge step refuses incomplete runs unless `--force` is used.

## Why This Design

### 1. Two-phase separation

Generation and evaluation are intentionally decoupled. This prevents:

- hidden re-retrieval during evaluation
- accidental answer regeneration
- mixing runtime failures with metric failures

### 2. Hybrid QA, not text-only RAG

The benchmark does not rely only on plain text retrieval for aggregation-heavy questions.
It combines:

- field-aware row chunks
- dense retrieval
- BM25 lexical retrieval
- reciprocal-rank fusion
- reranking
- a structured data layer with normalized helper fields
- ontology-backed query routing
- deterministic structured operations for filtering, grouping, counting, ranking, OEM-network mapping, and sole-source role analysis
- a final answer sanitizer to strip JSON/prompt leakage before artifacts are saved

In `v3`, deterministic structured outputs are the primary truth source for counts, totals, rankings, exhaustive filtered lists, and mapped supplier networks. Retrieval is supporting evidence rather than the main calculator.

### 3. Reproducibility

Each run stores:

- config snapshot
- prompt version
- dataset hashes
- corpus hash
- exact presented context
- retrieved chunk ids and metadata

## Prompts

Prompt templates live in [src/prompts.py](src/prompts.py).
There is also a short prompt catalog in [docs/prompt_catalog.md](docs/prompt_catalog.md).

- Active prompt version: `v3`
- Archived official baseline prompt version: `v2`
- RAG prompt: structured-evidence-first, plain-text-only, field-faithful
- No-RAG prompt: conservative closed-book baseline with honest uncertainty and no fabricated dataset specifics

Prompt changes are intentionally modest in `v3`; the main improvements are now in structured routing, ontology-backed filtering, and answer cleanup.

## V2 Recovery

The archived `v2` baseline can still be referenced via:

- config snapshot: `config/benchmark_v2.yaml`
- archived artifacts: `unused/legacy_results_*`

The active default config remains `config/benchmark.yaml`, which now targets `v3`.

## Regression Subset

The repo includes a named hard-question regression subset:

- `regression_v3`

It currently covers:

- Q13
- Q14
- Q15
- Q17
- Q23
- Q24
- Q27
- Q35
- Q43
- Q50

Use it before full reruns:

```bash
python main.py generate --pipeline qwen_rag --subset regression_v3 --no-resume
python main.py generate --pipeline gemma_rag --subset regression_v3 --no-resume
```

## Run Catalogs

To enumerate available generation and evaluation runs and write catalog tables:

```bash
python main.py summarize-results
```

## Output Schemas

See [docs/output_schema.md](docs/output_schema.md).

## Adding New Models Later

1. Add a new model entry in `config/benchmark.yaml`.
2. Implement a generator in `src/generators/`.
3. Extend `PIPELINE_REGISTRY` in `src/config_models.py`.
4. Re-run generation and evaluation as separate phases.

Because generation artifacts are fully materialized, new models can be added without changing prior evaluation outputs.

## Testing

```bash
pytest tests -q
```

Current tests cover:

- retrieval behavior
- ontology matching
- structured operations on known hard questions
- answer sanitization
- archive behavior
- RAGAS dataset building
