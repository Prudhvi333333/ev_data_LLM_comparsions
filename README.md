# RAG Evaluation Framework

This repository implements the full pipeline from `RAG_Evaluation_Blueprint.pdf`: KB ingestion, hybrid retrieval, RAG and No-RAG generation, RAGAS-style scoring, six per-pipeline Excel reports, resumable progress, and a final cross-pipeline comparison workbook.

The implementation keeps the blueprint's module boundaries and adds deterministic local fallbacks so the project remains runnable even when ChromaDB, Ollama, OpenRouter, or sentence-transformers are unavailable in the current environment.

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Pull Ollama models for online generation
ollama pull qwen2.5:14b
ollama pull gemma3:12b
```

Set API keys in [config/config.yaml](config/config.yaml):

- `api_keys.gemini`
- `api_keys.openrouter`

If you keep the bundled `local-dev-*` placeholder keys, the evaluator and Gemini generator use offline heuristic fallbacks instead of remote APIs.

Copy production data files into:

- [data/kb/gnem_auto_landscape.xlsx](data/kb/gnem_auto_landscape.xlsx)
- [data/questions/questions_50.xlsx](data/questions/questions_50.xlsx)

If those files are missing, the code auto-generates a synthetic 205-row KB and 50-question workbook so the pipeline and tests still run.

## Run
```bash
# Full run (all 6 pipelines)
python main.py

# Single pipeline
python main.py --pipeline qwen_rag

# Multiple selected pipelines
python main.py --pipeline qwen_rag gemini_norag

# Resume interrupted runs
python main.py --resume

# Disable resume checkpoints
python main.py --no-resume

# Force KB reindex
python main.py --force-reindex
```

## Outputs
- Per-pipeline reports: [outputs/reports](outputs/reports)
- Final comparison workbook: `outputs/reports/FINAL_COMPARISON.xlsx`
- Checkpoints: [outputs/progress](outputs/progress)
- Logs: [outputs/logs](outputs/logs)
- Persistent local index + BM25 pickle: [outputs/chroma_db](outputs/chroma_db)

## Test
```bash
python -m unittest discover tests -p "test_*.py" -v

# If pytest is installed
pytest tests/ -v --tb=short
```

## Configuration
All runtime parameters live in [config/config.yaml](config/config.yaml), including model names, paths, retrieval settings, metric weights, generation timeout, and concurrency controls.

## Module Map
- [src/kb_loader.py](src/kb_loader.py): Excel KB loading and document construction
- [src/indexer.py](src/indexer.py): persistent vector index plus BM25 fallback
- [src/retriever.py](src/retriever.py): hybrid retrieval, intent detection, reranking
- [src/context_compressor.py](src/context_compressor.py): token-budget context trimming
- [src/generator.py](src/generator.py): Qwen, Gemma, Gemini generation with local fallback
- [src/evaluator.py](src/evaluator.py): Kimi K2 judge integration plus heuristic fallback
- [src/reporter.py](src/reporter.py): per-pipeline Excel report generation
- [src/summary_reporter.py](src/summary_reporter.py): `FINAL_COMPARISON.xlsx`
- [main.py](main.py): async orchestration, checkpoints, CLI, and final summary
