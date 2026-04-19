# Prompt Catalog

Active benchmark prompt version: `v3`

Archived official baseline prompt version: `v2`

Prompt templates live in `src/prompts.py`.

## RAG Pipelines

Used by:

- `qwen_rag`
- `gemma_rag`
- `gemini_rag`

Prompt behavior:

- uses only saved provided context
- treats `STRUCTURED EVIDENCE` as primary truth for counts, totals, rankings, grouped results, OEM-linked mappings, and exhaustive filtered outputs
- treats retrieved chunks as secondary support rather than a source for recomputing totals
- emphasizes exact question alignment and prohibits answering a different question
- emphasizes field fidelity for role, product/service, OEM, county, employment, and category/tier
- requires plain-text-only final answers with no JSON, tool traces, or copied instructions
- returns the configured not-found string only when evidence truly does not exist

## No-RAG Pipelines

Used by:

- `qwen_norag`
- `gemma_norag`
- `gemini_norag`

Prompt behavior:

- closed-book baseline
- allows best-effort reasoning
- forbids invented Georgia-specific companies, exact counts, OEM mappings, and dataset-only facts
- prefers partial but honest answers over universal abstention
- requires plain-text-only answers and forbids JSON / prompt echo leakage
- uses the configured not-found string only when no safe partial answer can be given
- keeps the answer concise and explicit about uncertainty
