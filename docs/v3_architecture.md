# V3 Architecture

`v3` shifts the benchmark toward structured QA rather than prompt-centric rescue.

## Layer 1: Structured Data Layer

The Excel workbook remains the source of truth. During ingestion the repo now creates normalized helper fields for:

- company
- category / tier
- city / county / location
- EV supply-chain role
- product / service
- facility type
- Primary OEMs
- employment
- supplier flags

The loader also annotates each row with ontology tags from `config/ontology_v3.yaml`.

## Layer 2: Query / Evidence Routing

`src/query_router.py` routes questions into deterministic answer modes such as:

- county employment ranking
- OEM-linked supplier network
- ontology-filtered list
- ontology-backed count
- sole-source role analysis
- R&D facility lookup
- generic filtered list / lookup

For structured questions, retrieval is optional supporting evidence rather than the main truth source.

## Layer 3: LLM Answer Layer

The LLM receives:

- the user question
- a structured evidence block
- a small supporting evidence slice when helpful
- an explicit answer-field schema

`src/answer_sanitizer.py` then strips JSON spill, copied prompt sections, and tool/instruction leakage before the answer is saved.

## Known V3 Target Fixes

The first V3 pass specifically targets:

- Q13: OEM-linked supplier-network routing
- Q14 / Q24: ontology-backed battery-material / battery-parts matching
- Q15 / Q17: wiring-harness and electrical-distribution routing
- Q35: stricter lithium-ion materials / cells / electrolyte counting
- Gemma-style leakage: final answer sanitization

## Baseline Preservation

- `config/benchmark.yaml` is the active `v3` config
- `config/benchmark_v2.yaml` preserves the archived `v2` baseline config
- historical `v2` artifacts were moved under `unused/legacy_results_*`
