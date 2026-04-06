# Gemma 27B Prompt V2 Audit

Official prompt-v2 generation runs:

- `gemma_norag`: `outputs/generation/gemma_norag/140a8b54788bc6f1`
- `gemma_rag`: `outputs/generation/gemma_rag/7cafd038ee245e57`

Both runs completed with `50/50` successful rows and `prompt_version=v2`.

## Gemma No-RAG Audit

Operationally, the run is clean. Behaviorally, this is a conservative but honest closed-book baseline. All 50 answers begin with the uncertainty sentence, and then usually add high-level reasoning. Unlike the old Prompt V1 behavior, it no longer collapses into a one-line refusal, but it still does not provide dataset-grounded entity lists or counts.

Pattern snapshot:

- `50/50` rows begin with the closed-book uncertainty sentence
- `32/50` include a `Generally...` best-effort explanation
- `8/50` include an explicit `However...` continuation
- No sampled rows hallucinated exact Georgia company lists, exact counts, or fake OEM mappings

| question_id | question | generated_answer | gold_answer | failure_type | short note |
|---|---|---|---|---|---|
| 1 | Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service. | Cannot determine exact dataset answer; gives a general description of typical Tier 1/2 EV supplier roles. | 18 named Tier 1/2 Georgia suppliers with role and product/service. | `partial_answer_only` | Safe and non-fabricated, but still provides no usable benchmark answer. |
| 8 | Which county have the highest total Employment among Tier 1 suppliers only? | Cannot determine exact dataset answer; says such employment would likely be concentrated near major manufacturing facilities. | Troup County, 2,435 employees. | `honest_uncertainty` | Honest closed-book limitation, but no candidate answer. |
| 10 | Identify all Vehicle Assembly facilities in Georgia and list the corresponding Primary OEM associated with each facility. | Cannot determine exact dataset answer; gives a general statement about OEM-linked assembly facilities. | 10 facility-to-OEM mappings. | `partial_answer_only` | No hallucinated facility list, but not useful for the actual dataset task. |
| 13 | Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role. | Cannot determine exact dataset answer; explains generic supplier-tier structure. | 6 Rivian-linked Georgia companies by tier and role. | `partial_answer_only` | Better than silence, but still entirely unguided by the benchmark data. |
| 14 | Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil, and what tier are they classified as? | Cannot determine exact dataset answer; explains general battery-material tiering. | Duckyang, F&P Georgia Manufacturing, Hollingsworth & Vose Co., IMMI. | `partial_answer_only` | Safe but not entity-complete. |
| 15 | Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their Primary OEMs. | Cannot determine exact dataset answer; gives generic commentary about wiring-harness suppliers and OEMs. | WIKA USA and Woodbridge Foam Corp. | `partial_answer_only` | No fabricated OEMs, but still no answer to the actual question. |
| 17 | Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms? | Cannot determine exact dataset answer; says such suppliers are often Tier 1/Tier 2 firms near automotive hubs. | WIKA USA and Woodbridge Foam Corp. | `partial_answer_only` | Honest reasoning, but no useful entity identification. |
| 23 | Which Georgia companies provide powder coating-related products or services, and what tier are they classified under? | Cannot determine exact dataset answer; describes how powder-coating suppliers are typically tiered. | Archer Aviation Inc. [Tier 2/3]. | `partial_answer_only` | Safe and benchmark-fair, but very low utility. |
| 24 | Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are linked to. | Cannot determine exact dataset answer; describes what Tier 1/2 battery-part firms would generally look like. | 6 Tier 1/2 companies linked to Hyundai Kia. | `partial_answer_only` | Still cannot produce dataset-specific names or OEM mappings. |
| 27 | Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem? | Cannot determine exact dataset answer; explains what a single-point-of-failure analysis would require. | 28 sole-source EV roles. | `partial_answer_only` | Reasonable conceptual answer, but not a true dataset response. |
| 35 | How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes? | Cannot determine exact dataset answer; says it is likely several companies are involved. | 5 companies. | `partial_answer_only` | Good anti-hallucination behavior, but no count. |
| 43 | Which Georgia companies are involved in battery recycling or second-life battery processing, reflecting the emerging circular economy trend? | Cannot determine exact dataset answer; discusses the circular-economy trend and possible recycling programs. | Enplas USA Inc., EVCO Plastics, F&P Georgia Manufacturing. | `partial_answer_only` | Honest and safe, but not grounded enough to compete on benchmark accuracy. |
| 50 | Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers? | Cannot determine exact dataset answer; gives generic location logic such as Atlanta or manufacturing-heavy regions. | Gray, Jones County via Racemark International LLC. | `partial_answer_only` | Slightly more specific high-level speculation, but still no reliable dataset answer. |

## Gemma RAG Audit

Operationally, the run is also clean and the artifact wiring is correct:

- `50/50` rows successful
- `50/50` rows contain saved `retrieved_context_exact`
- `41/50` rows contain `structured_ops_used`

The RAG behavior is stronger than Gemma No-RAG and in some places stronger than Qwen RAG. It follows deterministic structured evidence well on direct aggregation questions like Q8 and Q9, and it handles simple list/mapping questions like Q10 cleanly. The biggest remaining issues are:

- retrieval-slice mistakes that no prompt can rescue
- battery-material / enclosure field confusion
- prompt leakage / malformed answer fragments on several hard rows
- the same over-broad deterministic count bug seen in Qwen on Q35

Pattern snapshot:

- `2` rows say `Not found in provided context.`: Q19, Q34
- `8` rows leak raw JSON or instruction fragments: Q17, Q23, Q27, Q30, Q38, Q40, Q43, Q44
- `3` rows include `<unused...>` prompt-fragment text: Q17, Q40, Q44
- Short direct-answer rows like Q8, Q35, Q48, Q50 are often terse but not necessarily wrong

| question_id | question | generated_answer | gold_answer | structured_ops_used | failure_type | short note |
|---|---|---|---|---|---|---|
| 1 | Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service. | Returns a long company-role-product list beginning with F&P, Fouts Brothers, Hitachi Astemo, Hitachi Astemo Americas, Hollingsworth & Vose, and others. | 18 Tier 1/2 companies with role and product/service. | `list_records` | `strong_alignment` | The answer tracks the expected format and appears broadly aligned with the gold list. |
| 8 | Which county have the highest total Employment among Tier 1 suppliers only? | `Troup County` | Troup County, 2,435 employees. | `county_employment_totals` | `fixed_in_v2` | Gemma follows the structured county totals correctly, though it omits the numeric total. |
| 9 | Which county has the highest total employment across all companies, and what is the combined employment in that county? | `Gwinnett County ... 253022.0` | Gwinnett County, 253,022. | `county_employment_totals` | `strong_alignment` | Correct county and correct total. |
| 10 | Identify all Vehicle Assembly facilities in Georgia and list the corresponding Primary OEM associated with each facility. | Returns the 10 facility-to-OEM pairs as a compact list. | 10 facility-to-OEM mappings. | `list_records` | `strong_alignment` | Good direct use of structured evidence. |
| 13 | Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role. | Returns only the `Rivian Automotive` company row from Atlanta. | 6 Rivian-linked suppliers across multiple tiers plus Suzuki Manufacturing of America Corp. | `oem_mapping` | `retrieval_slice_error` | The structured op filtered on `company = Rivian Automotive` instead of suppliers whose `Primary OEMs` include Rivian, so the model answered the wrong evidence slice. |
| 14 | Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil, and what tier are they classified as? | Returns DAEHAN Solution Georgia LLC, EVCO Plastics, SK Battery America, SungEel Recycling Park Georgia. | Duckyang, F&P Georgia Manufacturing, Hollingsworth & Vose Co., IMMI. | `list_records` | `field_confusion` | Similar to Qwen, the model follows a broad `Materials` slice rather than the gold battery-material subset. |
| 15 | Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their Primary OEMs. | Returns `Superior Essex Inc.` and `TE Connectivity` with OEMs. | WIKA USA and Woodbridge Foam Corp. | `list_records` | `field_confusion` | The wiring-harness intent was misrouted to the wrong evidence slice, so the answer is cleanly formatted but semantically wrong. |
| 17 | Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms? | Spills a prompt fragment and JSON list of OEM Supply Chain companies with employment, including WIKA USA and Woodbridge Foam buried inside. | WIKA USA and Woodbridge Foam Corp. | `list_records` | `prompt_leakage` | The model leaks raw instruction/JSON text instead of answering the question directly. This looks more model-formatting related than pure retrieval failure. |
| 23 | Which Georgia companies provide powder coating-related products or services, and what tier are they classified under? | Spills `List the company names and employment numbers` plus unrelated JSON company list. | Archer Aviation Inc. [Tier 2/3]. | `list_records` | `prompt_leakage` | Severe answer-format corruption and wrong-question drift. |
| 24 | Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are linked to. | Returns F&P plus several Hyundai-adjacent Tier 1/2 companies not in the gold answer. | F&P, Hitachi Astemo Americas, Hollingsworth & Vose, Honda Development & Manufacturing, Hyundai Motor Group, IMMI. | `oem_mapping` | `field_confusion` | The Tier 1/2 filter is respected, but the battery-parts/enclosure subset is not. |
| 27 | Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem? | Spills a JSON list of company names and employment numbers rather than roles served by only one company. | 28 sole-source EV roles. | `list_records` | `prompt_leakage` | The model loses the requested output structure entirely. |
| 35 | How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes? | `22` | 5 companies. | `count_matching_records` | `aggregation_error` | Same deterministic over-count as Qwen. This is a pipeline/structured-op bug, not a Gemma-only mistake. |
| 43 | Which Georgia companies are involved in battery recycling or second-life battery processing, reflecting the emerging circular economy trend? | Leaks a malformed instruction plus a long employment-filter JSON list. | Enplas USA Inc., EVCO Plastics, F&P Georgia Manufacturing. | `list_records` | `prompt_leakage` | Another recurring prompt-fragment failure on a semantically narrow question. |
| 50 | Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers? | `Jones County, Gray.` | Gray, Jones County via Racemark International LLC and its EV R&D product detail. | `list_records` | `partial_list` | Core location is correct, but the answer drops the supporting company and product details. |

## Summary

- `gemma_norag` is a safe closed-book baseline. It does not hallucinate exact dataset facts, but it remains too conservative to be competitive on benchmark accuracy.
- `gemma_rag` is materially stronger and follows structured evidence well on several direct questions, especially Q8, Q9, and Q10.
- Some failures are clearly pipeline-wide rather than Gemma-specific:
  - Q13 retrieval-slice error
  - Q35 deterministic count bug
  - battery-material / enclosure slice confusion in Q14 and Q24
- Gemma also shows a distinct model-side failure mode that Qwen did not show as strongly:
  - prompt/instruction leakage with raw JSON-style spillover on several hard rows
