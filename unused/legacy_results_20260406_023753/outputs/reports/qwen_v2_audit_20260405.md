# Qwen 14B Prompt V2 Audit

Official prompt-v2 generation runs:

- `qwen_norag`: `outputs/generation/qwen_norag/196f13be46797fa1`
- `qwen_rag`: `outputs/generation/qwen_rag/ef879e20e3908884`

Both runs completed with `50/50` successful rows and `prompt_version=v2`.

## Qwen No-RAG Audit

Operationally, the run is clean. Behaviorally, the model is still highly conservative, but it no longer collapses to a one-line abstention. All 50 answers begin with an honest uncertainty statement, then usually add a brief best-effort explanation or high-level reasoning.

| question_id | question | generated_answer | gold_answer | failure_type | short note |
|---|---|---|---|---|---|
| 1 | Show all "Tier 1/2" suppliers in Georgia, list their EV Supply Chain Role and Product / Service. | Cannot determine exact dataset answer; gives generic description of what Tier 1/2 suppliers usually do. | 18 named Georgia suppliers with role and product/service. | partial_answer_only | Improved over Prompt V1 because it adds a best-effort explanation, but still does not provide any dataset-grounded list. |
| 8 | Which county have the highest total Employment among Tier 1 suppliers only? | Cannot determine exact dataset answer; says it depends on county locations and facility sizes. | Troup County, 2,435 employees. | honest_uncertainty | No fabrication, but still no useful candidate answer for an aggregation question. |
| 10 | Identify all Vehicle Assembly facilities in Georgia and list the corresponding Primary OEM associated with each facility. | Cannot determine exact dataset answer; gives generic statement about vehicle assembly facilities and OEMs. | 10 facility-to-OEM mappings. | partial_answer_only | Safe but uninformative for a full mapping task. |
| 13 | Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role. | Cannot determine exact dataset answer; explains high-level supplier tier logic. | 6 Rivian-linked Georgia companies by tier and role. | partial_answer_only | Better than pure abstention, but still misses all requested entities. |
| 14 | Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil, and what tier are they classified as? | Cannot determine exact dataset answer; explains how tiering typically works. | 4 companies: Duckyang, F&P, Hollingsworth & Vose, IMMI. | partial_answer_only | Safe high-level reasoning, no grounded company names. |
| 15 | Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their Primary OEMs. | Cannot determine exact dataset answer; gives generic description of wiring-harness suppliers. | WIKA USA and Woodbridge Foam Corp. | partial_answer_only | No hallucinated specifics, but still unusable for benchmark accuracy. |
| 17 | Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms? | Cannot determine exact dataset answer; gives high-level description of likely supplier types. | WIKA USA and Woodbridge Foam Corp. | partial_answer_only | Best-effort reasoning now present, but no concrete answer. |
| 23 | Which Georgia companies provide powder coating-related products or services, and what tier are they classified under? | Cannot determine exact dataset answer; explains how such firms might be classified. | Archer Aviation Inc. [Tier 2/3]. | partial_answer_only | No fabricated company names, but also no useful retrieval-free answer. |
| 27 | Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem? | Cannot determine exact dataset answer; explains the concept of single-point-of-failure risk. | 28 sole-source EV supply-chain roles. | partial_answer_only | The new prompt allows honest synthesis, but closed-book still cannot answer the actual dataset question. |
| 35 | How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes? | Cannot determine exact dataset answer; says there are likely several such companies. | 5 companies. | partial_answer_only | No fake count, which is good; still not benchmark-useful beyond being a safe baseline. |
| 43 | Which Georgia companies are involved in battery recycling or second-life battery processing, reflecting the emerging circular economy trend? | Cannot determine exact dataset answer; gives generic circular-economy explanation. | Enplas USA Inc., EVCO Plastics, F&P Georgia Manufacturing. | partial_answer_only | Safe, honest, but not entity-complete. |
| 50 | Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers? | Cannot determine exact dataset answer; gives generic statement about automotive R&D locations. | Gray, Jones County via Racemark International LLC. | partial_answer_only | Still too conservative for closed-book benchmarking, though less degenerate than Prompt V1. |

## Qwen RAG Audit

Operationally, the run is clean and context persistence is correct: all 50 rows contain saved retrieved context and 41 rows contain structured-op outputs. Prompt V2 improved compliance with deterministic evidence for some aggregation questions, but several semantic failure modes remain.

| question_id | question | generated_answer | gold_answer | structured_ops_used | failure_type | short note |
|---|---|---|---|---|---|---|
| 8 | Which county have the highest total Employment among Tier 1 suppliers only? | Troup County with total employment 2435.0. | Troup County, 2,435 employees. | `county_employment_totals` | fixed_in_v2 | Prompt V2 successfully forced use of structured evidence here. |
| 9 | Which county has the highest total employment across all companies, and what is the combined employment in that county? | Gwinnett County, 253,022. | Gwinnett County, 253,022. | `county_employment_totals` | fixed_in_v2 | Another strong structured-aggregation answer under Prompt V2. |
| 13 | Show the full supplier network linked to Rivian Automotive in Georgia, broken down by tier and EV Supply Chain Role. | Not found in provided context. | 6 Rivian-linked suppliers across Tier 2/3, Tier 1/2, Tier 1, and OEM. | `oem_mapping` | false_negative | Structured op filtered on `company = Rivian Automotive` instead of finding suppliers linked via `Primary OEMs`, so the prompt cannot rescue the wrong evidence slice. |
| 14 | Which Georgia companies produce battery materials such as anodes, cathodes, electrolytes, or copper foil, and what tier are they classified as? | Returns DAEHAN Solution Georgia LLC, SungEel Recycling Park Georgia, EVCO Plastics. | Duckyang, F&P Georgia Manufacturing, Hollingsworth & Vose Co., IMMI. | `list_records` | field_confusion | Alias handling improved in prompt wording, but retrieval/structured filtering still maps the question to a broad `Materials` slice instead of the gold battery-material subset. |
| 15 | Identify all Georgia companies with an EV Supply Chain Role related to wiring harnesses and show their Primary OEMs. | Not found in provided context. | WIKA USA and Woodbridge Foam Corp. | `list_records` | false_negative | Structured evidence was built from the wrong filter (`category = OEM`), leading to a confident miss. |
| 17 | Which Georgia companies manufacture high-voltage wiring harnesses or EV electrical distribution components suitable for BEV platforms? | Answers with Valeo and stamped-body assemblies. | WIKA USA and Woodbridge Foam Corp. | `list_records` | answered_wrong_question | The model latched onto irrelevant retrieved evidence and produced an off-target company despite the question being wiring-harness specific. |
| 23 | Which Georgia companies provide powder coating-related products or services, and what tier are they classified under? | Answers with Yamaha Motor Manufacturing Corp. and advanced electrical architecture. | Archer Aviation Inc. [Tier 2/3]. | `list_records` | answered_wrong_question | Severe drift: the answer is about the wrong company and wrong field entirely. |
| 24 | Which Georgia companies manufacture battery parts or enclosure systems and are classified as Tier 1/2, making them ready for direct OEM engagement and show which Primary OEMs they are linked to. | Lists Hyundai Transys, LGES, Hyundai Industrial, etc. | F&P, Hitachi Astemo Americas, Hollingsworth & Vose, Honda Development & Manufacturing, Hyundai Motor Group, IMMI. | `oem_mapping` | field_confusion | The answer respects the Tier 1/2 filter but selects battery-adjacent companies rather than the requested parts/enclosure subset. |
| 27 | Which EV Supply Chain Roles in Georgia are served by only a single company, creating a single-point-of-failure risk for the state's EV ecosystem? | Answers with Yamaha Motor Manufacturing Corp. only. | 28 single-company roles. | `list_records` | partial_list | The model captured one valid-looking example but failed the requested exhaustive synthesis. |
| 35 | How many Georgia companies are now producing lithium-ion battery materials, cells, or electrolytes? | 22 companies. | 5 companies. | `count_matching_records` | aggregation_error | Structured op itself is over-broad because it counts all `Materials` rows, so Prompt V2 cannot correct the wrong deterministic result. |
| 43 | Which Georgia companies are involved in battery recycling or second-life battery processing, reflecting the emerging circular economy trend? | Answers with Valeo and stamped-body assemblies. | Enplas USA Inc., EVCO Plastics, F&P Georgia Manufacturing. | `list_records` | answered_wrong_question | Same drift pattern as Q17: irrelevant retrieved evidence dominates the response. |
| 50 | Which Georgia areas have R&D facility types in the automotive sector, suggesting innovation infrastructure suitable for EV technology development centers? | Jones County, specifically Gray. | Gray, Jones County via Racemark International LLC and its EV R&D product detail. | `list_records` | partial_list | Core location is correct, but the answer drops the supporting company and product fields that the question context supports. |

## Summary

- Prompt V2 fixed the most visible structured-evidence obedience failure for Q8 and preserved the correct Q9 aggregation behavior.
- `qwen_norag` is no longer a degenerate one-line abstention baseline, but it remains extremely conservative and mostly non-competitive on dataset-specific QA.
- The largest remaining `qwen_rag` problems are not prompt-only. They are mostly:
  - wrong structured-op/filter selection
  - retrieval drift to irrelevant rows
  - over-broad alias/category matching for battery-material questions
  - incomplete list synthesis on exhaustiveness-heavy prompts
