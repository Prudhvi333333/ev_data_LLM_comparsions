# Qwen RAG Full Run Manual Audit

## Run Outputs
- Report: `outputs/reports/QWEN_RAG_FINAL_FULLRUN_20260405_173522.xlsx`
- Comparison: `outputs/reports/QWEN_RAG_FINAL_COMPARISON_FULLRUN_20260405_173522.xlsx`

## Summary
- Total questions: 50
- Timeouts: 0
- Mean final score: 0.5696
- Median final score: 0.6173

## Manual Alignment Buckets (Generated vs Golden)

### Strong alignment
Q1, Q4, Q6, Q9, Q10, Q11, Q12, Q17, Q18, Q19, Q24, Q26, Q32, Q37, Q42

### Partial alignment
Q2, Q3, Q5, Q7, Q8, Q14, Q15, Q16, Q20, Q21, Q22, Q23, Q25, Q28, Q29, Q30, Q31, Q35, Q39, Q40, Q41, Q43, Q50

### Weak or mismatched
Q13, Q27, Q33, Q34, Q36, Q38, Q44, Q45, Q46, Q47, Q48, Q49

## Context Quality (Golden-to-Context Lexical Coverage)

### Strong context evidence (>= 0.50)
Q1, Q2, Q3, Q4, Q5, Q10, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q20, Q21, Q22, Q23, Q24, Q26, Q28, Q29, Q30, Q31, Q32, Q33, Q35, Q39, Q40, Q41, Q42, Q50

### Medium context evidence (0.30 - 0.49)
Q6, Q8, Q11, Q19, Q25, Q27, Q34, Q36, Q37, Q43

### Weak context evidence (< 0.30)
Q7, Q9, Q38, Q44, Q45, Q46, Q47, Q48, Q49

## Metric Sanity Check
- No metric-evaluation error markers found in the report.
- Score-to-answer-overlap correlation is positive (0.613), and score-to-context-overlap correlation is positive (0.535).
- This indicates score trends are directionally consistent with answer quality and retrieval quality, not random.
- Notable low-score rows with good context overlap suggest generation/aggregation issues rather than retrieval failure: Q13, Q20, Q22, Q25, Q33, Q39, Q50.

## Architectural Read
- Pipeline is stable end-to-end (0 timeouts in full run).
- Hybrid retrieval is generally good for direct entity/attribute lookups and constrained filtering.
- The weakest area remains global aggregation/set-difference/risk-logic style questions, where retrieved context is often broad but answer synthesis underperforms.
