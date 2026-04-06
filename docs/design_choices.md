# Design Choices

## Why structured operations are first-class

Many questions in this benchmark are not best answered by asking an LLM to mentally aggregate long noisy context.
The framework therefore computes deterministic evidence for:

- counts
- grouped totals
- top-N rankings
- company/location/facility mappings

The LLM still matters, but arithmetic and grouping do not have to be approximate when code can do them exactly.

## Why generation and evaluation are separated

This prevents several benchmark pathologies:

- evaluation-time re-retrieval
- drift between the context used for answering and the context scored by metrics
- conflating model failures with judge failures

## Why runs are resumable but row-safe

Rows marked `failed` are not considered complete. Resume only skips rows whose latest saved state is `success`.
This avoids false completeness and makes reruns fair.

## Why manifests store hashes

Artifacts store:

- config hash
- dataset hashes
- corpus hash
- prompt version

This makes later comparisons auditable and helps detect stale or mixed results.
