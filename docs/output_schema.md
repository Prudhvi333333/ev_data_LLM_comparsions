# Output Schemas

## Generation Row

Each row in `responses.jsonl` contains:

- `question_id`
- `question`
- `reference_answer`
- `pipeline_name`
- `model_name`
- `rag_enabled`
- `prompt_version`
- `system_prompt`
- `user_prompt`
- `retrieved_context_exact`
- `retrieved_chunks`
- `retrieved_chunk_ids`
- `retrieved_metadata`
- `reranked_chunk_ids`
- `structured_ops_used`
- `structured_ops_outputs`
- `final_context_presented_to_model`
- `query_route_name`
- `ontology_buckets`
- `generated_answer`
- `raw_generated_answer`
- `answer_sanitized`
- `answer_sanitizer_notes`
- `answer_status`
- `error_message`
- `latency_seconds`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `timestamp_utc`
- `attempt_count`
- `retrieval_diagnostics`
- `generation_metadata`

## Generation Manifest

`run_manifest.json` stores:

- pipeline and model identifiers
- target question count for the run
- config hash
- knowledge-base hash
- questions hash
- corpus hash
- chunking hash
- embedding backend
- repo commit
- config snapshot

## Evaluation Row

Each row in `evaluation_rows.jsonl` contains:

- `question_id`
- `pipeline_name`
- `generation_output_path`
- `evaluation_status`
- `faithfulness`
- `response_relevancy`
- `context_precision`
- `context_recall`
- `answer_correctness`
- `error_message`
- `timestamp_utc`
- `metric_payload`

## Evaluation Manifest

`evaluation_manifest.json` stores:

- pipeline identifier
- input generation artifact path
- target question count for the evaluation run
- enabled metrics
- config hash
- settings snapshot
