# Frozen-Context Batched LLM Prior Generation Report

- Provider: `cursor_api`
- Model: `gpt-5.5-low`
- Input prompt file: `outputs\event_context_corpus\frozen_context_batch_prompts.jsonl`
- Selected batches: `1`
- Selected cohorts: `15`
- Valid generated cohort priors: `0`
- Failed batches: `1`
- Output directory: `outputs\llm_event_features\frozen_context_cursor_api_pilot`

## Interpretation

No validated priors were generated. Treat this artifact as a runnable frozen-context replay protocol, not as completed frozen-context prior evidence.

## Runtime Errors

| Batch | Error |
|---|---|
| runtime_auth | Missing auth token env var: CURSOR_API_AUTH_TOKEN |

## Paper Use

Use this as the retrieval/frozen-context replay evidence for the LLM event-prior branch. If only a pilot subset is generated, report it as provenance validation rather than a full replacement for the existing GPT-reference priors.