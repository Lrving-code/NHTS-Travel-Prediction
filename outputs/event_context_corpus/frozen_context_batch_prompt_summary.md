# Batched LLM Prompt Summary

- Source cohort profiles: `outputs\llm_event_features\household_cohort_profiles.csv`
- Cohorts: `1327`
- Batches: `89`
- Batch size target: `15`
- Max cohorts in a batch: `15`
- Max message chars: `34304`
- Rough total input token estimate: `742072`
- Request reduction vs one-cohort prompts: `93.29%`
- Output JSONL: `outputs\event_context_corpus\frozen_context_batch_prompts.jsonl`

## Recommended Use

Use this when the LLM endpoint supports long context and reliable JSON output. Run a small pilot first, validate every returned record, then scale batch size upward. For the current 1327 cohorts, batch size 8 reduces requests to about 166; batch size 15 reduces them to about 89 but increases validation risk.

## Why This Is Better

- Fewer API calls and less scheduling overhead.
- Cohorts are still auditable and label-free.
- The LLM can compare related cohorts inside a batch, which may improve ranking consistency.
- The output remains a structured event-prior table, not direct trip-count prediction.
