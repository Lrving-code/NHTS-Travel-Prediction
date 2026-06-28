# Frozen-Context Single-Cohort Prompt Summary

- Source cohort profiles: `outputs\llm_event_features\household_cohort_profiles.csv`
- Frozen context: `outputs\event_context_corpus\frozen_event_context_prompt.md`
- Output JSONL: `outputs\event_context_corpus\frozen_context_single_prompts.jsonl`
- Cohorts: `1327`
- Max message chars: `6999`
- Rough total input token estimate: `2277536`

## Purpose

This prompt file supports local/open-source LLM replay under the same frozen event-context constraints used by the batched API prompts. It is compatible with `src/run_local_llm_prior_replication.py`.
The JSONL file can be regenerated from committed cohort profiles and frozen context, and may be ignored by git to avoid committing another large prompt artifact.

## Leakage Boundary

- Prompts include frozen ACS/BTS mechanism facts.
- Prompts exclude NHTS 2022 labels, weights, identifiers, and aggregate target outcomes.
- PSRC is explicitly excluded from prompt context and kept as validation-only evidence.
