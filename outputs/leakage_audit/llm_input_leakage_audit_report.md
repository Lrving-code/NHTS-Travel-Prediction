# LLM Input Leakage Audit

## Scope

This audit checks whether LLM cohort payloads contain target labels, survey weights, household IDs, trip-level labels, or target-derived aggregate outcomes. It treats forbidden terms inside explicit guardrail instructions as allowed and necessary warnings.

## Payload Audit

| Artifact | Records checked | Violations | Examples |
|---|---:|---:|---|
| outputs\llm_event_features\household_cohort_profiles.csv | 1327 | 0 |  |
| outputs\llm_event_features\household_cohort_batch_prompts_b15.jsonl | 1327 | 0 |  |
| outputs\llm_event_features\cursor_api_full_gpt55_low_c15\validated\llm_event_features_normalized.csv | 1 | 0 |  |

## Guardrail Instruction Check

- Required guardrail terms present: `True`
- Missing terms: ``

## Interpretation

A passing audit supports the label-free claim at the input-schema level: the LLM branch receives cohort covariates and event context, not 2022 target labels or evaluation outcomes. This does not eliminate retrospective world-knowledge risk; that remains a separate RAG/frozen-context requirement for a paper-grade version.
