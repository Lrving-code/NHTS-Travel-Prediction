# LLM Input Leakage Audit

## Scope

This audit checks whether LLM cohort payloads contain target labels, survey weights, household IDs, trip-level labels, or target-derived aggregate outcomes. It treats forbidden terms inside explicit guardrail instructions as allowed and necessary warnings.

## Payload Audit

| Artifact | Records checked | Violations | Examples |
|---|---:|---:|---|
| outputs\llm_event_features\household_cohort_profiles.csv | 1327 | 0 |  |
| outputs\event_context_corpus\frozen_context_batch_prompts.jsonl | 1327 | 0 |  |
| outputs\llm_event_features\cursor_api_full_gpt55_low_c15\validated\llm_event_features_normalized.csv | 1 | 0 |  |

## Guardrail Instruction Check

- Required guardrail terms present: `True`
- Missing terms: ``

## Interpretation

A passing audit supports the label-free claim at the input-schema level: the LLM branch receives cohort covariates and event context, not 2022 target labels or evaluation outcomes.

## Frozen Context Provenance

- Frozen context audit: `outputs\event_context_corpus\frozen_event_context_audit.md`
- This does not prove that all committed priors were regenerated under retrieval-only constraints, but it verifies that the frozen-context prompt package can be audited separately from target labels.
