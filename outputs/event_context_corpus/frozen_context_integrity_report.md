# Frozen Event-Context Integrity Manifest

## Summary

- Artifacts tracked: `16`
- Missing artifacts: `0`
- Total bytes hashed: `5720049`
- CSV manifest: `outputs\event_context_corpus\frozen_context_integrity_manifest.csv`
- JSON manifest: `outputs\event_context_corpus\frozen_context_integrity_manifest.json`

## Interpretation

This manifest gives the frozen event-context corpus and prompt package stable file-level SHA256 hashes. It supports provenance review by making later context or prompt edits detectable.

## Artifact Hashes

| Role | Artifact | Exists | Size bytes | Lines | SHA256 prefix |
|---|---|---:|---:|---:|---|
| source_input | `outputs/external_validation/acs_commute_mechanism_validation.csv` | True | 529 | 3 | `b18b93dbe1d4d741` |
| source_input | `outputs/external_validation/bts_annual_mobility_summary.csv` | True | 472 | 5 | `680bdbdac908f0f0` |
| source_input | `outputs/external_validation/psrc_household_year_summary.csv` | True | 530 | 6 | `020c05458f76a80c` |
| source_input | `outputs/external_validation/psrc_household_external_validation_metrics.csv` | True | 1525 | 7 | `b2e867dcedbbb5b8` |
| source_input | `outputs/external_validation/external_dataset_candidates.csv` | True | 1329 | 5 | `c46fc6cbdf9e9a82` |
| prompt_input | `outputs/llm_event_features/household_cohort_profiles.csv` | True | 2708596 | 1328 | `398ed88c2132b728` |
| prompt_input | `outputs/llm_event_features/event_feature_schema.json` | True | 1186 | 52 | `2048b8fd10c88e1b` |
| frozen_corpus | `outputs/event_context_corpus/frozen_event_context_sources.csv` | True | 3606 | 6 | `13bb4f8378d6d6d0` |
| frozen_corpus | `outputs/event_context_corpus/frozen_event_context_facts.json` | True | 4858 | 66 | `fd4ca71d41753eaa` |
| frozen_corpus | `outputs/event_context_corpus/frozen_event_context_prompt.md` | True | 3024 | 46 | `fdb69a7a598de62c` |
| frozen_corpus | `outputs/event_context_corpus/frozen_event_context_audit.md` | True | 1239 | 29 | `94d1e9160dce93e0` |
| frozen_prompt | `outputs/event_context_corpus/frozen_context_batch_prompt_summary.md` | True | 1065 | 22 | `72214ca68832c8f3` |
| frozen_prompt | `outputs/event_context_corpus/frozen_context_batch_prompts.jsonl` | True | 2990347 | 89 | `07c21e6c14bb703f` |
| frozen_prompt_leakage_audit | `outputs/event_context_corpus/leakage_audit/llm_input_leakage_audit.csv` | True | 315 | 4 | `41226f4860f55e92` |
| frozen_prompt_leakage_audit | `outputs/event_context_corpus/leakage_audit/llm_guardrail_instruction_check.csv` | True | 127 | 2 | `0f51d8415cb85ee2` |
| frozen_prompt_leakage_audit | `outputs/event_context_corpus/leakage_audit/llm_input_leakage_audit_report.md` | True | 1301 | 27 | `8f63b015cef2497e` |