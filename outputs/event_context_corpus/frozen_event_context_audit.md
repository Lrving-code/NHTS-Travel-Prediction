# Frozen Event-Context Corpus Audit

Freeze date: `2026-06-28`

## Summary

- Frozen allowed prompt-context facts: `2`
- Frozen guardrail context facts: `1`
- External validation-only facts: `1`
- Candidate sources not used in current priors: `1`
- Prospective event context file exists: `True`
- Forbidden target-field hits in allowed fact summaries: `0`

## Status Counts

- `candidate_not_used`: `1`
- `external_validation_only_not_prompt_context`: `1`
- `frozen_allowed_context`: `2`
- `frozen_guardrail_context`: `1`

## Artifacts

- Source manifest: `outputs\event_context_corpus\frozen_event_context_sources.csv`
- JSON facts: `outputs\event_context_corpus\frozen_event_context_facts.json`
- Prompt context: `outputs\event_context_corpus\frozen_event_context_prompt.md`

## Interpretation

This corpus turns the earlier prospective-context plan into an auditable source package. The LLM can be instructed to use the frozen ACS and BTS facts for mechanism direction and compatibility guardrails, while PSRC remains external validation-only. The corpus still does not prove full prospective deployment, but it materially reduces the risk that the priors are an unconstrained memory-based prompt artifact.
