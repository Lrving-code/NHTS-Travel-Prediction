# Top-Venue Adversarial Audit, Round 3

Date: 2026-06-28

Scope: paper package after adding the frozen event-context corpus and reconciling the zero-shot LLM rule-tree ablation.

## Reviewer-Level Verdict

The project is now stronger on provenance and rebuttal readiness than the Round-2 package. The main contribution should still be scoped as **label-free event adaptation for survey-based household mobility**, not as a general mobility foundation model. The strongest current paper story is:

> A historical household model learns routine mobility, while a frozen/auditable LLM event-context branch supplies semantic priors for societal-shock adaptation without using target-year NHTS labels.

## What Improved Since Round 2

| Risk area | New evidence | Current status |
|---|---|---|
| Direct LLM-tree challenge | Zero-shot rule tree now has an explicit trace/spec/report: wMAE `2.7508`, wBias `+0.5000`; pseudo-label tree wMAE `2.7767` | The teacher/reviewer question is answered as an ablation. |
| Retrospective world knowledge | `outputs/event_context_corpus/` freezes ACS/BTS prompt-context facts, PSRC validation-only evidence, candidate sources not used, and 89 batch prompts | Improved from "plan only" to auditable provenance package. |
| Prompt leakage boundary | Frozen context audit reports `0` forbidden target-field hits in allowed fact summaries; frozen-context prompt leakage audit checks 1,327 cohort records and 89 batches with `0` violations | Stronger input-level guardrail. |
| Context immutability | Frozen-context integrity manifest hashes 16 source, prompt, corpus, and audit artifacts with SHA256 | Later prompt/context drift is detectable. |
| Downstream consistency | README, method reports, paper draft, final project report, and audits use the same zero-shot metrics | Lower presentation/rebuttal risk. |

## Remaining Top-Tier Risks

| Risk | Why it still matters | Recommended next fix |
|---|---|---|
| One main shock year | NHTS 2022 remains the primary target shock. PSRC helps but is still COVID/recovery related. | Add a non-COVID regional disruption or another travel-survey shock if data access permits. |
| Frozen context is not full RAG | The corpus freezes source-level facts and prompts, but the committed GPT priors were not regenerated under retrieval-only constraints. | Regenerate priors using the frozen context and compare against the current GPT-reference priors. |
| Open-source LLM equivalence is weak | The 32-cohort local Qwen sensitivity control verifies GPU path but does not match the reference priors well. | Test a stronger local instruction model or calibrate local priors before claiming reproducibility equivalence. |
| Causal language remains guardrail-level | Negative controls and DAGs support plausibility but not identified treatment effects. | Keep "causal guardrails" wording; do not claim causal effects. |
| Purpose composition is exploratory | Purpose-specific LLM prior is not the headline positive result. | Keep purpose as a boundary/extension unless a stronger purpose adapter is added. |

## Safe Claim After Round 3

Use:

> LLM event priors, when constrained by a frozen source context and causal/leakage guardrails, can act as a label-free semantic adapter for historically grounded household travel models under a post-pandemic event shift.

Avoid:

> The LLM independently discovers the true 2022 travel model, or the current system is fully prospective and retrieval-only.

## Next Experiment Gate

The next substantive improvement should be one of:

1. Re-run LLM prior generation using `outputs/event_context_corpus/frozen_event_context_prompt.md` and compare prior distributions plus downstream metrics.
2. Add a stronger local/open-source LLM replication on the same frozen-context prompts.
3. Add an additional external shock or regional replication that is not simply the same national COVID recovery setting.
