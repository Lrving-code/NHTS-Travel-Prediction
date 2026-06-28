# Top-Venue Adversarial Audit, Round 4

Date: 2026-06-29

Scope: package after adding survey-weighted trip-level mode-choice transfer and frozen-context prior replay channels.

## Reviewer-Level Verdict

The project is now cleaner on survey-weighted evaluation and prompt provenance. The strongest defensible story remains:

> A historically grounded household mobility model is adapted to a post-pandemic event shift using frozen, auditable LLM event priors and causal/leakage guardrails, then evaluated across trip counts, mode behavior, purpose behavior, equity, uncertainty, and planning trade-offs.

The paper should still avoid claiming a general mobility foundation model or a fully completed retrieval-only prior regeneration.

## What Improved Since Round 3

| Risk area | New evidence | Current status |
|---|---|---|
| Mode-choice weighting | Trip-level `WTTRDFIN` is carried into `SAMPLE_WEIGHT`; selected mode-choice model is `xgboost_survey_weighted` | Stronger survey-methods alignment than the previous unweighted mode-choice table. |
| Mode-choice reporting | 2022 weighted accuracy `0.9297`, weighted balanced accuracy `0.4343`, weighted macro F1 `0.4663`; class-balanced rare-mode trade-off is explicit | Safer than reporting ordinary accuracy alone. |
| Frozen-context API replay | `src/generate_batched_llm_event_features.py` can parse frozen batch prompts, validate multi-record JSON outputs, and compare against reference priors | Code gap closed; execution currently blocked by missing `CURSOR_API_AUTH_TOKEN`. |
| Frozen-context local replay | `src/build_frozen_context_single_prompts.py` writes 1,327 single-cohort prompts compatible with local LLM replay | Prompt-path gap closed; execution currently blocked by missing local HuggingFace model cache or download permission. |
| Failure provenance | Both replay paths now write explicit blocker artifacts under `outputs/llm_event_features/frozen_context_cursor_api_pilot/` and `outputs/local_llm_prior_replication_frozen_context/` | The remaining issue is external credential/model availability, not an undocumented method gap. |

## Remaining Top-Tier Risks

| Risk | Why it still matters | Required next fix |
|---|---|---|
| Frozen-context priors are not regenerated yet | A reviewer may ask whether the main GPT priors would change under the frozen source context. | Provide `CURSOR_API_AUTH_TOKEN` for the local Cursor endpoint or run an equivalent provider, then execute at least a pilot and preferably all 89 frozen-context batches. |
| Local frozen-context replay did not load the model | The environment has CUDA and transformers, but `Qwen/Qwen2.5-1.5B-Instruct` is not available in the local HuggingFace cache and offline loading failed. | Either cache/download the model with `--allow-download`, provide a local model path, or select a cached instruction model. |
| External shock diversity remains limited | NHTS 2022 and PSRC 2023 are still post-pandemic/recovery settings. | Add a non-COVID regional disruption only if a compatible survey or panel becomes available. |
| Causal language remains guardrail-level | DAGs and negative controls do not identify causal treatment effects. | Keep "causal guardrails" wording and avoid treatment-effect claims. |
| Purpose composition remains exploratory | Purpose-specific LLM prior is not the primary positive result. | Keep purpose as a behavior-system extension unless a stronger purpose adapter is added. |

## Safe Claim After Round 4

Use:

> The current artifact package implements frozen-context replay paths for both batched API priors and local open-source priors, and documents the exact external credential/model-cache blockers that prevent completion of retrieval-only prior regeneration in the current environment.

Avoid:

> The main GPT priors have already been fully regenerated under frozen-context retrieval constraints.

## Next Experiment Gate

The next substantive gate is now concrete and external-state dependent:

1. Set `CURSOR_API_AUTH_TOKEN` and run:
   `python src/generate_batched_llm_event_features.py --input-jsonl outputs/event_context_corpus/frozen_context_batch_prompts.jsonl --output-dir outputs/llm_event_features/frozen_context_cursor_api_full --max-concurrency 1 --max-tokens 6000`
2. Or cache/download a local instruction model and run:
   `python src/run_local_llm_prior_replication.py --input-jsonl outputs/event_context_corpus/frozen_context_single_prompts.jsonl --output-dir outputs/local_llm_prior_replication_frozen_context --limit 32 --selection even --max-input-tokens 8192`
3. Compare regenerated priors against the current GPT-reference priors before changing the paper’s main quantitative claim.
