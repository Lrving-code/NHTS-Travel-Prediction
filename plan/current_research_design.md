# Current Research Design

## One-Sentence Summary

This project studies whether LLM-generated pandemic event-response priors can help historical travel-demand models adapt to the 2022 post-pandemic distribution shift in NHTS household trip counts.

## Problem

Historical household travel models trained on 2001, 2009, and 2017 NHTS data substantially overpredict 2022 household trips.

Current baseline finding:

- Best current historical model: pooled XGBoost with `survey_year`
- Best current 2022 weighted bias: `+3.7202`

Interpretation:

- The model learned routine mobility from pre-pandemic travel behavior.
- The 2022 survey reflects a post-pandemic mobility regime.
- Tabular covariates alone do not explicitly encode pandemic-era mechanisms such as remote work, transit avoidance, online substitution, and recovery sensitivity.

## Core Hypothesis

LLMs should not directly predict household trip counts.

Instead, LLMs are used as semantic event-prior generators. They convert household cohort profiles into structured features that describe plausible pandemic-response mechanisms.

These event priors may help correct the historical model's 2022 overprediction bias without using any 2022 trip-count labels.

The supervised residual-adaptation experiments are retained as a supplementary upper-bound diagnostic: they answer what happens if a small amount of 2022 `CNTTDHH` is available, but they are no longer the main paper setting.

## Method Overview

The main prediction setting is label-free target-year adaptation:

```text
y_hat_final = f_tabular(X) * clip(1 - alpha * h(z_llm), min_factor, 1)
```

Where:

- `f_tabular(X)` is a historical routine-mobility predictor trained on pre-2022 NHTS data.
- `z_llm` is a vector of LLM-generated event-response priors for a 2022 household cohort.
- `h(z_llm)` is a pre-declared event-pressure function, not fitted on 2022 labels.
- `alpha` is reported as a sensitivity grid rather than selected by 2022 target performance.

Supplementary upper-bound setting:

```text
y_hat_final = f_tabular(X) + g_residual(X, z_llm)
```

where `g_residual` is trained on a small 2022 calibration split. This is useful for diagnostics but does not match the stricter forecasting-style problem.

## Data Design

Current harmonized household dataset:

- Years: 2001, 2009, 2017, 2022
- Target: household daily trip count `CNTTDHH`
- 2022 rows: 7,893
- 2022 cohort prompts: 1,327

The LLM operates at cohort level first, not individual-household level.

Rationale:

- Lower cost.
- More stable generation.
- Easier manual audit.
- Easier to describe in a paper.

## LLM Input Design

Each prompt contains:

- Household cohort key.
- Cohort size.
- Safe household summaries.
- Safe categorical distributions.
- Variable descriptions.
- A strict instruction not to predict trip counts.

Excluded from prompts:

- `CNTTDHH`
- sample weights
- household IDs
- aggregate 2022 outcome statistics
- target-derived statistics

## LLM Output Schema

The LLM produces eight fields:

- `trip_suppression_risk`
- `remote_work_substitution_likelihood`
- `transit_avoidance_likelihood`
- `online_delivery_substitution_likelihood`
- `post_pandemic_recovery_sensitivity`
- `primary_event_mechanism`
- `short_explanation`
- `confidence`

All numeric scores must be in `[0, 1]`.

## Provider Design

Current runnable route:

- Provider route: local Cursor API
- Endpoint: `http://127.0.0.1:3008/v1/chat/completions`
- Pilot model: `gpt-5.5-low`
- Candidate full-run model: `gpt-5.5-high`, or `gpt-5.5-medium` if limits require it

Important limitation:

- Local Cursor API does not currently provide verified strict structured-output enforcement.
- Therefore, schema control is implemented with prompt instruction, JSON parsing, local validation, and repair retry.

## Generation Safeguards

The generation runner supports:

- dry-run token/record estimate
- cohort selection
- resumable output by `cohort_id`
- raw response JSONL
- parsed feature JSONL
- error JSONL
- one repair retry for invalid output
- conservative concurrency

Implemented scripts:

- `src/build_llm_household_profiles.py`
- `src/generate_llm_event_features.py`
- `src/llm_event_generation.py`
- `src/validate_llm_event_features.py`
- `src/summarize_llm_event_features.py`

## LLM Generation Status

Completed pilot:

- Model: `gpt-5.5-low`
- Records: 20
- Max concurrency: 1
- Generation errors: 0
- Validator result: 20 valid, 0 invalid

Completed full generation:

- Model: `gpt-5.5-low`
- Provider route: local Cursor API
- Records: 1,327
- Max concurrency: 15
- Generation errors: 0
- Validator result: 1,327 valid, 0 invalid
- Output directory: `outputs/llm_event_features/cursor_api_full_gpt55_low_c15/`

Numeric feature means from the 20-cohort pilot:

- `trip_suppression_risk`: 0.4615
- `remote_work_substitution_likelihood`: 0.3685
- `transit_avoidance_likelihood`: 0.3185
- `online_delivery_substitution_likelihood`: 0.5040
- `post_pandemic_recovery_sensitivity`: 0.4500
- `confidence`: 0.6595

Numeric feature means from the full 1,327-cohort run:

- `trip_suppression_risk`: 0.4559
- `remote_work_substitution_likelihood`: 0.3806
- `transit_avoidance_likelihood`: 0.3147
- `online_delivery_substitution_likelihood`: 0.5058
- `post_pandemic_recovery_sensitivity`: 0.4711
- `confidence`: 0.6586

Initial interpretation:

- Scores vary across cohorts rather than collapsing to constants.
- Zero-vehicle, rail-access, worker count, and urban/rural conditions affect the generated mechanisms.
- The full output is ready to use as cohort-level event features for residual adaptation.

## Label-Free Adaptation Status

Completed script:

- `src/run_label_free_llm_adaptation.py`

Main zero-label constraint:

- Train the historical XGBoost model only on 2001, 2009, and 2017 `CNTTDHH`.
- Generate 2022 LLM cohort priors without exposing `CNTTDHH`, sample weights, IDs, or aggregate 2022 target statistics.
- Use 2022 `CNTTDHH` only inside final evaluation metrics.

Current best zero-label sensitivity result:

- Method: `llm_trip_suppression_a1p25`
- Weighted MAE: `2.4820`
- Weighted RMSE: `3.6601`
- Weighted bias: `-0.5404`
- Weighted R2: `0.2244`
- Weighted MAE reduction vs historical baseline: `42.78%`
- Absolute weighted-bias reduction vs historical baseline: `85.01%`

Best bias/R2 tradeoff:

- Method: `gated_trip_suppression_a1_d0p15`
- Weighted MAE: `2.5023`
- Weighted RMSE: `3.6038`
- Weighted bias: `-0.0230`
- Weighted R2: `0.2480`
- Interpretation: this no-label gated correction gives up a small amount of MAE relative to the best MAE row, but nearly eliminates weighted bias and achieves the highest weighted R2.

Important diagnostic result:

- `trip_suppression_risk` is the strongest individual LLM prior.
- Global-pressure controls are also strong, meaning much of the gain comes from event-level downscaling rather than fine-grained household ranking.
- `llm_trip_suppression_a1p25` improves weighted MAE by `1.60%` over its global-mean control and by `6.22%` over its random-shuffle control.
- The broader recovery-adjusted composite should not be overclaimed because it does not beat global/random controls at the strongest alpha.
- No-label gated correction adds a useful bias/R2 tradeoff: `gated_trip_suppression_a1_d0p15` has weighted MAE `2.5023`, weighted bias `-0.0230`, and weighted R2 `0.2480`.
- This gated result is slightly worse than the best MAE row but is nearly unbiased and has the highest current weighted R2.
- Gated correction is a useful compromise when the paper emphasizes bias correction and distributional fit rather than MAE alone.

Subgroup diagnostic result:

- Output: `outputs/label_free_llm_adaptation/label_free_subgroup_diagnostics.md`
- LLM-specific gains over global trip-suppression control are largest for several behaviorally plausible groups, including low-income households, four-person households, three-worker households, high-vehicle households, Western-region households, zero-worker households, and zero-vehicle households.
- Some groups are better served by a common global downscaling factor, such as six-person households and several income categories.
- Current interpretation: LLM cohort ranking is locally useful but not uniformly better than global event-level correction.

## Evaluation Design

Primary metrics:

- weighted MAE
- weighted RMSE
- weighted bias

Core baselines:

- Historical XGBoost.
- Historical XGBoost with `survey_year`.
- Historical mean-trend shift using only pre-2022 labels.
- Global LLM-pressure controls.
- Random LLM-pressure controls.
- LLM event-prior correction.

Robustness checks:

- Pre-declared alpha sensitivity grid.
- Random-shuffle negative controls.
- Global-mean pressure controls.
- Cohort-level vs sampled household-level LLM features.
- `gpt-5.5-low` vs `gpt-5.5-high` if feasible.
- Local 4090 open-source LLM control if feasible.
- 2022 calibration fractions only as supplementary upper-bound experiments.

## Expected Paper Claim

If successful:

LLM event priors reduce 2022 overprediction bias in a zero-label target-year setting. The strongest current evidence supports label-free event-level downscaling through `trip_suppression_risk`, with modest but nonzero evidence for cohort-specific assignment beyond global and random controls.

If unsuccessful:

The negative result is still informative: it would show that generic LLM semantic priors are insufficient without stronger target-domain grounding or calibration.

## Next Step

Use subgroup diagnostics to design the next refinement: gated label-free correction that defaults to global event-level downscaling and applies LLM cohort ranking only in subgroups where it beats global controls.
