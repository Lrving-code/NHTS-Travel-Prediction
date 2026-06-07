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

These event priors may help a supervised residual adapter correct the historical model's 2022 overprediction bias.

## Method Overview

The final prediction has two stages:

```text
y_hat_final = f_tabular(X) + g_residual(X, z_llm)
```

Where:

- `f_tabular(X)` is a historical routine-mobility predictor trained on pre-2022 NHTS data.
- `z_llm` is a vector of LLM-generated event-response priors for a 2022 household cohort.
- `g_residual(X, z_llm)` is a supervised residual adapter trained on a small 2022 calibration split.

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

## Evaluation Design

Primary metrics:

- weighted MAE
- weighted RMSE
- weighted bias

Core baselines:

- Historical XGBoost.
- Historical XGBoost with `survey_year`.
- Target-mean calibration.
- Covariate reweighting.
- Hand-coded pandemic features.
- Random event features.
- LLM event features.

Robustness checks:

- Different 2022 calibration fractions.
- Multiple calibration/test splits.
- Cohort-level vs sampled household-level LLM features.
- `gpt-5.5-low` vs `gpt-5.5-high` if feasible.
- Local 4090 open-source LLM control if feasible.

## Expected Paper Claim

If successful:

LLM event priors reduce 2022 overprediction bias and improve subgroup robustness relative to historical tabular baselines and traditional adaptation controls.

If unsuccessful:

The negative result is still informative: it would show that generic LLM semantic priors are insufficient without stronger target-domain grounding or calibration.

## Next Step

Join the validated full-run cohort features back to 2022 household rows, then run residual-adaptation experiments.
