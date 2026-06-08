# Task Plan: LLM Event Adaptation Experiments

## Goal
Test whether LLM-derived pandemic event semantics can reduce the 2022 overprediction bias of historical household travel-demand models without using 2022 trip-count labels for training or calibration.

## Phase 0: Current Baseline State

Completed:

- Harmonized household dataset: `data/processed/household_harmonized.csv`
- Baseline results: `outputs/household_baseline/household_baseline_results.md`
- Best current model: pooled historical XGBoost with `survey_year`
- Best current weighted bias: `+3.7202`

Interpretation:

- Historical tabular models overpredict 2022 household trip counts.
- This is the target failure mode for LLM-guided adaptation.

## Phase 1: Define LLM Event Feature Schema

Create a strict JSON schema with numeric scores and short rationales.

Status: completed in `src/build_llm_household_profiles.py` and exported to `outputs/llm_event_features/event_feature_schema.json`.

Candidate features:

| Feature | Range | Meaning |
|---|---:|---|
| `trip_suppression_risk` | 0-1 | Probability-like score that the household's travel demand is suppressed by pandemic-era constraints |
| `remote_work_substitution_likelihood` | 0-1 | Likelihood that work-related trips are replaced by remote work |
| `transit_avoidance_likelihood` | 0-1 | Likelihood that transit-related trips decrease due to health-risk concerns |
| `online_delivery_substitution_likelihood` | 0-1 | Likelihood that shopping/service trips are substituted by online delivery |
| `post_pandemic_recovery_sensitivity` | 0-1 | Sensitivity of household travel to recovery-stage normalization |

Rationale fields:

- `primary_event_mechanism`
- `short_explanation`
- `confidence`

Important:

- The LLM must not output `CNTTDHH`.
- The LLM must not see 2022 labels or aggregate 2022 target statistics.
- The LLM should generate interpretable event-response priors only.

## Phase 2: Household Profile Construction

Convert household tabular rows into structured profiles.

Status: completed for 2022 cohort-level profiles.

Profile fields:

- survey year
- census region/division
- urban/rural category
- household income category
- household size
- number of adults
- worker count
- driver count
- vehicle count
- home ownership
- rail availability
- travel month
- travel weekday

Exclude:

- `CNTTDHH`
- `WTHHFIN`
- `HOUSEID`
- any direct target-derived statistic

Implemented leakage controls:

- Prompt JSONL does not contain `CNTTDHH`, `WTHHFIN`, or `HOUSEID`.
- Profile JSON uses a generic leakage note instead of exposing target/weight/ID column names.
- LLM instructions explicitly prohibit predicting or mentioning household trip counts.

## Phase 3: Cohort-Level vs Household-Level Generation

Start with cohort-level generation for cost and consistency.

Status: first cohort-level prompt set generated.

Generated files:

- `outputs/llm_event_features/household_cohort_profiles.csv`
- `outputs/llm_event_features/household_cohort_prompts.jsonl`
- `outputs/llm_event_features/household_cohort_profile_summary.md`
- `outputs/llm_event_features/event_feature_schema.json`

Current prompt set:

- Profile year: 2022
- Source rows: 7,893
- Cohorts: 1,327
- Coverage: 7,893 rows
- Minimum cohort size: 1

Suggested cohort keys:

- `HHFAMINC`
- `HHVEHCNT`
- `WRKCOUNT`
- `URBRUR`
- `RAIL`
- `CENSUS_R`

Workflow:

1. Group households into interpretable cohorts.
2. Generate one LLM event-feature vector per cohort.
3. Join cohort features back to household rows.
4. Compare with individual-level generation on a small sample as an ablation.

Why cohort-level first:

- cheaper
- more stable
- easier to audit
- easier to describe in a paper

## Phase 4: Label-Free Event-Prior Adaptation Design

Use the historical tabular model as the routine mobility predictor.

Main zero-label setting:

1. Train `f_tabular` on 2001, 2009, and 2017 labels.
2. Generate 2022 cohort-level LLM event priors without exposing `CNTTDHH`, sample weights, household IDs, or aggregate target statistics.
3. Compute a fixed event-pressure score from LLM priors.
4. Apply a pre-declared multiplicative correction:

```text
y_hat = f_tabular(X) * clip(1 - alpha * h(z_llm), min_factor, 1)
```

5. Use 2022 `CNTTDHH` only for final evaluation.

Current implemented script:

- `src/run_label_free_llm_adaptation.py`

Current best zero-label sensitivity result:

- `llm_trip_suppression_a1p25`
- Weighted MAE: `2.4820`
- Weighted RMSE: `3.6601`
- Weighted bias: `-0.5404`
- Weighted R2: `0.2244`
- Weighted MAE reduction vs historical baseline: `42.78%`
- Absolute weighted-bias reduction vs historical baseline: `85.01%`

Best bias/R2 tradeoff:

- `gated_trip_suppression_a1_d0p15`
- Weighted MAE: `2.5023`
- Weighted RMSE: `3.6038`
- Weighted bias: `-0.0230`
- Weighted R2: `0.2480`
- The gate uses only the distance between cohort-specific `trip_suppression_risk` and the global mean, so it does not require 2022 labels.

Strict controls:

- `global_trip_suppression_a1p25`: weighted MAE `2.5223`
- `random_trip_suppression_a1p25`: weighted MAE `2.6466 +/- 0.0141`
- `llm_trip_suppression_a1p25` improves weighted MAE by `1.60%` vs global mean and `6.22%` vs random shuffle.
- `gated_trip_suppression_a1_d0p15` gives the strongest bias/R2 tradeoff: weighted MAE `2.5023`, weighted bias `-0.0230`, weighted R2 `0.2480`.
- The gated rule is label-free because it uses only the distance between cohort-specific `trip_suppression_risk` and the global weighted mean, with a fixed threshold.

Interpretation:

- The strongest current signal is the single LLM score `trip_suppression_risk`.
- Broad composite event pressure is less reliable and should not be overclaimed.
- A large share of the gain comes from label-free event-level downscaling; cohort-specific LLM assignment adds a smaller but measurable increment for the trip-suppression score.
- No-label gating can reduce overcorrection and is a better candidate when weighted bias/R2 are prioritized over the single lowest MAE row.
- Gated trip-suppression correction is the strongest option when the objective is near-zero bias and higher weighted R2 rather than absolute minimum MAE.

Subgroup diagnostics:

- Output: `outputs/label_free_llm_adaptation/label_free_subgroup_diagnostics.md`
- LLM ranking beats global downscaling most clearly in selected low-income, household-size, worker-count, vehicle-count, and region groups.
- Global downscaling still beats LLM ranking in some groups, so the next model should be gated rather than uniformly applying cohort-specific LLM scores.

## Phase 4b: Supplementary Residual Adaptation Design

Use the historical tabular model as the routine mobility predictor.

Steps:

1. Train `f_tabular` on historical years.
2. Split 2022 into calibration and test subsets.
3. Compute residuals on 2022 calibration:

```text
residual = CNTTDHH - f_tabular(X)
```

4. Train `g_residual` using:

```text
X_tabular + z_llm -> residual
```

5. Evaluate final prediction on held-out 2022 test:

```text
y_hat_final = f_tabular(X) + g_residual(X, z_llm)
```

Use CUDA XGBoost for residual modeling when possible.

Status:

- Implemented in `src/run_llm_residual_adaptation.py`.
- This uses 2022 calibration labels, so it is an upper-bound/semi-supervised diagnostic rather than the main forecasting-style result.

## Phase 5: Baselines and Ablations

Required comparisons:

| Method | Purpose |
|---|---|
| Historical XGBoost | routine tabular baseline |
| Historical XGBoost + year | best current baseline |
| Historical mean trend shift | label-free historical trend baseline |
| Global event pressure | tests whether only global downscaling matters |
| Random event pressure | negative control for cohort assignment |
| LLM event pressure | proposed label-free method |
| 2022 calibration residual models | supplementary upper-bound diagnostic |

Primary metrics:

- weighted MAE
- weighted RMSE
- weighted bias

Secondary metrics:

- unweighted MAE/RMSE/bias
- subgroup weighted bias
- subgroup weighted MAE
- calibration curve by predicted trip-count bin

## Phase 6: Subgroup Analysis

Analyze performance by:

- `HHFAMINC`
- `HHVEHCNT`
- `WRKCOUNT`
- `URBRUR`
- `RAIL`
- `CENSUS_R`
- `HHSIZE`

Key question:

- Does LLM-guided adaptation mostly help households whose pandemic response is behaviorally plausible but not directly encoded in historical variables?

## Phase 7: Robustness and Leakage Checks

Robustness:

- pre-declared alpha sensitivity grid
- random-pressure seeds
- global-pressure controls
- multiple 2022 calibration/test splits only for supplementary residual experiments
- different calibration fractions: 5%, 10%, 20% only for supplementary residual experiments
- cohort-level vs household-level LLM features
- different LLMs if available

Leakage checks:

- prompts never contain `CNTTDHH`
- prompts never contain 2022 aggregate target statistics
- retrieved context excludes 2022 NHTS outcome summaries if forecasting framing is used
- prompt logs and generated JSON are saved for audit

## Phase 8: Implementation Files

Implemented files:

- `src/build_llm_household_profiles.py`
- `src/generate_llm_event_features.py`
- `src/llm_event_generation.py`
- `src/validate_llm_event_features.py`
- `src/run_label_free_llm_adaptation.py`
- `src/run_llm_residual_adaptation.py`

Generated result files:

- `outputs/label_free_llm_adaptation/label_free_llm_adaptation_results.md`
- `outputs/label_free_llm_adaptation/label_free_llm_adaptation_metrics.csv`
- `outputs/label_free_llm_adaptation/label_free_event_pressure_summary.csv`
- `outputs/llm_residual_adaptation/`

## Expected Paper Claim

If successful:

LLM event priors reduce 2022 weighted bias in a zero-label setting. The current strongest claim is that `trip_suppression_risk` provides an effective event-level correction, with modest cohort-specific signal beyond global and random controls.

If not successful:

The negative result is still useful: it would show that generic LLM semantic priors are insufficient without stronger grounding or target-domain calibration.

## Next Concrete Step

Draft the Results section and method description around the label-free LLM trip-suppression correction, global/random controls, and gated tradeoff.

Current implementation status:

- Cursor local API generation runner implemented.
- Single-cohort connectivity test passed.
- Five-cohort mini pilot passed and validated: 5 valid, 0 invalid.
- Twenty-cohort pilot passed and validated: 20 valid, 0 invalid.
- Full 1,327-cohort LLM event-feature generation completed with `gpt-5.5-low`, `max_concurrency=15`.
- Full validation result: 1,327 valid, 0 invalid.
- Label-free LLM event-prior adaptation implemented and run on CUDA.
- Supplementary 2022 calibration residual adaptation implemented and run on CUDA.
