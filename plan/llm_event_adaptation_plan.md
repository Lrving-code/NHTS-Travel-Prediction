# Task Plan: LLM Event Adaptation Experiments

## Goal
Test whether LLM-derived pandemic event semantics can reduce the 2022 overprediction bias of historical household travel-demand models.

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

## Phase 4: Residual Adaptation Design

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

## Phase 5: Baselines and Ablations

Required comparisons:

| Method | Purpose |
|---|---|
| Historical XGBoost | routine tabular baseline |
| Historical XGBoost + year | best current baseline |
| Target-mean calibration | simple non-LLM correction |
| Covariate reweighting | traditional domain adaptation |
| Hand-coded pandemic features | human-designed semantic baseline |
| Random event features | negative control |
| LLM event features | proposed method |
| LLM event features without rationales | test whether structured reasoning helps |

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

- multiple 2022 calibration/test splits
- different calibration fractions: 5%, 10%, 20%
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

Planned files:

- `src/run_llm_residual_adaptation.py`
- `outputs/llm_event_features/`
- `outputs/llm_residual_adaptation/`

## Expected Paper Claim

If successful:

LLM event features reduce 2022 weighted bias and improve subgroup robustness relative to historical XGBoost, target-mean calibration, and traditional reweighting.

If not successful:

The negative result is still useful: it would show that generic LLM semantic priors are insufficient without stronger grounding or target-domain calibration.

## Next Concrete Step

Implement and test the LLM output validator and normalized feature-table builder before calling an LLM. The request/model strategy is recorded in `plan/llm_request_strategy.md`.

Execution order:

1. Validate controlled sample outputs locally.
2. Add provider-agnostic request runner with dry-run cost estimation.
3. Select 20-50 representative cohorts for pilot.
4. Run low-concurrency pilot only after the validator passes.
5. Audit pilot outputs before any full 1,327-cohort run.

Current implementation status:

- Cursor local API generation runner implemented.
- Single-cohort connectivity test passed.
- Five-cohort mini pilot passed and validated: 5 valid, 0 invalid.
- Twenty-cohort pilot passed and validated: 20 valid, 0 invalid.
- Full 1,327-cohort LLM event-feature generation completed with `gpt-5.5-low`, `max_concurrency=15`.
- Full validation result: 1,327 valid, 0 invalid.
- Next step is joining full cohort features back to household rows and implementing residual adaptation.
