# Task Plan: Household Baseline Modeling

## Goal
Build the first reproducible household-level baseline for predicting 2022 household travel-day trip counts from historical NHTS waves.

## Phases
- [x] Phase 1: Inspect household columns and values
- [x] Phase 2: Build harmonized household modeling dataset
- [x] Phase 3: Train and evaluate 2017 -> 2022 baseline
- [x] Phase 4: Train and evaluate 2001+2009+2017 -> 2022 baseline
- [x] Phase 5: Summarize baseline findings and next domain adaptation steps

## Key Questions
1. Which common household variables are usable as predictors without leakage?
2. How large is the direct-transfer error from 2017 to 2022?
3. Does multi-year training improve robustness relative to 2017-only training?

## Decisions Made
- Primary target: `CNTTDHH`.
- Sample weight: `WTHHFIN`.
- ID variable `HOUSEID` is excluded from features.
- Raw `TDAYDATE` should be converted into a month feature instead of used as a raw date string.
- Start with tree-based baselines before adding domain adaptation or LLM-assisted residual correction.
- Use XGBoost CUDA training on the available RTX 4090; do not silently fall back to CPU.

## Errors Encountered
- None yet.

## Baseline Findings
- Harmonized household dataset rows: 357,553.
- 2022 household mean `CNTTDHH` is 3.937, much lower than 2017 mean 7.121.
- Direct 2017 -> 2022 XGBoost baseline overpredicts 2022 household trips.
- Weighted MAE/RMSE/Bias:
  - `direct_2017_to_2022`: 4.5921 / 5.6292 / +3.9598
  - `pooled_history_to_2022`: 4.7163 / 5.8269 / +4.1255
  - `pooled_history_with_year_to_2022`: 4.4062 / 5.4076 / +3.7202
- Multi-year training with `survey_year` is currently best among the three baselines, but all baselines still show strong positive bias.

## Next Domain Adaptation Steps
1. Add simple target-mean calibration using a small 2022 validation slice.
2. Add covariate shift sample reweighting from historical years to 2022.
3. Run subgroup error analysis by income, vehicle count, urban/rural status, and census region.

## Status
**Completed** - First household baselines ran successfully on CUDA; next work should focus on calibration and domain adaptation.
