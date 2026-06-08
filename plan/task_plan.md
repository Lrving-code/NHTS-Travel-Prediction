# Task Plan: Label-Free LLM Event Adaptation

## Goal
Turn the project into a reproducible zero-label 2022 adaptation study where 2022 `CNTTDHH` is used only for final evaluation.

## Phases
- [x] Phase 1: Initialize Git repository and create an initial commit.
- [x] Phase 2: Reframe the main study away from 2022 calibration labels.
- [x] Phase 3: Implement label-free LLM event-prior correction.
- [x] Phase 4: Run CUDA experiments and negative controls.
- [x] Phase 5: Update paper logic and result summaries.
- [x] Phase 6: Commit the new scripts and documents.

## Key Questions
1. Can LLM event priors reduce 2022 overprediction without seeing any 2022 trip-count labels?
2. Does the improvement exceed random event-prior controls?
3. Is the effect robust across fixed, pre-declared correction strengths?

## Decisions Made
- Main setting: train only on 2001, 2009, and 2017 labels; evaluate on 2022.
- 2022 calibration-label experiments are supplementary/upper-bound diagnostics, not the main claim.
- LLM priors are applied through fixed event-pressure correction rules and sensitivity grids.
- `trip_suppression_risk` is the strongest current LLM feature.
- No-label gated correction gives the best bias/R2 tradeoff.
- All XGBoost training must use CUDA.

## Current Results
- Best zero-label MAE row: `llm_trip_suppression_a1p25`, weighted MAE `2.4820`, weighted bias `-0.5404`.
- Best bias/R2 tradeoff: `gated_trip_suppression_a1_d0p15`, weighted MAE `2.5023`, weighted bias `-0.0230`, weighted R2 `0.2480`.
- Historical baseline: weighted MAE `4.3377`, weighted bias `+3.6052`.
- `llm_trip_suppression_a1p25` improves weighted MAE by `1.60%` over global trip suppression and `6.22%` over random trip suppression.

## Errors Encountered
- `git status` was briefly run in parallel before `git init` completed; rerunning after initialization resolved it.
- After the gated-summary patch, shell commands briefly timed out even for trivial commands. The backend recovered; process checks showed no residual training job, and final output files were complete.

## Status
**Completed** - label-free adaptation script, results, and updated planning documents are committed.
