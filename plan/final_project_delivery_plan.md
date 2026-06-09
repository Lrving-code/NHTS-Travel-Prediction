# Final Project Delivery Plan

## Goal
Finish the project as a reproducible large-coursework deliverable with full experiments, paper-ready figures, and a polished presentation deck.

## Scope
- Main experiment: zero-label 2022 household trip-count adaptation.
- Target: household daily trip count `CNTTDHH`.
- Main method: LLM-generated pandemic event priors, especially `trip_suppression_risk`.
- Main constraint: 2022 `CNTTDHH` is used only for final evaluation.

## Deliverables
- `outputs/final_project/final_project_report.md`
- `outputs/final_project/final_metrics_summary.csv`
- `outputs/final_project/household_accuracy_summary.csv`
- `outputs/final_project/figures/`
- `outputs/final_project/NHTS_LLM_Event_Adaptation_Presentation.pptx`
- `plan/ppt_storyboard.md`

## Phases
- [x] Phase 1: Validate current zero-label experiment outputs.
- [x] Phase 2: Generate final metrics and household-level accuracy.
- [x] Phase 3: Generate paper/PPT figures.
- [x] Phase 4: Generate PPT storyboard and `.pptx` draft.
- [x] Phase 5: Verify outputs and commit.

## Main Claims To Support
- Label-free LLM event-prior correction reduces household-level 2022 prediction error under post-pandemic distribution shift.
- Best MAE row: `llm_trip_suppression_a1p25`.
- Best bias/R2 tradeoff row: `gated_trip_suppression_a1_d0p15`.
- Cohort-level LLM generation cuts requests from 7,893 households to 1,327 cohorts, an 83.2% reduction.

## Status
**Complete** - final outputs are generated, verified, and ready for presentation review.
