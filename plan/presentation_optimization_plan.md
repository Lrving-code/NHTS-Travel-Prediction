# Presentation Optimization Plan

## Goal
Make the project easy to present and review by clarifying method comparisons, metrics, caveats, and repository usage.

## Deliverables
- `outputs/final_project/method_comparison_report.md`
- `outputs/final_project/method_comparison_report_zh.md`
- `outputs/final_project/method_comparison_summary.csv`
- `outputs/final_project/mode_llm_only_comparison.csv`
- `outputs/final_project/presentation_speaker_notes.md`
- `outputs/final_project/presentation_speaker_notes_zh.md`
- `outputs/final_project/NHTS_LLM_Event_Adaptation_Optimized_Presentation.pptx`
- Updated `README.md`

## Scope
- Main story: zero-label 2022 household trip-count adaptation.
- Auxiliary story: household-level mode composition extension.
- Comparison baseline: pure/LLM-only approximation for mode composition, defined as 2017 mean mode composition corrected only by LLM transit-avoidance priors.

## Metric Definitions To Explain
- Weighted MAE/RMSE/Bias/R2 for trip-count regression.
- Exact/within-k household tolerance accuracy for count prediction.
- Weighted total variation for mode composition.
- Weighted mean share MAE for all mode-share components.
- Weighted dominant-mode accuracy for household-level mode composition.
- Transit-share weighted MAE/Bias for the public-transit component.

## Caveats
- Trip-count regression metrics cannot be directly compared with trip-level `TRPTRANS` classification accuracy.
- The mode-composition experiment is auxiliary; the strongest claim remains trip-count adaptation.
- The current LLM-only mode baseline is not a direct LLM output baseline because the LLM was not asked to emit mode shares.

## Status
**Complete** - method comparison artifacts, optimized presentation, speaker notes, and README have been generated.
