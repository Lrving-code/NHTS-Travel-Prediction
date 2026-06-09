# Final Project Report

## Core Claim

This project predicts 2022 household daily trip counts under post-pandemic distribution shift. The main contribution is a label-free LLM event-prior correction: 2022 trip-count labels are used only for final evaluation.

## Main Result

| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 |
|---|---:|---:|---:|---:|
| Historical predictor | 4.3377 | 5.3169 | 3.6052 | -0.6367 |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 |
| LLM trip suppression | 2.4820 | 3.6601 | -0.5404 | 0.2244 |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 |

## Improvement Over Historical Predictor

- Primary strict no-label row: `gated_trip_suppression_a1_d0p15`.
- Primary weighted MAE drops from `4.3377` to `2.5023` (42.31% reduction).
- Primary weighted RMSE drops from `5.3169` to `3.6038` (32.22% reduction).
- Primary absolute weighted bias drops by `99.36%`.
- Best-MAE sensitivity row: `llm_trip_suppression_a1p25` reaches weighted MAE `2.4820` (42.78% reduction).

## Household-Level Accuracy

- Gated MAE: `2.4738` trips per household.
- Gated RMSE: `3.5872` trips per household.
- Exact rounded hit rate: `17.7%`.
- Within 1 trip: `29.8%`.
- Within 2 trips: `54.5%`.
- Within 3 trips: `71.2%`.

## Efficiency

- Household rows in 2022: `7,893`.
- LLM cohort prompts: `1,327`.
- LLM request reduction: `83.2%`, or about `5.95x` fewer requests than household-level prompting.

## Figures

- `metric_comparison`: `figures/metric_comparison.png`
- `bias_r2`: `figures/bias_r2_tradeoff.png`
- `household_accuracy`: `figures/household_tolerance_accuracy.png`
- `pressure_distribution`: `figures/event_pressure_distribution.png`
- `subgroup_gains`: `figures/subgroup_llm_gains.png`
- `workflow`: `figures/method_workflow.png`

## Presentation Framing

Avoid framing the project as a generic feature-only forecasting improvement. The cleaner narrative is: historical routine-mobility prediction fails under a rare event; LLMs provide event semantics that can be distilled into a lightweight correction rule.