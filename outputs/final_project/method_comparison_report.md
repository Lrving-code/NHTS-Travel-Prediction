# Method Comparison Report

## What Is Being Compared

The project has three household-level behavior outputs.

- Household trip generation: daily trip-count regression, target `CNTTDHH`.
- Household mode composition: mode-share prediction derived from trip-level `TRPTRANS`.
- Mode-specific trip counts can be formed as predicted total trips multiplied by predicted mode shares.

## Metric Definitions

- Weighted MAE: survey-weighted average absolute trip-count error. Lower is better.
- Weighted RMSE: survey-weighted root mean squared trip-count error. Lower is better.
- Weighted Bias: survey-weighted signed error. Values closer to zero mean less systematic over/under prediction.
- Weighted R2: weighted explained variation relative to predicting the weighted mean. Higher is better.
- Within-k accuracy: share of households whose trip-count error is within k trips.
- Weighted total variation: `0.5 * sum(abs(predicted mode shares - true mode shares))`, then survey-weighted. Lower is better.
- Transit-share weighted MAE: survey-weighted absolute error for the public-transit share component. Lower is better.

## Trip-Count Results

| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | MAE Gain vs Historical |
|---|---:|---:|---:|---:|---:|
| Historical predictor | 4.3377 | 5.3169 | 3.6052 | -0.6367 | 0.00% |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 | 4.32% |
| LLM-only pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 | 37.35% |
| Zero-shot LLM rule tree | 2.7508 | 3.7834 | 0.5000 | 0.1712 | 36.58% |
| Zero-shot pseudo-label tree | 2.7767 | 3.8197 | 0.5000 | 0.1553 | 35.99% |
| LLM rule + 500-history calibration | 2.7723 | 3.8226 | 0.4627 | 0.1538 | 36.09% |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 | 41.85% |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 | 38.99% |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 42.31% |

Trip-count takeaway:

- Primary strict no-label row: `gated_trip_suppression_a1_d0p15`, weighted MAE `2.5023`, weighted bias `-0.0230`, weighted R2 `0.2480`.
- LLM-only pressure baseline: weighted MAE `2.7175`. This shows that event priors help directionally but need a household historical predictor.
- Zero-shot LLM rule tree reaches weighted MAE `2.7508`, while the pseudo-label tree reaches `2.7767`. Direct LLM-authored trees are useful cold-start baselines, but their numerical calibration remains weaker than the hybrid adapter.
- LLM rule + 500 historical samples reaches weighted MAE `2.7723`. It integrates the rule-distillation plus small-data calibration route, but it reintroduces positive bias under the 2022 event shift.
- Compared with LLM-only pressure, the hybrid gated adapter reduces weighted MAE by `7.92%` and absolute weighted bias by `93.83%`.
- Gated household accuracy: exact `17.8%`, within 2 trips `54.0%`, within 3 trips `71.1%`.

## Stronger Non-LLM Tabular Baselines

| Method | Device | Weighted MAE | Weighted Bias | Weighted R2 |
|---|---|---:|---:|---:|
| catboost_gpu | gpu | 4.2196 | 3.4873 | -0.5712 |
| xgboost_stronger_cuda | cuda | 4.4173 | 3.7364 | -0.6990 |
| xgboost_covariate_shift_reweighted_cuda | cuda | 4.4304 | 3.7523 | -0.7141 |
| lightgbm_cpu | cpu | 4.4430 | 3.7646 | -0.7269 |

The best stronger non-LLM baseline is `catboost_gpu` with weighted MAE `4.2196` and weighted bias `3.4873`. The primary hybrid adapter is `40.70%` lower in weighted MAE, which supports the claim that model capacity and covariate-shift reweighting do not remove the post-pandemic mechanism shift.

## Mode-Composition Results

| Method | Weighted TV | Mean Share MAE | Dominant Accuracy | Transit MAE | Transit Bias |
|---|---:|---:|---:|---:|---:|
| llm_only_2017_mean_no_adapt | 0.2647 | 0.0882 | 0.8933 | 0.0567 | 0.0220 |
| llm_only_2017_mean_transit_a1.25 | 0.2570 | 0.0857 | 0.8933 | 0.0451 | 0.0087 |
| historical_mean_2017 | 0.2647 | 0.0882 | 0.8933 | 0.0567 | 0.0220 |
| historical_xgboost | 0.2008 | 0.0669 | 0.9029 | 0.0325 | 0.0035 |
| global_transit_avoidance_a1 | 0.1989 | 0.0663 | 0.9042 | 0.0289 | -0.0018 |
| llm_transit_avoidance_a1 | 0.1985 | 0.0662 | 0.9046 | 0.0269 | -0.0070 |

Mode-composition takeaway:

- LLM-only correction improves over a 2017 mean prior, but remains weaker than household-feature XGBoost.
- XGBoost + LLM transit prior reduces weighted TV from `0.2008` to `0.1985`.
- XGBoost + LLM transit prior reduces transit-share weighted MAE from `0.0325` to `0.0269`.
- LLM-only corrected transit MAE is `0.0451`, showing that LLM event priors help directionally but need a household-level historical predictor.

## Final Interpretation

The strongest claim is not that LLMs replace mobility models. The stronger and more defensible claim is that LLM event priors repair historical mobility predictors under post-pandemic event shift.