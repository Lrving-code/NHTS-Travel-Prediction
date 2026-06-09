# Final Project Report

## Core Claim

This project formulates 2022 NHTS household travel prediction as event-driven temporal adaptation. A traditional supervised baseline learns routine mobility from pre-2022 NHTS data, but the post-pandemic wave contains event mechanisms such as remote work substitution, transit avoidance, online delivery substitution, and uneven recovery. The main contribution is a label-free hybrid adapter: LLM-derived event priors correct the routine predictor, while 2022 trip-count labels are used only for final evaluation.

## Research Questions

- RQ1: How severely do historical household travel models overpredict 2022 post-pandemic trip generation?
- RQ2: Can event priors reduce this bias without using 2022 `CNTTDHH` labels for training or calibration?
- RQ3: Does the LLM replace historical prediction, or is the stronger design a hybrid of routine mobility and event semantics?

## Main Result

| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 |
|---|---:|---:|---:|---:|
| Historical mean | 5.5116 | 6.1251 | 4.4995 | -1.1722 |
| Traditional supervised baseline | 4.3377 | 5.3169 | 3.6052 | -0.6367 |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 |
| LLM-only pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 |
| LLM trip suppression | 2.4820 | 3.6601 | -0.5404 | 0.2244 |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 |

## Temporal Transfer Validation

Before interpreting 2022 as an event-shift target, we checked routine cross-year transfer:

| Check | Weighted MAE | Weighted Bias | R2 |
|---|---:|---:|---:|
| 2001 -> 2009 | 3.5207 | +0.5818 | 0.4413 |
| 2001+2009 -> 2017 | 4.1272 | +1.2230 | 0.2505 |
| 2001+2009+2017 -> 2022 | 4.4062 | +3.7202 | -0.5055 |

The pre-COVID checks have mean absolute weighted bias `0.9024`, while 2022 has absolute weighted bias `3.7202`. This supports the problem framing: 2022 is a stronger post-pandemic event shift rather than an ordinary transfer year.

## Improvement Over Historical Predictor

- Primary strict no-label row: `gated_trip_suppression_a1_d0p15`.
- Primary weighted MAE drops from `4.3377` to `2.5023` (42.31% reduction).
- Primary weighted RMSE drops from `5.3169` to `3.6038` (32.22% reduction).
- Primary absolute weighted bias drops by `99.36%`.
- Best-MAE sensitivity row: `llm_trip_suppression_a1p25` reaches weighted MAE `2.4820` (42.78% reduction).
- LLM-only pressure improves over naive historical baselines but remains weaker than the hybrid adapter, supporting the design choice that LLMs provide event semantics rather than standalone household predictions.

## Household-Level Accuracy

- Gated MAE: `2.4738` trips per household.
- Gated RMSE: `3.5872` trips per household.
- Exact rounded hit rate: `17.7%`.
- Within 1 trip: `29.8%`.
- Within 2 trips: `54.5%`.
- Within 3 trips: `71.2%`.

## Statistical Validation

Household bootstrap resampling gives the following 95% confidence intervals:

- Traditional supervised baseline weighted MAE: `[4.2418, 4.4312]`.
- Gated LLM correction weighted MAE: `[2.4292, 2.5808]`.
- Paired weighted-MAE reduction: `[1.7422, 1.9267]`.
- Paired absolute-bias reduction: `[3.3582, 3.6696]`.

This supports the claim that the main trip-count improvement is not a single point-estimate artifact.

## Behavior-System Extension

The project now reports household travel behavior as a multi-output system:

1. Trip generation: household `CNTTDHH`.
2. Mode composition: household mode-share vector from `TRPTRANS`.
3. Mode-specific trip volume: predicted total trips multiplied by predicted mode shares.
4. Purpose composition: household purpose-share vector from `TRIPPURP`.

Mode-specific trip-volume results:

- Traditional count × traditional mode total mode-trip MAE: `4.8467`.
- Gated count × LLM mode total mode-trip MAE: `3.2802`.

Purpose-composition results:

- Traditional XGBoost purpose weighted TV: `0.5661`.
- LLM purpose prior is not the overall best row, but it adds a third behavior dimension and exposes a clear future-work target for purpose-specific event adaptation.

## Efficiency

- Household rows in 2022: `7,893`.
- LLM cohort prompts: `1,327`.
- LLM request reduction: `83.2%`, or about `5.95x` fewer requests than household-level prompting.
- Batch prompting with batch size 15 further compresses `1,327` cohort prompts into `89` batch prompts, a `93.29%` request reduction relative to one-cohort prompts.

## LLM Generalization Role

The LLM should be framed as an event-generalization module, not as a direct predictor. It maps pandemic mechanisms such as remote work, transit avoidance, online delivery substitution, and uneven recovery onto unlabeled household cohorts. A prospective event-context file is included at `plan/prospective_event_context_2022.md` to make this role more auditable and reduce retrospective leakage risk.

## Robustness Check

We ran additional robustness checks in `outputs/robustness_checks/`.

- 500-run permutation control for `llm_trip_suppression_a1p25`: actual weighted MAE `2.4820`, random-permutation mean `2.6425`, empirical p-value `0.0020`.
- 500-run permutation control for the primary gated rule: actual weighted MAE `2.5023`, random-permutation mean `2.5808`, empirical p-value `0.0020`.
- Same-alpha global pressure remains strong: primary gated weighted MAE `2.5023` vs global-a1 weighted MAE `2.5531`.
- Leakage scan passes for LLM-facing profile/feature files: they exclude `HOUSEID`, `CNTTDHH`, and `WTHHFIN`.

Interpretation for the course report: the dominant contribution is event-level label-free adaptation. Cohort-specific LLM ranking provides measurable incremental signal, but it should not be described as the sole source of improvement.

## Figures

- `metric_comparison`: `figures/metric_comparison.png`
- `bias_r2`: `figures/bias_r2_tradeoff.png`
- `household_accuracy`: `figures/household_tolerance_accuracy.png`
- `pressure_distribution`: `figures/event_pressure_distribution.png`
- `subgroup_gains`: `figures/subgroup_llm_gains.png`
- `workflow`: `figures/method_workflow.png`

## Presentation Framing

Avoid framing the project as a generic feature-only forecasting improvement. The cleaner narrative is: a traditional supervised baseline fails under a rare event; LLMs provide event semantics that can be distilled into a lightweight correction rule.

Do not overstate the mode-composition or purpose-composition extensions. The strongest result remains trip generation under event-driven temporal adaptation; the extensions show a broader behavior system and planning relevance.
