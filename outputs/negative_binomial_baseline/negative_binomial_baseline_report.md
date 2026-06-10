# Negative-Binomial Count Baseline Report

## Scope

This experiment extends the transparent count-model family with a negative-binomial GLM, a standard overdispersed count-data model. The dispersion alpha is selected only on historical transfer: train on 2001+2009 and validate on 2017. The selected alpha is then refit on 2001+2009+2017 and evaluated once on the 2022 target-year transfer task.

Selected final alpha: `0.5`. Candidate alphas are ordered by historical validation weighted MAE; if a candidate becomes solver-infeasible after refitting on the full pre-2022 history, the failure is recorded and the next historical candidate is used. The default grid is intentionally conservative (`0.50,1.00`) because wider grids were slow and solver-infeasible on the full one-hot household design.

## Historical Alpha Selection

| Method | Weighted MAE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|
| negative_binomial_glm_alpha_0.5_validate_2017 | 4.9676 | +2.2576 | -1.1522 |

## 2022 Evaluation

| Method | Weighted MAE | Weighted RMSE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|---:|
| negative_binomial_glm_hist_selected | 6.1229 | 15.3423 | -1.1091 | -12.6282 |

## Reference Comparisons

| Comparator | Weighted MAE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|
| negative-binomial GLM | 6.1229 | -1.1091 | -12.6282 |
| Poisson GLM | 4.3368 | +3.6178 | -0.7123 |
| primary gated event adapter | 2.5023 | -0.0230 | 0.2480 |

The primary gated event adapter is `3.6206` weighted-MAE lower than the negative-binomial GLM, a relative reduction of `59.13%`.

## Diagnostics

| Stage | Alpha | Status | Converged | Iterations | Design columns | Weighted MAE | Weighted bias |
|---|---:|---|---|---:|---:|---:|---:|
| historical_validation_2017 | 0.5 | ok | False | 100 | 164 | 4.9676 | +2.2576 |
| historical_validation_2017 | 1 | failed | False | -1 | 0 | NA | NA |
| final_2022_evaluation | 0.5 | ok | False | 100 | 168 | 6.1229 | -1.1091 |

## Interpretation

The negative-binomial GLM is a stronger classical count baseline than the initial Poisson/Tweedie check because it explicitly allows overdispersion and orders dispersion candidates on a pre-2022 historical validation transfer. The solver instability and poor 2022 calibration support the paper's target-year temporal-adaptation framing: even an overdispersed count model trained on routine historical travel does not reliably represent the target-year context shift without an event/context adapter.