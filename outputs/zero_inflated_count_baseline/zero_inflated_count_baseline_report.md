# Zero-Inflated Count Baseline Report

## Scope

This experiment adds zero-inflated count-model baselines because household daily trip counts can contain structural zeros. Models train only on 2001, 2009, and 2017 NHTS households and evaluate once on 2022. No 2022 labels are used for training or calibration.

The implementation uses a compact statsmodels design matrix rather than the full one-hot household design, because full zero-inflated maximum-likelihood optimization is slow and numerically fragile on this survey table. Fits are unweighted due to statsmodels discrete zero-inflated model limitations; all reported metrics are still survey-weighted. The default committed run uses zero-inflated Poisson; zero-inflated negative binomial remains available through `--methods zinb` but is not the default because it is much less stable on this design.

## Results

| Method | Weighted MAE | Weighted RMSE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|---:|
| zero_inflated_zip_compact | 4.1742 | 5.2719 | +3.4233 | -0.6091 |

## Diagnostics

| Method | Status | Converged | Iterations | Weighted MAE | Weighted bias |
|---|---|---|---:|---:|---:|
| zip | ok | False | -1 | 4.1742 | +3.4233 |

## Reference Comparisons

| Comparator | Weighted MAE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|
| best zero-inflated count: zero_inflated_zip_compact | 4.1742 | +3.4233 | -0.6091 |
| Poisson GLM | 4.3368 | +3.6178 | -0.7123 |
| Negative-binomial GLM | 6.1229 | -1.1091 | -12.6282 |
| primary gated event adapter | 2.5023 | -0.0230 | 0.2480 |

The primary gated event adapter is `1.6720` weighted-MAE lower than the best zero-inflated count baseline, a relative reduction of `40.05%`.

## Interpretation

The zero-inflated models are a reviewer-facing stress test for count-data adequacy. If they still overpredict 2022 or underperform the event adapter, that supports the paper's claim that the central issue is target-year event shift rather than merely choosing a count-family likelihood.

Do not present this as the final word on all possible travel-demand count models. It is a compact, transparent robustness baseline with explicit solver and weighting caveats.

## Solver Caveat

At least one zero-inflated fit emitted warnings or did not report convergence. The diagnostics CSV preserves this state so the paper can report the baseline honestly.