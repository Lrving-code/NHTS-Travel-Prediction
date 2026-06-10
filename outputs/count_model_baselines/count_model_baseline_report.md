# Count-Model Baseline Report

## Scope

This experiment adds transparent count-model baselines familiar to transportation and travel-demand reviewers. Models train only on 2001, 2009, and 2017 NHTS household labels and evaluate on 2022. No 2022 `CNTTDHH` labels are used for training, calibration, model selection, or preprocessing beyond final evaluation.

The implemented baselines are linear generalized count-model families over the same harmonized household features used by the ML baselines:

- `poisson_glm_l2`: Poisson GLM with log link and L2 regularization.
- `tweedie_glm_p1p5`: Tweedie GLM with log link and power 1.5, included as an overdispersed count-like baseline.

Regularization alpha: `0.01`; max iterations: `1000`.

## Results

| Method | Weighted MAE | Weighted RMSE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|---:|
| poisson_glm_l2 | 4.3368 | 5.4383 | +3.6178 | -0.7123 |
| tweedie_glm_p1p5 | 4.3761 | 5.5757 | +3.6677 | -0.7999 |

## Solver Diagnostics

| Method | n_iter | convergence warning |
|---|---:|---|
| poisson_glm_l2 | 1000 | yes |
| tweedie_glm_p1p5 | 1000 | yes |

## Reference Comparisons

| Comparator | Weighted MAE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|
| best count model: poisson_glm_l2 | 4.3368 | +3.6178 | -0.7123 |
| strongest non-LLM tabular: catboost_gpu | 4.2196 | +3.4873 | -0.5712 |
| primary gated event adapter | 2.5023 | -0.0230 | 0.2480 |

The primary gated event adapter is `1.8345` weighted-MAE lower than the best count model, a relative reduction of `42.30%`.

## Interpretation

The count-model baselines are intentionally transparent rather than high-capacity. Their persistent positive bias on 2022 supports the paper's main framing: historically fitted routine travel-demand models, including classical count families, do not encode the post-pandemic event mechanisms needed to avoid overpredicting 2022 household travel.

This result should be used as a reviewer-facing baseline, not as a claim that Poisson/Tweedie models are the strongest possible transportation models.

## Solver Caveat

The sklearn L-BFGS solver reached the iteration limit for at least one count model. The results are retained as transparent reviewer-facing baselines, and the diagnostic CSV records the warning. They should not be presented as exhaustively optimized count models.