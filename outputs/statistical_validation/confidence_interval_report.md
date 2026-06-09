# Statistical Validation Report

## Scope

This report uses household bootstrap resampling to estimate uncertainty for the final 2022 trip-count metrics. It keeps survey weights inside each bootstrap sample.

## Key Confidence Intervals

| Method | Metric | Point | 95% CI |
|---|---|---:|---:|
| gated_llm_correction | weighted_mae | 2.5023 | [2.4292, 2.5808] |
| gated_llm_correction | weighted_bias | -0.0230 | [-0.1304, 0.0878] |
| gated_llm_correction | weighted_within_2_trips | 0.5395 | [0.5256, 0.5531] |
| traditional_supervised_baseline | weighted_mae | 4.3377 | [4.2418, 4.4312] |
| traditional_supervised_baseline | weighted_bias | 3.6052 | [3.4867, 3.7253] |
| traditional_supervised_baseline | weighted_within_2_trips | 0.2339 | [0.2220, 0.2462] |

## Paired Improvement vs Traditional Baseline

| Metric | Mean | 95% CI |
|---|---:|---:|
| weighted_mae_reduction | 1.8355 | [1.7422, 1.9267] |
| weighted_rmse_reduction | 1.7134 | [1.5664, 1.8644] |
| absolute_bias_reduction | 3.5591 | [3.3582, 3.6696] |
| within2_gain | 0.3056 | [0.2863, 0.3246] |

## Interpretation

The main result should be presented with uncertainty rather than a single point estimate. If the paired MAE-reduction interval stays positive, it supports the claim that the hybrid event adapter robustly improves household trip-count prediction.
