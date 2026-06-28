# Cohort-Prior Value Analysis

## Scope

This analysis addresses the reviewer question: if a global post-pandemic downscaling prior is already strong, where do cohort-specific LLM event priors add value?

The main comparison uses the final `primary_gated_adapter` and a same-alpha `global_trip_suppression_a1` control to isolate cohort-specific assignment from correction strength. It also reports the previously used low-cost `global_trip_suppression_a1p25` baseline for consistency with the main result table.

## Overall Metrics

| method | weighted_mae | weighted_rmse | weighted_bias | abs_weighted_bias |
| --- | --- | --- | --- | --- |
| primary_gated_adapter | 2.5023 | 3.6038 | -0.0230 | 0.0230 |
| global_trip_suppression_a1p25 | 2.5223 | 3.7432 | -0.7477 | 0.7477 |
| global_trip_suppression_a1 | 2.5531 | 3.6272 | 0.1229 | 0.1229 |
| cohort_trip_suppression_a1 | 2.5615 | 3.6104 | 0.2874 | 0.2874 |
| historical_xgboost | 4.3377 | 5.3169 | 3.6052 | 3.6052 |

## Summary

| quantity | value | interpretation |
| --- | --- | --- |
| overall_primary_vs_global_a1_mae_delta | 0.0508 | positive means primary gated adapter beats same-alpha global prior |
| overall_primary_vs_global_a1p25_mae_delta | 0.0201 | positive means primary gated adapter beats reported low-cost global baseline |
| overall_primary_vs_global_a1_abs_bias_delta | 0.0998 | positive means primary gated adapter is better calibrated |
| share_subgroup_cells_primary_beats_global_a1 | 0.7778 | unweighted share across evaluated subgroup cells |
| share_subgroup_cells_primary_beats_global_a1p25 | 0.6667 | unweighted share across evaluated subgroup cells |
| median_primary_vs_global_a1_mae_delta | 0.0374 | median subgroup-cell MAE gain over same-alpha global prior |
| gated_uses_cohort_prior_share_rows | 0.1414 | share of households where the gate uses cohort-specific LLM pressure |
| gated_uses_cohort_prior_share_weighted | 0.1749 | survey-weighted share where the gate uses cohort-specific LLM pressure |

Main takeaway:

- The primary gated adapter is `0.0508` weighted-MAE lower than the same-alpha global prior and `0.0201` lower than the reported global-a1.25 prior.
- It beats the same-alpha global prior in `77.8%` of evaluated subgroup cells, while using cohort-specific pressure for `17.5%` of survey-weighted households.

## Subgroups Where Cohort-Specific Priors Help Most

| subgroup_column | subgroup_value | rows | primary_vs_global_a1_mae_delta | primary_vs_global_a1p25_mae_delta | primary_vs_global_a1_abs_bias_delta |
| --- | --- | --- | --- | --- | --- |
| HHFAMINC | 1 | 300 | 0.5515 | 0.1250 | 0.9556 |
| HHVEHCNT | 0 | 476 | 0.3150 | 0.1120 | 0.6401 |
| HHFAMINC | 2 | 242 | 0.2555 | 0.0497 | 0.8435 |
| WRKCOUNT | 0 | 2758 | 0.1170 | -0.0283 | 0.2923 |
| HHSIZE | 5 | 343 | 0.0992 | 0.0257 | -0.1363 |
| HHSIZE | 4 | 786 | 0.0884 | 0.1394 | -0.1297 |
| HHFAMINC | 5 | 824 | 0.0848 | -0.0408 | 0.1702 |
| HHVEHCNT | 1 | 2623 | 0.0702 | -0.0140 | 0.2178 |
| CENSUS_R | 1 | 1363 | 0.0696 | 0.0433 | -0.1963 |
| CENSUS_R | 4 | 1754 | 0.0640 | 0.0904 | -0.1164 |

## Subgroups Where Global Prior Is More Competitive

| subgroup_column | subgroup_value | rows | primary_vs_global_a1_mae_delta | primary_vs_global_a1p25_mae_delta | primary_vs_global_a1_abs_bias_delta |
| --- | --- | --- | --- | --- | --- |
| HHVEHCNT | 5 | 122 | -0.0799 | 0.0737 | 0.1012 |
| HHFAMINC | -7 | 93 | -0.0772 | -0.2425 | -0.0772 |
| WRKCOUNT | 3 | 266 | -0.0495 | 0.1913 | 0.0397 |
| HHVEHCNT | 4 | 360 | -0.0283 | 0.1658 | 0.0768 |
| HHFAMINC | 11 | 806 | -0.0227 | 0.1002 | 0.1338 |
| HHFAMINC | 8 | 890 | -0.0146 | 0.0664 | -0.0160 |
| HHFAMINC | 9 | 611 | -0.0113 | 0.1923 | -0.0003 |
| HHFAMINC | 10 | 593 | -0.0084 | 0.1096 | 0.0173 |
| HHFAMINC | 7 | 1120 | 0.0013 | 0.1446 | -0.0030 |
| WRKCOUNT | 2 | 2026 | 0.0065 | 0.1356 | -0.0287 |

## Interpretation

- The global event prior remains a strong baseline and should stay visible in the paper.
- The final gated adapter is not a claim that cohort-specific ranking explains all gains; it is a calibration-first mechanism that uses cohort-specific priors only when they differ meaningfully from the global event pressure.
- Positive subgroup deltas show where differentiated LLM priors add value beyond event-level downscaling; negative deltas show groups where the common event correction is enough or more stable.
- These subgroup cells overlap across variables, so the shares should be interpreted as diagnostic coverage rather than independent population partitions.