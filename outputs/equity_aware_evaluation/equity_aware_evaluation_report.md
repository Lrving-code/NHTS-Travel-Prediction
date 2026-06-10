# Equity-Aware Evaluation Report

## Scope

This report evaluates whether event-adapted household trip predictions improve not only the average household error but also subgroup robustness across worker count, vehicle count, urban/rural status, rail access, family income, census region, and household size.

## Method-Level Equity Summary

| Method | wMAE | |bias| | Worst subgroup MAE | Worst gap | Share groups improved |
|---|---:|---:|---:|---:|---:|
| llm_trip_suppression_a1p25 | 2.4820 | 0.5404 | 4.6218 | 2.1399 | 100.00% |
| gated_trip_suppression_a1_d0p15 | 2.5023 | 0.0230 | 4.7091 | 2.2068 | 100.00% |
| global_trip_suppression_a1p25 | 2.5223 | 0.7477 | 4.7348 | 2.2124 | 100.00% |
| llm_only_trip_suppression_a1p25 | 2.7175 | 0.3732 | 5.0595 | 2.3420 | 100.00% |
| historical_mean_only | 5.5116 | 4.4995 | 7.1077 | 1.5961 | 25.00% |
| historical_xgboost | 4.3377 | 3.6052 | 8.3778 | 4.0401 | 0.00% |

## Equity-Aware Operating Points

| Profile | Selected method | Score | wMAE | |bias| | Worst subgroup MAE | Share improved |
|---|---|---:|---:|---:|---:|---:|
| equity_first | gated_trip_suppression_a1_d0p15 | 0.0136 | 2.5023 | 0.0230 | 4.7091 | 100.00% |
| balanced_equity | gated_trip_suppression_a1_d0p15 | 0.0100 | 2.5023 | 0.0230 | 4.7091 | 100.00% |
| accuracy_first_core_methods | gated_trip_suppression_a1_d0p15 | 0.0085 | 2.5023 | 0.0230 | 4.7091 | 100.00% |

## Main Interpretation

The primary gated event adapter reduces overall weighted MAE from `4.3377` to `2.5023` and worst-subgroup weighted MAE from `8.3778` to `4.7091`.

This supports the mobility-inequality framing: the event adapter improves average performance and reduces the worst subgroup error among the evaluated core methods. It remains a predictive subgroup robustness result, not a causal equity claim.
