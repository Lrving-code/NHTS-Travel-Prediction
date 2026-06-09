# Household Purpose Composition Extension

## Scope

This extension predicts household-level trip-purpose shares. It adds a third behavior dimension beyond trip generation and mode composition.

## Purpose Mapping

- 2017 `TRIPPURP`: `HBW`, `HBSHOP`, `HBSOCREC`, `HBO`, `NHB`.
- 2022 `TRIPPURP`: `1`, `2`, `3`, `4`, `5`, mapped to the same five broad categories using cross-tabs with `WHYTRP90` and `WHYTRP1S`.

## Weighted Purpose Distribution

| Year | Households | Trips/HH | Work | Shopping | Social/Rec | Other HB | NHB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 117222 | 8.55 | 0.177 | 0.220 | 0.120 | 0.205 | 0.278 |
| 2022 | 6187 | 5.03 | 0.208 | 0.242 | 0.152 | 0.232 | 0.167 |

## Main Metrics

| Method | Weighted TV | Mean share MAE | Dominant purpose acc. | Work MAE | Shopping MAE | Social MAE |
|---|---:|---:|---:|---:|---:|---:|
| historical_mean_2017 | 0.6137 | 0.2455 | 0.1139 | 0.2602 | 0.2724 | 0.2029 |
| historical_xgboost | 0.5661 | 0.2264 | 0.3431 | 0.2024 | 0.2601 | 0.1993 |
| llm_purpose_prior_a0p5 | 0.5719 | 0.2288 | 0.2214 | 0.2009 | 0.2550 | 0.2009 |
| global_purpose_prior_a0p5 | 0.5703 | 0.2281 | 0.2395 | 0.2010 | 0.2543 | 0.2005 |
| llm_purpose_prior_a0p75 | 0.5775 | 0.2310 | 0.1979 | 0.2006 | 0.2524 | 0.2021 |
| global_purpose_prior_a0p75 | 0.5741 | 0.2296 | 0.2060 | 0.2004 | 0.2513 | 0.2013 |
| llm_purpose_prior_a1 | 0.5861 | 0.2344 | 0.1884 | 0.2010 | 0.2501 | 0.2035 |
| global_purpose_prior_a1 | 0.5799 | 0.2320 | 0.1902 | 0.1998 | 0.2486 | 0.2022 |

## Interpretation

- Traditional XGBoost weighted TV: `0.5661`.
- Best reported row: `historical_xgboost`, weighted TV `0.5661`.
- Best row changes weighted TV by `0.00%` vs traditional XGBoost.
- This purpose task is useful for scope expansion even if the LLM prior is not the main source of improvement.

## Figures

- `figures/purpose_distribution_shift.png`
- `figures/purpose_metric_comparison.png`
