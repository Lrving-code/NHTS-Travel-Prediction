# Residual Adaptation Comparison

| Calibration | Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5% | bias_shift_calibration | 5 | 2.7606 +/- 0.0589 | 3.9012 | 0.1455 +/- 0.3543 | 0.1212 | 0.1439 |
| 5% | residual_tabular_only | 5 | 2.8192 +/- 0.0789 | 3.9717 | 0.3571 +/- 0.2899 | 0.0889 | 0.1047 |
| 5% | residual_tabular_plus_llm | 5 | 2.8379 +/- 0.0833 | 4.0063 | 0.3960 +/- 0.2839 | 0.0727 | 0.0910 |
| 5% | residual_llm_only | 5 | 3.1303 +/- 0.0466 | 4.3657 | 0.5157 +/- 0.2802 | -0.1006 | -0.0552 |
| 5% | historical_xgboost | 5 | 4.3415 +/- 0.0178 | 5.3206 | 3.6092 +/- 0.0187 | -0.6344 | -0.4513 |
| 10% | residual_tabular_only | 5 | 2.7353 +/- 0.0738 | 3.8988 | 0.3047 +/- 0.2282 | 0.1200 | 0.1379 |
| 10% | residual_tabular_plus_llm | 5 | 2.7411 +/- 0.0810 | 3.9061 | 0.3367 +/- 0.2214 | 0.1167 | 0.1367 |
| 10% | bias_shift_calibration | 5 | 2.7508 +/- 0.0317 | 3.8915 | 0.0959 +/- 0.2122 | 0.1242 | 0.1447 |
| 10% | residual_llm_only | 5 | 2.9734 +/- 0.0378 | 4.1648 | 0.3664 +/- 0.1413 | -0.0035 | 0.0372 |
| 10% | historical_xgboost | 5 | 4.3425 +/- 0.0194 | 5.3204 | 3.6087 +/- 0.0237 | -0.6370 | -0.4507 |
| 20% | residual_tabular_only | 5 | 2.6355 +/- 0.0532 | 3.7566 | 0.2354 +/- 0.1162 | 0.1778 | 0.1812 |
| 20% | residual_tabular_plus_llm | 5 | 2.6498 +/- 0.0380 | 3.7774 | 0.2692 +/- 0.1248 | 0.1686 | 0.1751 |
| 20% | bias_shift_calibration | 5 | 2.7391 +/- 0.0249 | 3.8730 | 0.1477 +/- 0.1132 | 0.1261 | 0.1443 |
| 20% | residual_llm_only | 5 | 2.9048 +/- 0.0502 | 4.0804 | 0.2911 +/- 0.0954 | 0.0295 | 0.0622 |
| 20% | historical_xgboost | 5 | 4.3450 +/- 0.0234 | 5.3165 | 3.6231 +/- 0.0241 | -0.6467 | -0.4635 |

## Key Takeaways

- Best 20% calibration method: `residual_tabular_only`.
- 20% calibration weighted MAE reduction vs historical baseline: 39.34%.
- 20% calibration absolute weighted-bias reduction vs historical baseline: 93.50%.
- `residual_tabular_plus_llm` is close to but slightly worse than `residual_tabular_only` in the current XGBoost setup.
- `residual_llm_only` still greatly improves over the historical baseline, but it does not beat simple bias-shift or tabular residual calibration.