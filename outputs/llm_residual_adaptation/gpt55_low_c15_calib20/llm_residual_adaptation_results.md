# LLM Residual Adaptation Results

| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| residual_tabular_only | 5 | 2.6355 +/- 0.0532 | 3.7566 | 0.2354 +/- 0.1162 | 0.1778 | 0.1812 |
| residual_tabular_plus_llm | 5 | 2.6498 +/- 0.0380 | 3.7774 | 0.2692 +/- 0.1248 | 0.1686 | 0.1751 |
| bias_shift_calibration | 5 | 2.7391 +/- 0.0249 | 3.8730 | 0.1477 +/- 0.1132 | 0.1261 | 0.1443 |
| residual_llm_only | 5 | 2.9048 +/- 0.0502 | 4.0804 | 0.2911 +/- 0.0954 | 0.0295 | 0.0622 |
| historical_xgboost | 5 | 4.3450 +/- 0.0234 | 5.3165 | 3.6231 +/- 0.0241 | -0.6467 | -0.4635 |

Notes:

- Historical model is trained on 2001, 2009, and 2017 households.
- 2022 is split into calibration/test subsets for each seed.
- Residual target is `CNTTDHH - historical_prediction`.
- All XGBoost models use CUDA.
- Lower MAE/RMSE and bias closer to zero are better.