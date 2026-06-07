# LLM Residual Adaptation Results

| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| bias_shift_calibration | 5 | 2.7606 +/- 0.0589 | 3.9012 | 0.1455 +/- 0.3543 | 0.1212 | 0.1439 |
| residual_tabular_only | 5 | 2.8192 +/- 0.0789 | 3.9717 | 0.3571 +/- 0.2899 | 0.0889 | 0.1047 |
| residual_tabular_plus_llm | 5 | 2.8379 +/- 0.0833 | 4.0063 | 0.3960 +/- 0.2839 | 0.0727 | 0.0910 |
| residual_llm_only | 5 | 3.1303 +/- 0.0466 | 4.3657 | 0.5157 +/- 0.2802 | -0.1006 | -0.0552 |
| historical_xgboost | 5 | 4.3415 +/- 0.0178 | 5.3206 | 3.6092 +/- 0.0187 | -0.6344 | -0.4513 |

Notes:

- Historical model is trained on 2001, 2009, and 2017 households.
- 2022 is split into calibration/test subsets for each seed.
- Residual target is `CNTTDHH - historical_prediction`.
- All XGBoost models use CUDA.
- Lower MAE/RMSE and bias closer to zero are better.