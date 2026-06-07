# LLM Residual Adaptation Results

| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| residual_tabular_only | 5 | 2.7353 +/- 0.0738 | 3.8988 | 0.3047 +/- 0.2282 | 0.1200 | 0.1379 |
| residual_tabular_plus_llm | 5 | 2.7411 +/- 0.0810 | 3.9061 | 0.3367 +/- 0.2214 | 0.1167 | 0.1367 |
| bias_shift_calibration | 5 | 2.7508 +/- 0.0317 | 3.8915 | 0.0959 +/- 0.2122 | 0.1242 | 0.1447 |
| residual_llm_only | 5 | 2.9734 +/- 0.0378 | 4.1648 | 0.3664 +/- 0.1413 | -0.0035 | 0.0372 |
| historical_xgboost | 5 | 4.3425 +/- 0.0194 | 5.3204 | 3.6087 +/- 0.0237 | -0.6370 | -0.4507 |

Notes:

- Historical model is trained on 2001, 2009, and 2017 households.
- 2022 is split into calibration/test subsets for each seed.
- Residual target is `CNTTDHH - historical_prediction`.
- All XGBoost models use CUDA.
- Lower MAE/RMSE and bias closer to zero are better.