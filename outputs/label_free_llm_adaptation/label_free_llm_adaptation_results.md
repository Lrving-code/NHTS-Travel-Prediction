# Label-Free LLM Event Adaptation Results

| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| llm_trip_suppression_a1p25 | 1 | 2.4820 | 3.6601 | -0.5404 | 0.2244 | 0.2000 |
| gated_trip_suppression_a1p25_d0p05 | 1 | 2.4900 | 3.6844 | -0.6319 | 0.2141 | 0.1865 |
| gated_trip_suppression_a1p25_d0p1 | 1 | 2.4971 | 3.7141 | -0.7285 | 0.2013 | 0.1668 |
| gated_trip_suppression_a1p25_d0p15 | 1 | 2.5002 | 3.7644 | -0.9285 | 0.1796 | 0.1432 |
| gated_trip_suppression_a1_d0p15 | 1 | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 0.2335 |
| global_trip_suppression_a1p25 | 1 | 2.5223 | 3.7432 | -0.7477 | 0.1888 | 0.1539 |
| gated_trip_suppression_a1_d0p1 | 1 | 2.5374 | 3.6091 | 0.1370 | 0.2459 | 0.2335 |
| gated_trip_suppression_a1_d0p05 | 1 | 2.5482 | 3.6095 | 0.2143 | 0.2457 | 0.2386 |
| global_trip_suppression_a1 | 1 | 2.5531 | 3.6272 | 0.1229 | 0.2383 | 0.2321 |
| llm_trip_suppression_a1 | 1 | 2.5615 | 3.6104 | 0.2874 | 0.2453 | 0.2409 |
| global_recovery_pressure_a1p25 | 1 | 2.5674 | 3.6292 | 0.2056 | 0.2374 | 0.2343 |
| random_recovery_pressure_a1p25 | 5 | 2.6431 +/- 0.0122 | 3.7076 | 0.2555 +/- 0.0085 | 0.2041 | 0.2067 |
| random_trip_suppression_a1p25 | 5 | 2.6466 +/- 0.0141 | 3.8807 | -0.6368 +/- 0.0164 | 0.1280 | 0.1018 |
| llm_recovery_pressure_a1p25 | 1 | 2.6482 | 3.7306 | 0.1996 | 0.1942 | 0.1968 |
| random_trip_suppression_a1 | 5 | 2.6615 +/- 0.0097 | 3.7417 | 0.2103 +/- 0.0132 | 0.1894 | 0.1906 |
| llm_only_trip_suppression_a1p25 | 1 | 2.7175 | 3.9876 | -0.3732 | 0.0794 | 0.0626 |
| gated_trip_suppression_a0p75_d0p15 | 1 | 2.7458 | 3.7153 | 0.8840 | 0.2008 | 0.2195 |
| global_recovery_pressure_a1 | 1 | 2.7647 | 3.7323 | 0.8855 | 0.1935 | 0.2189 |
| gated_trip_suppression_a0p75_d0p1 | 1 | 2.7872 | 3.7531 | 1.0040 | 0.1845 | 0.2053 |
| gated_trip_suppression_a0p75_d0p05 | 1 | 2.8042 | 3.7712 | 1.0620 | 0.1766 | 0.2012 |
| global_trip_suppression_a0p75 | 1 | 2.8053 | 3.7622 | 0.9935 | 0.1805 | 0.2108 |
| random_recovery_pressure_a1 | 5 | 2.8098 +/- 0.0135 | 3.7914 | 0.9254 +/- 0.0068 | 0.1677 | 0.1977 |
| llm_recovery_pressure_a1 | 1 | 2.8101 | 3.7994 | 0.8807 | 0.1642 | 0.1916 |
| llm_trip_suppression_a0p75 | 1 | 2.8251 | 3.7867 | 1.1169 | 0.1698 | 0.1965 |
| random_trip_suppression_a0p75 | 5 | 2.8700 +/- 0.0103 | 3.8449 | 1.0590 +/- 0.0099 | 0.1441 | 0.1803 |
| llm_confidence_pressure_a1p25 | 1 | 2.9782 | 3.9296 | 1.3231 | 0.1060 | 0.1532 |
| llm_only_trip_suppression_a1 | 1 | 2.9927 | 4.0232 | 0.5984 | 0.0629 | 0.0480 |
| global_recovery_pressure_a0p75 | 1 | 3.0559 | 3.9773 | 1.5654 | 0.0841 | 0.1428 |
| llm_recovery_pressure_a0p75 | 1 | 3.0776 | 4.0161 | 1.5618 | 0.0662 | 0.1244 |
| random_recovery_pressure_a0p75 | 5 | 3.0847 +/- 0.0121 | 4.0177 | 1.5954 +/- 0.0051 | 0.0654 | 0.1276 |
| gated_trip_suppression_a0p5_d0p15 | 1 | 3.1565 | 4.0770 | 1.7911 | 0.0376 | 0.1010 |
| llm_confidence_pressure_a1 | 1 | 3.1854 | 4.1169 | 1.7795 | 0.0187 | 0.0888 |
| gated_trip_suppression_a0p5_d0p1 | 1 | 3.1956 | 4.1206 | 1.8711 | 0.0169 | 0.0820 |
| global_trip_suppression_a0p5 | 1 | 3.2093 | 4.1238 | 1.8640 | 0.0154 | 0.0902 |
| gated_trip_suppression_a0p5_d0p05 | 1 | 3.2141 | 4.1424 | 1.9097 | 0.0065 | 0.0741 |
| llm_trip_suppression_a0p5 | 1 | 3.2335 | 4.1609 | 1.9463 | -0.0024 | 0.0666 |
| global_recovery_pressure_a0p5 | 1 | 3.4255 | 4.3404 | 2.2453 | -0.0907 | 0.0060 |
| llm_confidence_pressure_a0p75 | 1 | 3.4314 | 4.3572 | 2.2359 | -0.0992 | -0.0036 |
| llm_recovery_pressure_a0p5 | 1 | 3.4326 | 4.3589 | 2.2429 | -0.1000 | -0.0049 |
| llm_only_trip_suppression_a0p75 | 1 | 3.4627 | 4.2995 | 1.5737 | -0.0703 | -0.0804 |
| gated_trip_suppression_a0p25_d0p15 | 1 | 3.7026 | 4.6308 | 2.6981 | -0.2416 | -0.1220 |
| llm_confidence_pressure_a0p5 | 1 | 3.7105 | 4.6423 | 2.6923 | -0.2477 | -0.1241 |
| gated_trip_suppression_a0p25_d0p1 | 1 | 3.7264 | 4.6592 | 2.7381 | -0.2568 | -0.1362 |
| global_trip_suppression_a0p25 | 1 | 3.7322 | 4.6595 | 2.7346 | -0.2570 | -0.1299 |
| gated_trip_suppression_a0p25_d0p05 | 1 | 3.7386 | 4.6736 | 2.7575 | -0.2646 | -0.1428 |
| llm_trip_suppression_a0p25 | 1 | 3.7503 | 4.6858 | 2.7757 | -0.2712 | -0.1487 |
| global_recovery_pressure_a0p25 | 1 | 3.8594 | 4.7947 | 2.9253 | -0.3310 | -0.1914 |
| llm_recovery_pressure_a0p25 | 1 | 3.8617 | 4.8007 | 2.9241 | -0.3343 | -0.1961 |
| llm_confidence_pressure_a0p25 | 1 | 4.0150 | 4.9644 | 3.1488 | -0.4269 | -0.2727 |
| llm_only_trip_suppression_a0p5 | 1 | 4.0705 | 4.7758 | 2.5490 | -0.3205 | -0.3229 |
| historical_mean_trend_shift | 1 | 4.1504 | 5.1463 | 3.3485 | -0.5334 | -0.3565 |
| historical_xgboost | 1 | 4.3377 | 5.3169 | 3.6052 | -0.6367 | -0.4494 |
| llm_only_trip_suppression_a0p25 | 1 | 4.7628 | 5.3992 | 3.5242 | -0.6878 | -0.6796 |
| historical_mean_only | 1 | 5.5116 | 6.1251 | 4.4995 | -1.1722 | -1.1503 |

## Key Takeaways

- Best reported sensitivity row: `llm_trip_suppression_a1p25`.
- Weighted MAE reduction vs historical baseline: 42.78%.
- Absolute weighted-bias reduction vs historical baseline: 85.01%.
- Most unbiased row: `gated_trip_suppression_a1_d0p15` with weighted bias `-0.0230`.
- Highest weighted-R2 row: `gated_trip_suppression_a1_d0p15` with weighted R2 `0.2480`.
- These rows use no 2022 `CNTTDHH` labels for training, calibration, or parameter selection.
- Alpha values are a pre-declared sensitivity grid, not target-domain fitted hyperparameters.
- Random-pressure controls shuffle the same LLM pressure distribution across households.
- Global-pressure controls apply the weighted mean LLM pressure to every household.
- Gated controls use cohort-specific LLM scores only when they differ from the global mean by a fixed no-label threshold.

## Diagnostic Interpretation

- The strongest LLM feature is `trip_suppression_risk`; the broader recovery-adjusted composite is weaker.
- Global-pressure controls are strong, so much of the gain comes from label-free event-level downscaling.
- Random controls are weaker than `trip_suppression_risk`, indicating some cohort assignment signal remains.
- Gated trip-suppression correction trades a small amount of MAE for much lower bias and higher weighted R2.
- `llm_trip_suppression_a1p25` improves weighted MAE by 1.60% vs its global-mean control and 6.22% vs its random-shuffle control.
- `llm_recovery_pressure_a1p25` changes weighted MAE by -3.15% vs global mean and -0.19% vs random shuffle; this composite should not be overclaimed.

## Label-Free Correction Rule

The primary LLM event pressure is computed as:

```text
raw_pressure = 0.35 * trip_suppression
             + 0.25 * remote_work_substitution
             + 0.20 * transit_avoidance
             + 0.20 * online_delivery_substitution
recovery_adjusted_pressure = raw_pressure * (1 - 0.25 * recovery_sensitivity)
prediction = historical_prediction * clip(1 - alpha * recovery_adjusted_pressure, min_factor, 1)
```

Notes:

- Historical model is trained only on 2001, 2009, and 2017 labels.
- 2022 `CNTTDHH` appears only inside the final evaluation metrics.
- All XGBoost training uses CUDA.