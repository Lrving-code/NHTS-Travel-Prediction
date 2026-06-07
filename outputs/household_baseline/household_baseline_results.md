# Household Baseline Results

| Experiment | Train Years | Test Year | Device | Weighted MAE | Weighted RMSE | Weighted Bias | R2 |
|---|---|---:|---|---:|---:|---:|---:|
| direct_2017_to_2022 | 2017 | 2022 | cuda | 4.5921 | 5.6292 | 3.9598 | -0.6017 |
| pooled_history_to_2022 | 2001+2009+2017 | 2022 | cuda | 4.7163 | 5.8269 | 4.1255 | -0.7121 |
| pooled_history_with_year_to_2022 | 2001+2009+2017 | 2022 | cuda | 4.4062 | 5.4076 | 3.7202 | -0.5055 |

Notes:

- Target: `CNTTDHH`.
- Training uses NHTS household final weights `WTHHFIN`.
- Metrics include both unweighted values in CSV and weighted values in this summary.