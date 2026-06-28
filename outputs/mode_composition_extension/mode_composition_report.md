# Household Mode Composition Extension

## Scope

This extension predicts household-level 2022 mode shares derived from trip records. It is separate from the main `CNTTDHH` trip-count task and should not be compared with trip-level `TRPTRANS` classification accuracy.

## Dataset

- Training year: `2017` households with at least one valid trip.
- Test year: `2022` households with at least one valid trip.
- Targets: private vehicle, walk, bike, transit, taxi/ridehail, and other shares.
- 2017 and 2022 use year-specific official `TRPTRANS` codebook mappings into common broad categories.

## Weighted Mode Distribution

| Year | Households | Trips/HH | Private | Walk | Bike | Transit | Taxi/Ridehail | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 117222 | 8.55 | 0.818 | 0.105 | 0.010 | 0.041 | 0.005 | 0.021 |
| 2022 | 6188 | 5.03 | 0.859 | 0.087 | 0.009 | 0.019 | 0.005 | 0.022 |

## Main Metrics

| Method | Weighted TV | Weighted Mean Share MAE | Weighted Dominant Accuracy | Transit Weighted MAE | Transit Weighted Bias |
|---|---:|---:|---:|---:|---:|

## Derived Mode-Specific Trip Volumes

Mode shares can be combined with trip-count predictions to estimate trips by mode. This turns the project into a household travel-behavior system rather than a single-output regression task.

| Method | Total mode-trip MAE | Mean component MAE | Total trip bias | Transit trip MAE | Active trip MAE |
|---|---:|---:|---:|---:|---:|
| traditional_count_x_traditional_mode | 4.8467 | 0.8078 | 2.9008 | 0.1649 | 0.8079 |
| gated_count_x_traditional_mode | 3.2918 | 0.5486 | -0.9119 | 0.1114 | 0.5956 |
| gated_count_x_llm_mode | 3.2802 | 0.5467 | -0.9119 | 0.0956 | 0.5972 |
| best_count_x_llm_mode | 3.3384 | 0.5564 | -1.3865 | 0.0914 | 0.5713 |
| historical_mean_2017 | 0.2647 | 0.0882 | 0.8933 | 0.0567 | 0.0220 |
| historical_xgboost | 0.2008 | 0.0669 | 0.9029 | 0.0325 | 0.0035 |
| llm_transit_avoidance_a0p5 | 0.1996 | 0.0665 | 0.9042 | 0.0297 | -0.0014 |
| global_transit_avoidance_a0p5 | 0.1998 | 0.0666 | 0.9033 | 0.0307 | 0.0009 |
| llm_transit_avoidance_a0p75 | 0.1990 | 0.0663 | 0.9045 | 0.0283 | -0.0041 |
| global_transit_avoidance_a0p75 | 0.1993 | 0.0664 | 0.9037 | 0.0298 | -0.0004 |
| llm_transit_avoidance_a1 | 0.1985 | 0.0662 | 0.9046 | 0.0269 | -0.0070 |
| global_transit_avoidance_a1 | 0.1989 | 0.0663 | 0.9042 | 0.0289 | -0.0018 |

## Interpretation

- Historical XGBoost weighted total variation: `0.2008`.
- Historical mean weighted total variation: `0.2647`.
- Best reported row by weighted total variation: `llm_transit_avoidance_a1`.
- Best row reduces weighted total variation by `1.13%` vs historical XGBoost.
- Best row reduces transit-share weighted MAE by `17.35%` vs historical XGBoost.

## Figures

- `figures/mode_distribution_shift.png`
- `figures/mode_metric_comparison.png`

## Caveat

Treat this as an exploratory extension. The main course-project story is broader: trip generation, mode composition, and derived mode-specific trip volumes.