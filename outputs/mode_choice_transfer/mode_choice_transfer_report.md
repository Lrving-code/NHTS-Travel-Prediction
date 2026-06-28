# Harmonized Mode-Choice Transfer Experiment

## Scope

This experiment evaluates trip-level mode-choice transfer after mapping year-specific NHTS `TRPTRANS` codes into comparable `MODE_GROUP` labels. It trains only on 2017 labels, selects the XGBoost operating point by 2017 validation survey-weighted macro F1, and uses 2022 labels only for final evaluation.

## 2022 Test Metrics

| Method | Weighted accuracy | Weighted balanced acc. | Weighted macro F1 | Survey-weighted F1 | Weighted top-2 | Weighted log loss | Unweighted accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost_survey_weighted | 0.9297 | 0.4343 | 0.4663 | 0.9241 | 0.9754 | 0.2339 | 0.9373 |
| xgboost_unweighted | 0.9280 | 0.4259 | 0.4511 | 0.9235 | 0.9765 | 0.2319 | 0.9373 |
| xgboost_class_balanced | 0.7852 | 0.5501 | 0.3722 | 0.8355 | 0.9124 | 0.7026 | 0.7989 |
| historical_prior_2017 | 0.8696 | 0.1429 | 0.1329 | 0.8089 | 0.9381 | 0.5605 | 0.8821 |

## Validation-Selected Operating Point

The validation-selected model is `xgboost_survey_weighted`. On 2022 it reaches survey-weighted accuracy `0.9297`, weighted balanced accuracy `0.4343`, and weighted macro F1 `0.4663`. The corresponding unweighted accuracy is `0.9373`.

The highest weighted-balanced-accuracy model is `xgboost_class_balanced` at `0.5501`, but it lowers weighted macro F1 and log-loss calibration. The paper-safe interpretation should therefore report the trade-off rather than treating ordinary accuracy as sufficient.

The ordinary accuracy is high because private-vehicle trips dominate the target distribution. The paper-safe interpretation should therefore emphasize balanced accuracy, macro F1, and per-class behavior rather than claiming a solved full mode-choice task.

- Confusion matrix: `outputs/mode_choice_transfer/xgboost_survey_weighted_test_confusion_matrix.png`

## Per-Class Test Metrics for Selected Model

| Mode group | Label | Weighted precision | Weighted recall | Weighted F1 | Weighted support | Unweighted F1 | Rows |
|---|---|---:|---:|---:|---:|---:|---:|
| private_vehicle | Private vehicle | 0.9630 | 0.9752 | 0.9690 | 27020.6 | 0.9715 | 27409 |
| walk | Walk | 0.7043 | 0.7454 | 0.7243 | 2129.6 | 0.7506 | 2103 |
| bus_paratransit | Bus, shuttle, or paratransit | 0.7136 | 0.7205 | 0.7170 | 1193.7 | 0.7019 | 849 |
| bike_micromobility | Bike or micromobility | 0.2030 | 0.0785 | 0.1132 | 302.9 | 0.1381 | 304 |
| rail_transit | Rail transit | 0.4864 | 0.0499 | 0.0906 | 166.3 | 0.1461 | 155 |
| taxi_ridehail | Taxi or ride-hail | 0.4293 | 0.2005 | 0.2734 | 141.8 | 0.1512 | 118 |
| air_water_other | Air, water, or other | 0.6203 | 0.2701 | 0.3763 | 119.0 | 0.4000 | 136 |

## Generated Files

- `mode_choice_transfer_metrics.csv`: validation and 2022 test aggregate metrics with unweighted and survey-weighted columns.
- `mode_choice_per_class_metrics.csv`: per-class unweighted and survey-weighted precision, recall, F1, and support.
- `*_test_confusion_matrix.png`: row-normalized survey-weighted 2022 confusion matrix for the validation-selected model.

