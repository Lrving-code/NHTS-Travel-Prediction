# Harmonized Mode-Choice Transfer Experiment

## Scope

This experiment evaluates trip-level mode-choice transfer after mapping year-specific NHTS `TRPTRANS` codes into comparable `MODE_GROUP` labels. It trains only on 2017 labels, selects the XGBoost operating point by 2017 validation macro F1, and uses 2022 labels only for final evaluation.

## 2022 Test Metrics

| Method | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Top-2 accuracy | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| xgboost_unweighted | 0.9373 | 0.4309 | 0.4658 | 0.9324 | 0.9786 | 0.2110 |
| xgboost_class_balanced | 0.6758 | 0.6167 | 0.3553 | 0.7702 | 0.8738 | 0.9115 |
| historical_prior_2017 | 0.8821 | 0.1429 | 0.1339 | 0.8268 | 0.9497 | 0.5117 |

## Validation-Selected Operating Point

The validation-selected model is `xgboost_unweighted`. On 2022 it reaches accuracy `0.9373`, balanced accuracy `0.4309`, and macro F1 `0.4658`.

The highest balanced-accuracy model is `xgboost_class_balanced` at `0.6167`, but it lowers macro F1 and log-loss calibration. The paper-safe interpretation should therefore report the trade-off rather than treating ordinary accuracy as sufficient.

The ordinary accuracy is high because private-vehicle trips dominate the target distribution. The paper-safe interpretation should therefore emphasize balanced accuracy, macro F1, and per-class behavior rather than claiming a solved full mode-choice task.

- Confusion matrix: `outputs/mode_choice_transfer/xgboost_unweighted_test_confusion_matrix.png`

## Per-Class Test Metrics for Selected Model

| Mode group | Label | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| private_vehicle | Private vehicle | 0.9669 | 0.9770 | 0.9719 | 27409 |
| walk | Walk | 0.7232 | 0.7903 | 0.7553 | 2103 |
| bus_paratransit | Bus, shuttle, or paratransit | 0.7152 | 0.6832 | 0.6988 | 849 |
| bike_micromobility | Bike or micromobility | 0.2747 | 0.1645 | 0.2058 | 304 |
| rail_transit | Rail transit | 0.7000 | 0.1355 | 0.2270 | 155 |
| air_water_other | Air, water, or other | 0.7778 | 0.2574 | 0.3867 | 136 |
| taxi_ridehail | Taxi or ride-hail | 0.0714 | 0.0085 | 0.0152 | 118 |

## Generated Files

- `mode_choice_transfer_metrics.csv`: validation and 2022 test aggregate metrics.
- `mode_choice_per_class_metrics.csv`: per-class precision, recall, F1, and support.
- `*_test_confusion_matrix.png`: row-normalized 2022 confusion matrix for the validation-selected model.

