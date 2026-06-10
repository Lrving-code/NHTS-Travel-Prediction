# PSRC Household-Level External Microdata Validation

This validation uses public Puget Sound Regional Council household travel survey
microdata, independent of NHTS. It is an external recovery-transfer check:
train on PSRC 2021 household/day records, predict later PSRC household/day trip
rates, and apply a BTS-derived recovery factor without using target-year PSRC
labels for calibration.

PSRC source: https://psrc-psregcncl.hub.arcgis.com/datasets/PSREGCNCL::household-travel-survey-households/about

## Data Harmonization

- Target: household trips per observed diary day.
- Source table: PSRC `Days`, aggregated from person-day `num_trips` to
  household-day trips, then averaged by household and survey wave.
- Household covariates: household size, vehicle count, worker count, broad
  income bin, diary platform, and observed diary days.
- Weights: PSRC `hh_weight`.

## Survey-Year Summary

| Year | Households | Weighted trips/day | Weighted zero-trip share | Mean observed days |
|---:|---:|---:|---:|---:|
| 2021 | 573 | 6.2703 | 0.0574 | 1.00 |
| 2023 | 3860 | 7.2902 | 0.0790 | 2.39 |
| 2025 | 2769 | 8.1978 | 0.0331 | 2.05 |

## Method Comparison

| Method | Test year | Event factor | wMAE | wRMSE | wBias | wR2 |
|---|---:|---:|---:|---:|---:|---:|
| bts_recovery_event_adapter | 2023 | 1.1438 | 3.6855 | 5.2503 | -0.2646 | 0.1949 |
| psrc_2021_xgboost | 2023 | 1.0000 | 3.6908 | 5.3440 | -1.1480 | 0.1659 |
| bts_recovery_event_adapter | 2025 | 1.1438 | 3.9570 | 6.0450 | -1.1363 | 0.2280 |
| psrc_2021_xgboost | 2025 | 1.0000 | 4.0294 | 6.3147 | -2.0242 | 0.1575 |

## Interpretation

- On the primary external 2021->2023 recovery transfer, the BTS recovery adapter
  changes weighted MAE from `3.6908` to
  `3.6855`.
- The MAE gain is modest (`0.0052` trips/day), but the systematic bias
  improves more clearly: absolute weighted bias changes from
  `1.1480` to `0.2646`
  trips/day.
- This is not a claim that the NHTS 2022 model directly transfers to PSRC.
  It is household-level external evidence that event-scale mobility recovery
  factors can improve label-free temporal transfer on an independent travel
  survey.

## Limitation

The current PSRC Hub CSV exposes complete person-day microdata for 2021, 2023,
and 2025. The older 2017/2019 day/trip microdata are not exposed in the same
current CSV endpoint, so this validation uses 2021 as the source wave rather
than pre-pandemic PSRC microdata.
