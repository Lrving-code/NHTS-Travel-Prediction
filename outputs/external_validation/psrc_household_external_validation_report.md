# PSRC Household-Level External Microdata Validation

This validation uses public Puget Sound Regional Council household travel survey
microdata, independent of NHTS. It is a direct external pre/post replication:
train on PSRC 2017+2019 household/day records, predict later PSRC household/day
trip rates, and compare fixed event factors from independent national sources
without using target-year PSRC labels for calibration.

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
| 2017 | 3275 | 8.3609 | 0.0211 | 2.27 |
| 2019 | 3044 | 8.9555 | 0.0350 | 3.06 |
| 2021 | 1929 | 6.3386 | 0.0571 | 1.00 |
| 2023 | 3860 | 7.2902 | 0.0790 | 2.39 |
| 2025 | 2769 | 8.1978 | 0.0331 | 2.05 |

## Method Comparison

| Method | Train years | Test year | Event factor | wMAE | wRMSE | wBias | wR2 |
|---|---|---:|---:|---:|---:|---:|---:|
| acs_remote_work_suppression_adapter | 2017+2019 | 2023 | 0.9050 | 3.2498 | 4.6158 | +0.2605 | 0.3777 |
| bts_recovery_event_adapter | 2017+2019 | 2023 | 1.0679 | 3.6228 | 5.0331 | +1.6196 | 0.2601 |
| psrc_pre_pandemic_xgboost | 2017+2019 | 2023 | 1.0000 | 3.4271 | 4.8025 | +1.0532 | 0.3264 |
| acs_remote_work_suppression_adapter | 2017+2019 | 2025 | 0.9050 | 3.2602 | 5.0524 | -0.9809 | 0.4607 |
| bts_recovery_event_adapter | 2017+2019 | 2025 | 1.0679 | 3.3239 | 4.8745 | +0.3180 | 0.4980 |
| psrc_pre_pandemic_xgboost | 2017+2019 | 2025 | 1.0000 | 3.2561 | 4.8958 | -0.2233 | 0.4936 |

## Interpretation

- On the primary external 2017+2019->2023 pre/post replication, the ACS remote-work
  suppression adapter
  changes weighted MAE from `3.4271` to
  `3.2498`.
- The MAE gain is `0.1773` trips/day, and the systematic bias
  also improves: absolute weighted bias changes from
  `1.0532` to `0.2605`
  trips/day.
- The BTS recovery factor is retained as a compatibility guardrail: it changes
  2023 wMAE to `3.6228` and wBias to
  `+1.6196`, showing that device-mobility recovery rates
  should not be treated as direct household-survey trip-count adapters.
- This is not a claim that the NHTS 2022 model directly transfers to PSRC.
  It is household-level external evidence that event-scale mobility factors can
  improve label-free pre/post temporal transfer on an independent travel survey.

## Limitation

PSRC is a regional travel survey and uses a different sample frame, questionnaire,
and diary protocol from NHTS. The evidence should therefore be reported as
external replication of the event-adaptation principle, not as direct numerical
validation of the NHTS 2022 household predictions.
