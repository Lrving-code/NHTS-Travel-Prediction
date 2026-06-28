# External Validation and Compatibility Audit

This audit uses independent non-NHTS sources only for evaluation and
mechanism checking. No external data are used to train or calibrate the NHTS
prediction model.

## Mechanism-Level External Validation

The strongest currently available external evidence is mechanism-level rather
than household-level. The U.S. Census Bureau ACS commuting brief reports that
post-pandemic commuting retained two structural shifts that match the LLM event
priors used in this project:

Source: https://www2.census.gov/library/publications/2024/demo/acsbr-018.pdf

| Mechanism | External measure | 2019 | 2021 | 2022 | 2019->2022 change | LLM prior |
|---|---|---:|---:|---:|---:|---|
| remote_work_substitution | ACS worked-from-home commute share | 5.7% | 17.9% | 15.2% | +9.5 pp | increase |
| transit_avoidance | ACS public-transportation commute share | 5.0% | 2.5% | 3.1% | -1.9 pp | decrease |

Interpretation: ACS supports the event semantics used by the LLM adapter:
work-from-home remained far above the 2019 level, while public-transportation
commuting remained below the 2019 level in 2022. This validates the direction
of `remote_work_substitution` and `transit_avoidance` priors, but it is not a
direct household trip-count accuracy test.

## BTS Trip-Count Compatibility Guardrail

BTS / University of Maryland Daily Mobility Statistics provide independent
daily national trip counts from mobile-device-derived mobility statistics:
https://data.bts.gov/Research-and-Statistics/Daily-Mobility-Statistics-National-and-State/aksz-j95y

We tested whether BTS trips per person can be used as an external numeric target
for NHTS household travel-day trip counts. The answer is **no**: the two data
products have different trip definitions, observation mechanisms, and baseline
years. The check is retained as a guardrail against overclaiming.

### BTS Reference

| Year | Days | Trips per person per day | Stay-home share |
|---:|---:|---:|---:|
| 2019 | 365 | 4.1463 | 0.1943 |
| 2022 | 365 | 4.0655 | 0.2152 |

BTS 2019 to 2022 trip-per-person shift:
`-1.95%`.

### NHTS-to-BTS Compatibility Check

| Method | NHTS 2017->2022 shift | BTS shift | External error | Direction match |
|---|---:|---:|---:|---|
| historical_xgboost | -4.10% | -1.95% | 2.15 pp | yes |
| recovery_event_prior_a1 | -39.23% | -1.95% | 37.28 pp | yes |
| global_event_prior_a1 | -49.00% | -1.95% | 47.05 pp | yes |
| nhts_2022_observed_evaluation_only | -50.59% | -1.95% | 48.64 pp | yes |
| gated_trip_suppression_a1_d0p15 | -50.88% | -1.95% | 48.94 pp | yes |

## Interpretation

- ACS mechanism evidence supports the qualitative event priors used by the
  LLM adapter.
- BTS aggregate device trips should **not** be used as a direct external
  numeric label for NHTS household `CNTTDHH`: even the observed 2022 NHTS
  target has a large shift mismatch against BTS.
- Therefore, the defensible external claim is mechanism-level validation plus
  a trip-count compatibility guardrail, not external household-level MAE.

## Scope

This closes a course-project-level external evidence gap but does not close the
stronger paper-submission requirement of household-level external microdata
validation. For a paper submission, this result should be reported as
mechanism-level external validation and paired with a limitation statement.
