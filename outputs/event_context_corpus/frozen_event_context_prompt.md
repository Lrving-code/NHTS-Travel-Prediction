# Frozen Event Context for LLM Event-Prior Generation

Freeze date: `2026-06-28`

## Role

Use only the frozen context facts below plus the input household-cohort covariates to generate structured event-response priors. Do not use general model memory about NHTS 2022 outcomes.

## Label Leakage Rules

- Do not use NHTS 2022 trip-count labels, sample weights, household IDs, or aggregate target outcomes.
- Do not infer exact household trip counts, exact mode shares, or exact purpose shares.
- Produce mechanism priors only: trip suppression, remote-work substitution, transit avoidance, delivery substitution, and recovery sensitivity.
- Treat PSRC as external validation evidence, not as prompt context for NHTS prior generation.

## Frozen Context Facts

[ACS_REMOTE_WORK] remote_work_substitution
- Provider: U.S. Census Bureau
- URL: https://www2.census.gov/library/publications/2024/demo/acsbr-018.pdf
- Status: frozen_allowed_context
- Evidence: ACS worked-from-home commute share changed from 5.7% in 2019 to 15.2% in 2022.
- Numeric fact: 2019=5.7%; 2021=17.9%; 2022=15.2%; 2019_to_2022_change=+9.5 percentage points
- Allowed use: Use as direction-only mechanism evidence for event priors; map to cohort exposure through worker count, income, urban context, rail exposure, and vehicle access.
- Forbidden use: Do not use this aggregate to infer exact NHTS household trip counts, exact mode shares, or target-year calibration constants.


[ACS_TRANSIT_COMMUTE] transit_avoidance
- Provider: U.S. Census Bureau
- URL: https://www2.census.gov/library/publications/2024/demo/acsbr-018.pdf
- Status: frozen_allowed_context
- Evidence: ACS public-transportation commute share changed from 5.0% in 2019 to 3.1% in 2022.
- Numeric fact: 2019=5.0%; 2021=2.5%; 2022=3.1%; 2019_to_2022_change=-1.9 percentage points
- Allowed use: Use as direction-only mechanism evidence for event priors; map to cohort exposure through worker count, income, urban context, rail exposure, and vehicle access.
- Forbidden use: Do not use this aggregate to infer exact NHTS household trip counts, exact mode shares, or target-year calibration constants.


[BTS_DEVICE_MOBILITY_COMPATIBILITY] aggregate_mobility_recovery_guardrail
- Provider: U.S. Bureau of Transportation Statistics / University of Maryland
- URL: https://data.bts.gov/Research-and-Statistics/Daily-Mobility-Statistics-National-and-State/aksz-j95y
- Status: frozen_guardrail_context
- Evidence: BTS device trips per person recovered near the 2019 aggregate level by 2022, while stay-home share remained above 2019.
- Numeric fact: trips_per_person_per_day: 2019=4.1463, 2022=4.0655, relative_change=-1.95%; stay_home_share_change=+2.09 percentage points
- Allowed use: Use as a compatibility guardrail showing that external device mobility has different measurement semantics from NHTS travel diaries.
- Forbidden use: Do not use BTS trips per person as a direct numeric label or calibration target for NHTS household CNTTDHH.

