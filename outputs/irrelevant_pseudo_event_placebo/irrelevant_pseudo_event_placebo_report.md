# Irrelevant Pseudo-Event Placebo Control

## Purpose

This is a negative-control experiment. It asks whether the 2022 improvement can be reproduced by any cohort-level score that has the same marginal pressure distribution as the LLM event prior, even when the score is unrelated to post-pandemic travel mechanisms.

The answer should not be read as causal proof. It is a guardrail against the weaker explanation that the method works merely because it applies an arbitrary household-level shrinkage score.

## Design

- Target and evaluation year: 2022 household `CNTTDHH`.
- Base model: historical XGBoost routine prediction trained without 2022 labels.
- True event prior: LLM trip-suppression pressure, used only through fixed no-label correction rules.
- Pseudo-event priors: deterministic non-travel rankings for media-upgrade adoption, calendar-app adoption, and household-ID hash noise.
- Distribution matching: every pseudo-event pressure is rank-matched to the actual LLM pressure distribution and mean-adjusted to the same weighted pressure mean.

## Key Results

- Historical XGBoost wMAE: `4.3377`.
- Global actual event prior wMAE: `2.5531`.
- Primary gated event adapter wMAE: `2.5023`.
- Best irrelevant ranked pseudo-event wMAE: `2.6274` (`pseudo_random_hash_rank_matched_a1`).
- Best irrelevant gated pseudo-event wMAE: `2.5610` (`pseudo_random_hash_rank_matched_gated_a1_d0p15`).
- Best irrelevant ranked pseudo-event is `+0.1252` wMAE relative to the primary method.

## Interpretation

A global event downshift is a strong baseline because 2022 has a broad post-pandemic travel suppression. However, irrelevant cohort rankings do not replace the mechanism-aligned event prior. The defensible claim is therefore not that any LLM-generated score works, but that a scoped event prior can adapt a routine household model under a known societal shock while preserving label-free calibration discipline.

## Pressure Matching Check

| Pseudo-event | Weighted mean | Weighted std | Actual weighted mean | Actual weighted std |
|---|---:|---:|---:|---:|
| pseudo_media_upgrade | 0.46823503 | 0.11112700 | 0.46823503 | 0.11744767 |
| pseudo_calendar_app | 0.46823503 | 0.10381432 | 0.46823503 | 0.11744767 |
| pseudo_random_hash | 0.46823503 | 0.10812333 | 0.46823503 | 0.11744767 |

## Full Metric Table

| Method | Family | wMAE | wRMSE | wBias | wR2 | Delta vs primary |
|---|---|---:|---:|---:|---:|---:|
| global_actual_event_a1 | global_event_prior | 2.5531 | 3.6272 | +0.1229 | 0.2383 | +0.0508 |
| llm_only_pressure_a1p25 | llm_only_control | 2.7175 | 3.9876 | -0.3732 | 0.0794 | +0.2152 |
| primary_gated_event_a1_d0p15 | main_method | 2.5023 | 3.6038 | -0.0230 | 0.2480 | +0.0000 |
| pseudo_random_hash_rank_matched_gated_a1_d0p15 | pseudo_event_gated | 2.5610 | 3.6781 | -0.0703 | 0.2168 | +0.0587 |
| pseudo_calendar_app_rank_matched_gated_a1_d0p15 | pseudo_event_gated | 2.5720 | 3.6931 | -0.0371 | 0.2103 | +0.0697 |
| pseudo_media_upgrade_rank_matched_gated_a1_d0p15 | pseudo_event_gated | 2.6298 | 3.7797 | -0.1619 | 0.1728 | +0.1275 |
| pseudo_media_upgrade_global_same_mean_a1 | pseudo_event_global | 2.5531 | 3.6272 | +0.1229 | 0.2383 | +0.0508 |
| pseudo_calendar_app_global_same_mean_a1 | pseudo_event_global | 2.5531 | 3.6272 | +0.1229 | 0.2383 | +0.0508 |
| pseudo_random_hash_global_same_mean_a1 | pseudo_event_global | 2.5531 | 3.6272 | +0.1229 | 0.2383 | +0.0508 |
| pseudo_random_hash_rank_matched_a1 | pseudo_event_ranked | 2.6274 | 3.7242 | +0.1143 | 0.1970 | +0.1252 |
| pseudo_calendar_app_rank_matched_a1 | pseudo_event_ranked | 2.6360 | 3.7310 | +0.1388 | 0.1941 | +0.1337 |
| pseudo_media_upgrade_rank_matched_a1 | pseudo_event_ranked | 2.7332 | 3.8681 | -0.0307 | 0.1337 | +0.2309 |
| historical_xgboost | traditional_baseline | 4.3377 | 5.3169 | +3.6052 | -0.6367 | +1.8354 |

## Artifact

- Figure: `outputs\irrelevant_pseudo_event_placebo\irrelevant_pseudo_event_placebo_mae.png`
- Metrics: `outputs\irrelevant_pseudo_event_placebo\irrelevant_pseudo_event_placebo_metrics.csv`
- Pressure summary: `outputs\irrelevant_pseudo_event_placebo\irrelevant_pseudo_event_pressure_summary.csv`
