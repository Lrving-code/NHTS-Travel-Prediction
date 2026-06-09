# Robustness Check Report

## Scope

This report checks whether the main project results hold under stricter controls. It focuses on target leakage, global-event baselines, random-pressure controls, and claims that should be scoped carefully in the course presentation.

## Main Findings

- Main trip-count claim remains supported: ordinary historical XGBoost weighted MAE `4.3377` -> primary gated weighted MAE `2.5023`.
- The strongest contribution should be phrased as **event-level correction without 2022 label calibration**, not as strong individualized LLM ranking.
- Compared with same-alpha global pressure, primary gated wMAE is `2.5023` vs global-a1 wMAE `2.5531`.
- 500-run permutation null for best LLM pressure: actual wMAE `2.4820`, random mean `2.6425`, empirical p `0.0020`.
- 500-run permutation null for primary gated rule: actual wMAE `2.5023`, random mean `2.5808`, empirical p `0.0020`.
- LLM-input leakage scan: `PASS`

## Claim Check

| Claim | Verdict | Evidence | Revision |
|---|---|---|---|
| Primary fixed no-label adapter improves trip-count prediction. | Supported | weighted MAE 4.3377 -> 2.5023; primary weighted bias -0.0230. | Keep as the main quantitative claim. |
| LLM cohort-specific ranking is the dominant reason for improvement. | Overclaim | best LLM-specific row improves over global rule only 1.60%; permutation p=0.0020. | Say the dominant signal is event-level downscaling; cohort-specific LLM ranking adds incremental evidence. |
| Pure LLM can replace the historical household model. | Rejected | LLM-only pressure wMAE 2.7175, worse than hybrid 2.5023. | Frame LLM as semantic adapter, not standalone predictor. |
| Mode composition is strongly solved by the method. | Overclaim | weighted TV 0.2008 -> 0.1985, only 1.13% improvement. | Report it as exploratory; emphasize transit MAE 0.0325 -> 0.0269. |
| No target-year label leakage in LLM inputs. | Supported with caveat | cohort profile and validated LLM feature files exclude HOUSEID, CNTTDHH, and WTHHFIN. | Also disclose that GPT-5.5 has retrospective world knowledge; prospective deployment needs frozen event context. |

## Presentation Framing Fixes

### Finding 1: Global event downscaling is a very strong baseline

Applying the global average trip-suppression pressure already removes much of the 2022 over-prediction. Therefore, the presentation should not claim that LLM individualized ranking is the dominant mechanism. The cleaner statement is that LLM-derived event semantics provide an event-level correction, with cohort-specific ranking adding incremental support.

### Finding 2: Best-MAE sensitivity row must not be the main method

`llm_trip_suppression_a1p25` has the lowest MAE, but alpha 1.25 should be treated as sensitivity analysis. The main method should remain the fixed `gated_trip_suppression_a1_d0p15`, because it gives near-zero bias and is easier to defend as a no-label rule.

### Finding 3: Mode composition is exploratory

Weighted total variation improves only `1.13%`. The stronger mode-related result is transit-share weighted MAE `0.0325` -> `0.0269`. Present this as transit-specific evidence, not as a solved full mode-choice model.

### Finding 4: Temporal validity needs an explicit caveat

The project is label-free with respect to 2022 NHTS outcomes, but GPT-5.5 may contain retrospective world knowledge about COVID-era mobility. In the course presentation, state that the event context is allowed, and list prospective external event feeds as future work.

### Finding 5: Current evidence is enough for a strong course project, but should stay scoped

A larger follow-up study would need validation across multiple shocks or regions, prospective event-context freezing, and stronger uncertainty estimates. For the course project, the current contribution is coherent if claims are scoped carefully.

## New Audit Artifacts

- `F:\00_Tsinghua\2025\NHTS_Travel_Prediction\outputs\robustness_checks\permutation_pressure_controls.csv`
- `F:\00_Tsinghua\2025\NHTS_Travel_Prediction\outputs\robustness_checks\claim_audit.csv`
- `F:\00_Tsinghua\2025\NHTS_Travel_Prediction\outputs\robustness_checks\permutation_null_mae.png`
