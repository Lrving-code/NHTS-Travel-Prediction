# Causal Guardrail Evidence Pack

## Purpose

This evidence pack consolidates the checks that keep the LLM component auditable. It supports causal plausibility and label-free event adaptation; it does not claim causal-effect identification.

## Core Claim Boundary

- Allowed: LLM-derived event priors are structured mechanism proxies for post-pandemic mobility adaptation.
- Allowed: Causal guardrails reduce leakage and arbitrary-correction risk.
- Not allowed: The LLM estimates the causal effect of COVID-19 on household travel.
- Not allowed: The LLM directly predicts household trip counts better than tabular models.

## Headline Evidence

- Historical XGBoost wMAE `4.3377` -> primary gated adapter `2.5023` (42.31% reduction).
- Weighted bias `3.6052` -> `-0.0230`; weighted R2 = `0.2480`.
- Best stronger non-LLM baseline CatBoost GPU wMAE `4.2196`; primary adapter is 40.70% lower.
- Leakage audit: `1327` cohort profiles, `1327` prompt payload records, `1` validated feature-output record(s), and `89` batch prompts checked, `0` forbidden-field violations.
- Random-pressure negative control: empirical p = `0.0020` over `500` permutations.
- Zero-shot LLM rule-tree baseline wMAE `2.7508`; primary hybrid is lower by `0.2485` wMAE.
- LLM rule + 500 historical calibration wMAE `2.7723`; full-history rule calibration wMAE `2.7870`.
- Best irrelevant ranked pseudo-event wMAE `2.6274`; best irrelevant gated pseudo-event wMAE `2.5610`.
- Pre-COVID placebo: full 2022-style suppression worsens 2017 wMAE by `0.1225`; stronger a1.25 worsens it by `0.6361`.

## Evidence Table

| Check | Verdict | Key result | Implication | Artifact |
|---|---|---|---|---|
| 2022 is an event-shift target | Supported | pre-COVID mean absolute weighted bias = 0.9024; 2022 absolute weighted bias = 3.7202 | The target year is harder than routine temporal transfer; ordinary historical models systematically overpredict post-pandemic travel. | `outputs\temporal_transfer_validation\temporal_transfer_metrics.csv` |
| Main label-free event adapter improves prediction | Supported | historical XGBoost wMAE 4.3377 -> primary gated wMAE 2.5023 (42.31% reduction); wBias 3.6052 -> -0.0230; wR2 0.2480 | The strongest quantitative claim is label-free event adaptation, not direct LLM prediction. | `outputs\label_free_llm_adaptation\label_free_llm_adaptation_metrics.csv` |
| Stronger non-LLM tabular baselines do not remove the shift | Supported | best extra tabular baseline CatBoost GPU wMAE = 4.2196, wBias = 3.4873; primary adapter is 40.70% lower in wMAE | Model capacity alone does not solve the mechanism shift; the event mechanism must be represented. | `outputs\strong_baselines\strong_tabular_baseline_metrics.csv` |
| Prompt/schema target leakage | Passed with caveat | 1327 cohort profiles, 1327 prompt payload records, 1 validated feature-output record(s), 89 batch prompts; forbidden-field violations = 0; required guardrails present = True | The LLM branch receives cohort covariates and event context, not target labels, weights, IDs, or aggregate 2022 target outcomes. Retrospective world knowledge remains a limitation. | `outputs\leakage_audit\llm_input_leakage_audit.csv` |
| Random-pressure negative control | Supported | actual gated pressure beats 500 random assignments; empirical p = 0.0020 | The structured event pressure contains more information than arbitrary random perturbation. | `outputs\robustness_checks\permutation_pressure_controls.csv` |
| Global event prior as a low-cost baseline | Boundary condition | global a1 wMAE = 2.5531; primary gated wMAE = 2.5023; absolute gap = 0.0508 | The dominant signal is event-level suppression. Cohort-aware LLM priors add auditable refinement, not the entire gain. | `outputs\robustness_checks\permutation_pressure_controls.csv` |
| Pre-COVID placebo event correction | Supported as guardrail | 2017 routine wMAE = 4.1272; full 2022 suppression a1 wMAE = 4.2497 (delta 0.1225); a1.25 wMAE = 4.7632 (delta 0.6361) | The event prior is not a universal downshift. It needs event context and strength discipline. | `outputs\pre_covid_placebo_event_correction\pre_covid_placebo_event_correction_metrics.csv` |
| Hybrid design versus LLM-only prediction | Supported | LLM-only pressure wMAE = 2.7175; primary hybrid wMAE = 2.5023 | The LLM is useful as an event-prior generator, while the routine household predictor remains necessary. | `outputs\label_free_llm_adaptation\label_free_llm_adaptation_metrics.csv` |
| Zero-shot LLM-authored rule tree | Supported as ablation | zero-shot rule-tree wMAE = 2.7508, wBias = 0.5000; pseudo-label tree wMAE = 2.7767, wBias = 0.5000; primary hybrid wMAE = 2.5023 | A direct LLM tree is a useful cold-start baseline, but it behaves like a qualitative belief tree and remains less calibrated than the routine-model-plus-event-prior design. | `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_metrics.csv` |
| LLM rule plus small historical calibration | Useful bridge baseline, not main winner | rule + 500 historical samples wMAE = 2.7723, wBias = 0.4627; full-history rule calibration wMAE = 2.7870, wBias = 0.6068; primary hybrid wMAE = 2.5023 | Calibrating LLM-extracted rules with routine historical data is a coherent cold-start bridge, but it can overpredict 2022 unless the event-shift correction remains explicit. | `outputs\llm_rule_small_data_calibration\method_spectrum_metrics.csv` |
| Irrelevant pseudo-event placebo | Supported as negative control | best irrelevant ranked pseudo-event pseudo_random_hash_rank_matched_a1 wMAE = 2.6274 (gap 0.1252); best irrelevant gated pseudo-event pseudo_random_hash_rank_matched_gated_a1_d0p15 wMAE = 2.5610 (gap 0.0587) | The improvement is not reproduced by arbitrary distribution-matched cohort rankings; the event prior must remain mechanism-aligned. | `outputs\irrelevant_pseudo_event_placebo\irrelevant_pseudo_event_placebo_metrics.csv` |
| Planning-oriented operating point | Supported | balanced profile selects gated_trip_suppression_a1_d0p15; low-cost profile selects global_trip_suppression_a1 | The method exposes an auditable choice between accuracy/calibration and request cost. | `outputs\multi_objective_pareto\preference_operating_points.csv` |

## How to Use in the Paper/PPT

1. Put the DAG in the method or robustness section.
2. Report the leakage audit before discussing LLM gains.
3. Use the random-pressure and placebo checks as guardrails, not as causal proof.
4. State that the global prior is a strong low-cost baseline; the gated LLM adapter is the balanced operating point.
5. Keep retrospective world-knowledge risk as a limitation unless a frozen event-context RAG is implemented.

## Generated Artifacts

- `outputs/causal_guardrails/causal_guardrail_summary.csv`
- `outputs/causal_guardrails/causal_dag.png`
- `outputs/causal_guardrails/causal_guardrail_evidence_report.md`
