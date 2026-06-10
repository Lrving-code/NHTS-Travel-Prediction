# Submission Readiness Audit

Generated from current repository artifacts. This is an evidence audit, not a claim that the project is fully submission-ready.

Overall readiness score: `1.00` / 1.00

## Category Scores

| Category | Score |
|---|---:|
| Baselines and controls | 1.00 |
| Causal and leakage guardrails | 1.00 |
| Core results | 1.00 |
| Literature and paper story | 1.00 |
| Mobility behavior system | 1.00 |
| Presentation readiness | 1.00 |
| Reproducibility and GPU | 1.00 |
| Temporal and external validity | 1.00 |

## Hard Blockers

No FAIL-level blockers were detected from the checked artifacts.

## Partial Items To Fix Before Paper Submission

No PARTIAL items were detected.

## Full Evidence Matrix

| Category | Item | Status | Score | Evidence |
|---|---|---:|---:|---|
| Core results | Primary trip-count result | PASS | 1.0 | historical wMAE 4.3377 -> primary wMAE 2.5023; primary wBias -0.0230. |
| Core results | Core result artifacts | PASS | 1.0 | Final report, accuracy summary, and CI report exist. |
| Baselines and controls | Stronger tabular baseline | PASS | 1.0 | outputs/strong_baselines/strong_tabular_baseline_metrics.csv |
| Baselines and controls | Zero-shot rule tree | PASS | 1.0 | outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv |
| Baselines and controls | Small historical calibration | PASS | 1.0 | outputs/llm_rule_small_data_calibration/method_spectrum_metrics.csv |
| Baselines and controls | Irrelevant pseudo-event placebo | PASS | 1.0 | outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv |
| Baselines and controls | Permutation robustness | PASS | 1.0 | outputs/robustness_checks/permutation_pressure_controls.csv |
| Causal and leakage guardrails | Leakage audit | PASS | 1.0 | outputs/leakage_audit/llm_input_leakage_audit_report.md |
| Causal and leakage guardrails | Causal evidence pack | PASS | 1.0 | outputs/causal_guardrails/causal_guardrail_evidence_report.md |
| Causal and leakage guardrails | Prospective event context | PASS | 1.0 | plan/prospective_event_context_2022.md |
| Causal and leakage guardrails | Pre-COVID placebo | PASS | 1.0 | outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md |
| Causal and leakage guardrails | No-target-label framing | PASS | 1.0 | Final report states target-year labels are evaluation-only. |
| Mobility behavior system | Mode composition | PASS | 1.0 | outputs/mode_composition_extension/mode_composition_metrics.csv |
| Mobility behavior system | Mode-specific trips | PASS | 1.0 | outputs/mode_composition_extension/mode_specific_trip_count_metrics.csv |
| Mobility behavior system | Purpose composition | PASS | 1.0 | outputs/purpose_composition_extension/purpose_composition_metrics.csv |
| Mobility behavior system | Equity-aware evaluation | PASS | 1.0 | outputs/equity_aware_evaluation/subgroup_equity_metrics.csv |
| Mobility behavior system | Multi-objective Pareto | PASS | 1.0 | outputs/multi_objective_pareto/preference_operating_points.csv |
| Temporal and external validity | Temporal transfer validation | PASS | 1.0 | Pre-COVID and 2022 transfer report exists. |
| Temporal and external validity | Pre-COVID placebo validation | PASS | 1.0 | Pre-COVID event-correction placebo report exists. |
| Temporal and external validity | External mechanism validation beyond NHTS | PASS | 1.0 | ACS commute-mode mechanism validation and BTS trip-count compatibility guardrail exist. |
| Temporal and external validity | Household-level external microdata validation | PASS | 1.0 | PSRC 2017+2019->2023 household microdata: wMAE 3.4271->3.2498; wBias +1.0532->+0.2605. |
| Temporal and external validity | External validation scope statement | PASS | 1.0 | PSRC report states that the regional survey is external replication of the event-adaptation principle, not direct NHTS numerical validation. |
| Literature and paper story | 2025-2026 literature grounding | PASS | 1.0 | Literature grounding note contains current anchors. |
| Literature and paper story | Single paper spine | PASS | 1.0 | paper_logic_chain.md states the event-adaptation spine. |
| Literature and paper story | Reviewer Q&A backup | PASS | 1.0 | Reviewer Q&A backup document exists. |
| Literature and paper story | Paper draft and claim ledger | PASS | 1.0 | Paper draft and claim-evidence matrix exist. |
| Presentation readiness | Main deck plus backup | PASS | 1.0 | Template PPT has 32 slides with B1-B8 backup. |
| Presentation readiness | Key defense content in deck | PASS | 1.0 | Deck contains literature anchors, method spectrum, and distillation deployment content. |
| Presentation readiness | Speaker notes | PASS | 1.0 | Chinese speaker notes include talk path and backup map. |
| Reproducibility and GPU | Stronger baseline script | PASS | 1.0 | Strong baseline script and metrics exist. |
| Reproducibility and GPU | GPU execution evidence | PASS | 1.0 | label_free_llm_adaptation_metrics.csv records device=cuda. |
| Reproducibility and GPU | Environment manifest | PASS | 1.0 | Environment manifest and freeze file exist under outputs/submission_readiness. |

## Interpretation

- Course-project readiness is strong: the core result, baselines, guardrails, deck, Q&A material, and paper draft package are present.
- No artifact-level FAIL or PARTIAL items remain in this audit; remaining work is LaTeX formatting, advisor feedback, and optional additional replications.
- The defensible paper claim should remain scoped to label-free event adaptation for survey-based household mobility under a post-pandemic shift.