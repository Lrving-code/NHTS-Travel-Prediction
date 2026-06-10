# Submission Readiness Audit

Generated from current repository artifacts. This is an evidence audit, not a claim that the project is fully submission-ready.

Overall readiness score: `0.99` / 1.00

## Category Scores

| Category | Score |
|---|---:|
| Baselines and controls | 1.00 |
| Causal and leakage guardrails | 1.00 |
| Core results | 1.00 |
| Literature and paper story | 1.00 |
| Mobility behavior system | 1.00 |
| Presentation readiness | 1.00 |
| Reproducibility and GPU | 0.88 |
| Temporal and external validity | 1.00 |

## Hard Blockers

No FAIL-level blockers were detected from the checked artifacts.

## Partial Items To Fix Before Paper Submission

| Category | Item | Evidence | Recommendation |
|---|---|---|---|
| Reproducibility and GPU | Local open-source LLM prior replication | Protocol and environment audit exist, but status is BLOCKED_TORCH_CPU. | Install CUDA-enabled PyTorch or run the local model in a GPU-ready environment, then regenerate local priors. |

## Full Evidence Matrix

| Category | Item | Status | Score | Evidence |
|---|---|---:|---:|---|
| Core results | Primary trip-count result | PASS | 1.0 | historical wMAE 4.3377 -> primary wMAE 2.5023; primary wBias -0.0230. |
| Core results | Core result artifacts | PASS | 1.0 | Final report, accuracy summary, and CI report exist. |
| Baselines and controls | Stronger tabular baseline | PASS | 1.0 | outputs/strong_baselines/strong_tabular_baseline_metrics.csv |
| Baselines and controls | Transparent count-model baseline | PASS | 1.0 | outputs/count_model_baselines/count_model_baseline_metrics.csv |
| Baselines and controls | Count-model solver diagnostics | PASS | 1.0 | outputs/count_model_baselines/count_model_solver_diagnostics.csv |
| Baselines and controls | Negative-binomial count baseline | PASS | 1.0 | outputs/negative_binomial_baseline/negative_binomial_2022_metrics.csv |
| Baselines and controls | Negative-binomial diagnostics | PASS | 1.0 | outputs/negative_binomial_baseline/negative_binomial_diagnostics.csv |
| Baselines and controls | Negative-binomial report | PASS | 1.0 | outputs/negative_binomial_baseline/negative_binomial_baseline_report.md |
| Baselines and controls | Zero-shot rule tree | PASS | 1.0 | outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv |
| Baselines and controls | Small historical calibration | PASS | 1.0 | outputs/llm_rule_small_data_calibration/method_spectrum_metrics.csv |
| Baselines and controls | Irrelevant pseudo-event placebo | PASS | 1.0 | outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv |
| Baselines and controls | Permutation robustness | PASS | 1.0 | outputs/robustness_checks/permutation_pressure_controls.csv |
| Baselines and controls | Cohort-prior value analysis | PASS | 1.0 | outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv |
| Baselines and controls | Cohort-prior value report | PASS | 1.0 | outputs/cohort_prior_value_analysis/cohort_prior_value_report.md |
| Baselines and controls | Cohort-prior value figure | PASS | 1.0 | outputs/cohort_prior_value_analysis/cohort_prior_value_top_groups.png |
| Baselines and controls | Count-model comparison | PASS | 1.0 | Poisson GLM wMAE 4.3368, wBias +3.6178; primary adapter is 42.30% lower in wMAE. |
| Baselines and controls | Negative-binomial comparison | PASS | 1.0 | NB GLM wMAE 6.1229, wBias -1.1091; primary adapter is 59.13% lower in wMAE, with diagnostics recorded. |
| Baselines and controls | Cohort-prior incremental value | PASS | 1.0 | Primary gated adapter beats same-alpha global prior by 0.0508 wMAE and wins in 77.8% of evaluated subgroup cells. |
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
| Literature and paper story | LaTeX manuscript and verified references | PASS | 1.0 | LaTeX skeleton, core figures, BibTeX, and citation verification log exist. |
| Literature and paper story | Top-venue adversarial audit | PASS | 1.0 | Round-2 top-venue audit states remaining risks and safe claims. |
| Presentation readiness | Main deck plus backup | PASS | 1.0 | 0611 integrated PPT has 35 slides with B1-B10 backup. |
| Presentation readiness | Key defense content in deck | PASS | 1.0 | Integrated deck contains precise title, 2025-2026 literature anchors, method spectrum, selector branch, prior-generation deployment, and cohort-prior defense content. |
| Presentation readiness | Speaker notes | PASS | 1.0 | Integrated speaker notes include talk path and title-wording guardrails. |
| Reproducibility and GPU | Stronger baseline script | PASS | 1.0 | Strong baseline script and metrics exist. |
| Reproducibility and GPU | GPU execution evidence | PASS | 1.0 | label_free_llm_adaptation_metrics.csv records device=cuda. |
| Reproducibility and GPU | Environment manifest | PASS | 1.0 | Environment manifest and freeze file exist under outputs/submission_readiness. |
| Reproducibility and GPU | Local open-source LLM prior replication | PARTIAL | 0.5 | Protocol and environment audit exist, but status is BLOCKED_TORCH_CPU. |

## Interpretation

- Course-project readiness is strong: the core result, baselines, guardrails, deck, Q&A material, and paper draft package are present.
- No FAIL-level blockers remain, but PARTIAL items should be resolved before a serious paper submission.
- The defensible paper claim should remain scoped to label-free event adaptation for survey-based household mobility under a post-pandemic shift.