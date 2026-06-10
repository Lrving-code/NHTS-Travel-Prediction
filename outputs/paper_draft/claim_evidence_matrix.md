# Claim-Evidence Matrix

This file constrains paper and presentation claims to evidence that exists in the repository.

## Core Claims

| Claim | Evidence | Supported wording | Unsafe wording |
|---|---|---|---|
| 2022 NHTS is an event-shift target, not an ordinary cross-year transfer target. | `outputs/temporal_transfer_validation/temporal_transfer_validation_report.md`; `outputs/final_project/final_project_report.md` | "2022 shows substantially larger positive transfer bias than pre-COVID transfer checks." | "We prove COVID causally reduced trips for every household." |
| Historical supervised and count-model baselines struggle on 2022 household trips. | `outputs/final_project/final_metrics_summary.csv`; `outputs/strong_baselines/strong_tabular_baseline_metrics.csv`; `outputs/count_model_baselines/count_model_baseline_metrics.csv`; `outputs/negative_binomial_baseline/negative_binomial_2022_metrics.csv`; `outputs/negative_binomial_baseline/negative_binomial_diagnostics.csv` | "Ordinary XGBoost has weighted bias `+3.6052`; CatBoost GPU has weighted bias `+3.4873`; Poisson GLM has weighted bias `+3.6178`; the negative-binomial GLM has weighted MAE `6.1229` with solver diagnostics recorded." | "Traditional methods cannot model travel demand." |
| The primary method improves trip-count prediction without target-year label calibration. | `outputs/final_project/final_metrics_summary.csv`; `outputs/leakage_audit/llm_input_leakage_audit_report.md`; `plan/prospective_event_context_2022.md` | "The fixed gated adapter reduces weighted MAE from `4.3377` to `2.5023` and moves weighted bias to `-0.0230`." | "The LLM predicts 2022 household trips accurately by itself." |
| Pure LLM-style prediction is useful but under-calibrated. | `outputs/final_project/final_metrics_summary.csv`; `outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv` | "LLM-only pressure and zero-shot rule tree improve over historical baselines but remain weaker than the hybrid adapter." | "LLM reasoning alone is superior to historical household data." |
| Most gain is event-level correction; cohort ranking is incremental. | `outputs/final_project/final_metrics_summary.csv`; `outputs/robustness_checks/robustness_check_report.md`; `outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv`; `outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv` | "Global event prior is strong; gated cohort refinement improves calibration and adds selective subgroup value." | "Cohort-specific LLM ranking explains all of the improvement." |
| The method is robust to several negative controls. | `outputs/robustness_checks/permutation_pressure_controls.csv`; `outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv`; `outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md` | "Permutation and irrelevant pseudo-event controls are weaker than the primary event adapter." | "All possible placebo and confounding explanations are eliminated." |
| External evidence supports event mechanisms but does not directly validate NHTS 2022 numeric predictions. | `outputs/external_validation/external_validation_and_compatibility_report.md`; `outputs/external_validation/psrc_household_external_validation_report.md`; `outputs/final_project/project_quality_assessment.md` | "ACS supports remote-work and transit-avoidance priors; PSRC provides a regional household-survey replication of the adaptation principle." | "PSRC proves the NHTS 2022 predictions are externally correct." |
| The behavior system extends beyond trip counts, but trip generation is the main result. | `outputs/mode_composition_extension/mode_composition_metrics.csv`; `outputs/mode_composition_extension/mode_specific_trip_count_metrics.csv`; `outputs/purpose_composition_extension/purpose_composition_metrics.csv` | "Transit-share and mode-trip volume improve; purpose composition is exploratory." | "The project solves full mode choice and purpose choice." |
| The LLM design is efficient and auditable. | `outputs/final_project/llm_distillation_deployment_notes.md`; `outputs/llm_event_features/`; `outputs/final_project/NHTS_Travel_Behavior_0611_Integrated_Presentation.pptx` | "7,893 households are aggregated to 1,327 cohorts and processed in 89 batch prompts; inference uses a deterministic adapter." | "The system asks the LLM for every household prediction." |
| Local/open-source LLM replication is prepared but not yet completed. | `src/run_local_llm_prior_replication.py`; `outputs/local_llm_prior_replication/local_llm_environment_audit.md` | "A local LLM replication protocol and environment audit are implemented; current CUDA generation is blocked by CPU-only PyTorch." | "We have completed an open-source GPU LLM replication." |

## Numeric Claims To Reuse

| Quantity | Value | Evidence |
|---|---:|---|
| Ordinary XGBoost weighted MAE | `4.3377` | `outputs/final_project/final_metrics_summary.csv` |
| Ordinary XGBoost weighted bias | `+3.6052` | `outputs/final_project/final_metrics_summary.csv` |
| Primary gated weighted MAE | `2.5023` | `outputs/final_project/final_metrics_summary.csv` |
| Primary gated weighted RMSE | `3.6038` | `outputs/final_project/final_metrics_summary.csv` |
| Primary gated weighted bias | `-0.0230` | `outputs/final_project/final_metrics_summary.csv` |
| Primary gated weighted R2 | `0.2480` | `outputs/final_project/final_metrics_summary.csv` |
| Weighted MAE reduction | `42.31%` | `outputs/final_project/final_metrics_summary.csv` |
| Absolute weighted-bias reduction | `99.36%` | `outputs/final_project/final_metrics_summary.csv` |
| LLM-only pressure weighted MAE | `2.7175` | `outputs/final_project/final_metrics_summary.csv` |
| Global event prior weighted MAE | `2.5223` | `outputs/final_project/final_metrics_summary.csv` |
| Zero-shot LLM rule tree weighted MAE | `2.6019` | `outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv` |
| LLM rule + 500 historical calibration weighted MAE | `2.7723` | `outputs/llm_rule_small_data_calibration/method_spectrum_metrics.csv` |
| CatBoost GPU weighted MAE | `4.2196` | `outputs/strong_baselines/strong_tabular_baseline_metrics.csv` |
| Poisson GLM weighted MAE | `4.3368` | `outputs/count_model_baselines/count_model_baseline_metrics.csv` |
| Poisson GLM weighted bias | `+3.6178` | `outputs/count_model_baselines/count_model_baseline_metrics.csv` |
| Tweedie GLM weighted MAE | `4.3761` | `outputs/count_model_baselines/count_model_baseline_metrics.csv` |
| Negative-binomial GLM weighted MAE | `6.1229` | `outputs/negative_binomial_baseline/negative_binomial_2022_metrics.csv` |
| Negative-binomial GLM weighted bias | `-1.1091` | `outputs/negative_binomial_baseline/negative_binomial_2022_metrics.csv` |
| Negative-binomial GLM selected alpha | `0.5` | `outputs/negative_binomial_baseline/negative_binomial_diagnostics.csv` |
| Primary gated MAE reduction vs best count model | `42.30%` | `outputs/count_model_baselines/count_model_baseline_report.md` |
| Primary gated MAE reduction vs negative-binomial GLM | `59.13%` | `outputs/negative_binomial_baseline/negative_binomial_baseline_report.md` |
| Primary gated vs same-alpha global weighted MAE delta | `0.0508` | `outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv` |
| Primary gated vs reported global weighted MAE delta | `0.0201` | `outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv` |
| Share of subgroup cells where primary beats same-alpha global prior | `77.8%` | `outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv` |
| Survey-weighted share where the gate uses cohort-specific prior | `17.5%` | `outputs/cohort_prior_value_analysis/cohort_prior_value_summary.csv` |
| Transit-share weighted MAE improvement | `0.0325 -> 0.0269` | `outputs/mode_composition_extension/mode_composition_metrics.csv` |
| Mode-trip volume MAE improvement | `4.8467 -> 3.2802` | `outputs/mode_composition_extension/mode_specific_trip_count_metrics.csv` |
| PSRC external weighted MAE improvement | `3.4271 -> 3.2498` | `outputs/external_validation/psrc_household_external_validation_metrics.csv` |
| PSRC external weighted bias improvement | `+1.0532 -> +0.2605` | `outputs/external_validation/psrc_household_external_validation_metrics.csv` |

## Paper-Safe Contribution Wording

Use:

> We propose a label-free event-adaptation framework for survey-based household travel demand. The LLM is constrained to generate event priors, while historical NHTS labels provide the household-level numerical baseline.

Avoid:

> We build a general mobility foundation model, prove COVID causal effects, or show that LLMs directly outperform transportation models.

## Reviewer-Sensitive Boundaries

- The global event prior is a strong baseline. Do not hide it.
- Cohort-specific LLM priors should be framed as selective refinement over a strong global event correction, not as the sole source of improvement.
- PSRC is external household microdata, but it is regional and not direct NHTS 2022 numeric validation.
- Purpose composition is implemented but exploratory.
- The primary adapter is fixed and no-label; any 2022 calibration experiment is an upper-bound or bridge baseline, not the main method.
- Poisson/Tweedie/negative-binomial count models are transparent reviewer-facing baselines with solver diagnostics recorded; do not present them as fully optimized count-model state of the art.
- Local/open-source LLM replication is currently an implemented protocol plus environment audit, not a completed evidence row.
- LLM pretraining may contain post-pandemic knowledge; the prospective event-context file and leakage audit reduce target-label leakage, but cannot make a pure historical-information claim about the LLM's pretraining corpus.
