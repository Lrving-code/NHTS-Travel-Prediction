# Final Project Report

## Core Claim

This project formulates 2022 NHTS household travel prediction as event-driven temporal adaptation. Historical models learn routine mobility, but the post-pandemic wave contains event mechanisms such as remote work substitution, transit avoidance, online delivery substitution, and uneven recovery. The main contribution is a label-free hybrid adapter: LLM-derived event priors correct a historical routine-mobility predictor, while 2022 trip-count labels are used only for final evaluation.

## Research Questions

- RQ1: How severely do historical household travel models overpredict 2022 post-pandemic trip generation?
- RQ2: Can event priors reduce this bias without using 2022 `CNTTDHH` labels for training or calibration?
- RQ3: Does the LLM replace historical prediction, or is the stronger design a hybrid of routine mobility and event semantics?

## Literature Position

The 2025-2026 literature grounding is summarized in `plan/literature_grounding_2026.md`. The closest directions are event-driven LLM mobility generation (ELLMob, ICLR 2026), LLM-derived causal public-event features for mobility prediction (CausalMob, KDD 2025), zero-shot LLM mobility agents (AgentMove, NAACL 2025), efficient LLM mobility pipelines (ELP-Mob, SIGSPATIAL/GIS 2025), and universal mobility foundation models (UniMob, KDD 2025).

Our gap is survey-based household mobility under a post-pandemic event shift: most recent work targets trajectories, flows, traffic sensors, or public-event time series, while this project uses historical NHTS labels to ground routine household demand and uses LLM priors only for no-label event adaptation.

## Main Result

| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 |
|---|---:|---:|---:|---:|
| Historical mean | 5.5116 | 6.1251 | 4.4995 | -1.1722 |
| Traditional supervised baseline | 4.3377 | 5.3169 | 3.6052 | -0.6367 |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 |
| LLM-only pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 |
| Zero-shot LLM rule tree | 2.6019 | 3.8300 | -0.4072 | 0.1507 |
| Zero-shot pseudo-label tree | 2.6040 | 3.8368 | -0.4072 | 0.1477 |
| LLM rule + 500-history calibration | 2.7723 | 3.8226 | 0.4627 | 0.1538 |

## Temporal Transfer Validation

Before interpreting 2022 as an event-shift target, we checked routine cross-year transfer:

| Check | Weighted MAE | Weighted Bias | R2 |
|---|---:|---:|---:|
| 2001 -> 2009 | 3.5207 | +0.5818 | 0.4413 |
| 2001+2009 -> 2017 | 4.1272 | +1.2230 | 0.2505 |
| 2001+2009+2017 -> 2022 | 4.4062 | +3.7202 | -0.5055 |

The pre-COVID checks have mean absolute weighted bias `0.9024`, while 2022 has absolute weighted bias `3.7202`. This supports the problem framing: 2022 is a stronger post-pandemic event shift rather than an ordinary transfer year.

## External Mechanism Validation

We added an external validation and compatibility audit in `outputs/external_validation/`. The external evidence is deliberately scoped as mechanism-level validation rather than household-level MAE.

| External source | Finding | Relevance |
|---|---|---|
| ACS commuting brief | Worked-from-home commute share remains much higher in 2022 than 2019: `5.7% -> 15.2%`. | Supports `remote_work_substitution` event prior. |
| ACS commuting brief | Public-transportation commute share remains lower in 2022 than 2019: `5.0% -> 3.1%`. | Supports `transit_avoidance` event prior. |
| BTS/UMD daily mobility | BTS trips/person changes only `-1.95%` from 2019 to 2022, while NHTS diary `CNTTDHH` has a much larger 2017-to-2022 shift. | Shows BTS device mobility is not a direct numeric label for NHTS household trips. |
| PSRC household travel survey | 2017+2019->2023 household/day pre/post replication wMAE changes `3.4271 -> 3.2498`; wBias changes `+1.0532 -> +0.2605` using an ACS-derived remote-work suppression factor. | Provides household-level external microdata evidence that event-scale suppression improves transfer on an independent travel survey. |

Interpretation: external data supports the event semantics used by the LLM adapter and now includes a household-level external pre/post replication. The PSRC result is not direct numerical validation of NHTS 2022 household predictions because PSRC is a regional survey with different sampling and diary protocols, but it strengthens the paper claim that fixed event priors can improve label-free transfer under post-pandemic survey shifts. The external adapter is also no-label: it is specified from ACS event context, without target-year PSRC label calibration.

## Improvement Over Historical Predictor

- Primary strict no-label row: `gated_trip_suppression_a1_d0p15`.
- Primary weighted MAE drops from `4.3377` to `2.5023` (42.31% reduction).
- Primary weighted RMSE drops from `5.3169` to `3.6038` (32.22% reduction).
- Primary absolute weighted bias drops by `99.36%`.
- LLM-only pressure improves over naive historical baselines but remains weaker than the hybrid adapter, supporting the design choice that LLMs provide event semantics rather than standalone household predictions.
- Zero-shot LLM rule tree reaches weighted MAE `2.6019`, and the pseudo-label tree distilled from it reaches `2.6040`; both are weaker and more biased than the primary hybrid adapter.
- LLM rule + 500 historical calibration reaches weighted MAE `2.7723`; it is a useful bridge baseline for data-sparse settings, but it reintroduces positive 2022 bias.

## Household-Level Accuracy

- Gated MAE: `2.4738` trips per household.
- Gated RMSE: `3.5872` trips per household.
- Exact rounded hit rate: `17.7%`.
- Within 1 trip: `29.8%`.
- Within 2 trips: `54.5%`.
- Within 3 trips: `71.2%`.

## Statistical Validation

Household bootstrap resampling gives the following 95% confidence intervals:

- Traditional supervised baseline weighted MAE: `[4.2418, 4.4312]`.
- Gated LLM correction weighted MAE: `[2.4292, 2.5808]`.
- Paired weighted-MAE reduction: `[1.7422, 1.9267]`.
- Paired absolute-bias reduction: `[3.3582, 3.6696]`.

This supports the claim that the main trip-count improvement is not a single point-estimate artifact.

## Behavior-System Extension

The project now reports household travel behavior as a multi-output system:

1. Trip generation: household `CNTTDHH`.
2. Mode composition: household mode-share vector from `TRPTRANS`.
3. Mode-specific trip volume: predicted total trips multiplied by predicted mode shares.
4. Purpose composition: household purpose-share vector from `TRIPPURP`.

Mode-specific trip-volume results:

- Traditional count x traditional mode total mode-trip MAE: `4.8467`.
- Gated count x LLM mode total mode-trip MAE: `3.2802`.

Purpose-composition results:

- Traditional XGBoost purpose weighted TV: `0.5661`.
- LLM purpose prior is not the overall best row, but it adds a third behavior dimension and exposes a clear future-work target for purpose-specific event adaptation.

## Efficiency

- Household rows in 2022: `7,893`.
- LLM cohort prompts: `1,327`.
- LLM request reduction: `83.2%`, or about `5.95x` fewer requests than household-level prompting.
- Batch prompting with batch size 15 further compresses `1,327` cohort prompts into `89` batch prompts, a `93.29%` request reduction relative to one-cohort prompts.

## LLM Generalization Role

The LLM should be framed as an event-generalization module, not as a direct predictor. It maps pandemic mechanisms such as remote work, transit avoidance, online delivery substitution, and uneven recovery onto unlabeled household cohorts. A prospective event-context file is included at `plan/prospective_event_context_2022.md` to make this role more auditable and reduce retrospective leakage risk.

## Why Not a Zero-Shot LLM Decision Tree

A decision tree needs labels to learn split thresholds and leaf-level numerical predictions. Without 2022 `CNTTDHH` labels, an LLM-generated tree would be a belief tree or a synthetic-label model rather than a data-fitted 2022 tree. This is why the project uses the LLM as an event-prior generator and keeps numerical prediction grounded in a historical household model trained on real NHTS data.
We now include this as an explicit ablation. A zero-shot LLM-style semantic rule tree reaches weighted MAE `2.6019`, while a pseudo-label tree reaches `2.6040`. The primary hybrid adapter remains better at weighted MAE `2.5023`.

## LLM Rules Plus Small Historical Calibration

We also include the collaborator-proposed bridge route: let an LLM-style rule structure define routine demand leaves, calibrate leaf values with historical samples, and then apply the same 2022 event factor. With 500 historical calibration rows, weighted MAE is `2.7723`; with full-history rule calibration, weighted MAE is `2.7870`. This supports the method spectrum but also shows why routine historical calibration alone can overpredict under a post-pandemic shift.

## Robustness Check

We ran additional robustness checks in `outputs/robustness_checks/`.

- 500-run permutation control for the primary gated rule: actual weighted MAE `2.5023`, random-permutation mean `2.5808`, empirical p-value `0.0020`.
- Same-alpha global pressure remains strong: primary gated weighted MAE `2.5023` vs global-a1 weighted MAE `2.5531`.
- LLM rule + small historical calibration is a coherent bridge baseline but not a replacement: 500-row calibration weighted MAE `2.7723`.
- Irrelevant pseudo-event placebo controls are weaker than the primary method: best ranked pseudo-event weighted MAE `2.6274`, best gated pseudo-event weighted MAE `2.5610`.
- Leakage scan passes for LLM-facing profile/feature files: they exclude `HOUSEID`, `CNTTDHH`, and `WTHHFIN`.

Interpretation for the course report: the dominant contribution is event-level label-free adaptation. Cohort-specific LLM ranking provides measurable incremental signal, but it should not be described as the sole source of improvement.

## Figures

- `metric_comparison`: `figures/metric_comparison.png`
- `bias_r2`: `figures/bias_r2_tradeoff.png`
- `household_accuracy`: `figures/household_tolerance_accuracy.png`
- `pressure_distribution`: `figures/event_pressure_distribution.png`
- `subgroup_gains`: `figures/subgroup_llm_gains.png`
- `workflow`: `figures/method_workflow.png`

## Presentation Framing

Avoid framing the project as a generic feature-only forecasting improvement. The cleaner narrative is: the traditional supervised baseline fails under a rare event; LLMs provide event semantics that can be distilled into a lightweight correction rule.

Do not overstate the mode-composition or purpose-composition extensions. The strongest result remains trip generation under event-driven temporal adaptation; the extensions show a broader behavior system and planning relevance.
