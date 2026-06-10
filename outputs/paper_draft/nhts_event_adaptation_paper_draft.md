# Label-Free Event-Aware LLM Adaptation for Post-Pandemic Household Travel Demand Prediction

## Abstract

Household travel surveys provide calibrated evidence for transportation planning, but their temporal transfer can fail when rare societal shocks change the behavioral meaning of otherwise stable household covariates. We study this problem using the U.S. National Household Travel Survey (NHTS), training routine-mobility predictors on pre-2022 waves and evaluating on the 2022 post-pandemic recovery wave. A historical XGBoost model substantially overpredicts 2022 household trip generation, with weighted MAE `4.3377` and weighted bias `+3.6052`. Transparent Poisson/Tweedie/negative-binomial count-model baselines show the same target-year transfer difficulty. We propose a label-free event-aware adaptation framework in which a large language model (LLM) does not directly predict household trips. Instead, it generates cohort-level event priors for mechanisms such as trip suppression, remote-work substitution, transit avoidance, delivery substitution, and recovery sensitivity. These priors are distilled into a fixed no-label event adapter that corrects the historical routine predictor without using 2022 trip-count labels for training or calibration. The primary gated adapter reduces weighted MAE to `2.5023`, reduces weighted RMSE by `32.22%`, and nearly removes systematic overprediction with weighted bias `-0.0230`. We evaluate the method against historical, trend, global-event, random-prior, LLM-only, zero-shot rule-tree, small-history rule-calibration, transparent count-model, and stronger non-LLM baselines, and add leakage audits, placebo controls, cohort-prior value analysis, bootstrap confidence intervals, equity-aware subgroup analysis, mode/purpose extensions, and external mechanism checks with ACS, BTS, and PSRC data. The results suggest that LLMs are most useful in this setting as constrained event-prior generators rather than standalone numerical predictors.

## 1. Introduction

Household travel demand prediction is usually grounded in historical survey data and classical travel-demand modeling, including discrete-choice foundations for travel behavior analysis. Models trained on prior survey waves can learn routine relationships between household attributes and travel behavior: household size, vehicle ownership, worker count, income, urban form, day of week, and regional context all influence how much a household travels on a diary day. This assumption becomes fragile when the target year is affected by a rare societal event. The 2022 NHTS wave is a post-pandemic recovery wave, after remote work, public-transit avoidance, online shopping, and uneven activity recovery changed daily travel patterns.

This paper asks whether an LLM can help adapt a historical household travel model to a post-pandemic target year without using target-year trip-count labels for training or calibration. The key design choice is to separate routine mobility from event response. A supervised tabular model estimates household-level routine demand from historical NHTS waves, while the LLM supplies structured event priors for how the post-pandemic context may suppress or reshape travel. The LLM is therefore not a direct trip-count regressor. It is a constrained event-generalization module whose outputs are validated and distilled into an auditable correction rule.

This framing is motivated by recent mobility research. ELLMob frames event-driven mobility as a conflict between habitual patterns and event constraints; CausalMob transforms public-event text into LLM-derived causal/intention features for mobility prediction; AgentMove shows that zero-shot LLM mobility agents require systematic task decomposition; AgentMob points to a fast-path plus selective reasoning pattern for efficient evidence-grounded mobility prediction; and UniMob/STFM work reflects a broader move toward generalizable mobility foundation models. Our target differs from these settings: the prediction unit is a household survey record, not a GPS trajectory, road sensor, or flow node, and the target-year labels should be unavailable in the intended prospective setting.

### Contributions

1. We formulate 2022 NHTS household trip generation as event-driven temporal adaptation rather than ordinary cross-year forecasting.
2. We introduce a label-free LLM event-prior adapter that corrects a historical household predictor without using 2022 `CNTTDHH` labels for calibration.
3. We evaluate against a broad method spectrum: historical mean, ordinary XGBoost, trend shift, transparent Poisson/Tweedie/negative-binomial count models, CatBoost GPU, LLM-only pressure, global event prior, random prior, zero-shot LLM rule tree, pseudo-label tree, and LLM rule plus historical calibration.
4. We add robustness and credibility checks: leakage audit, prospective event context, placebo events, permutation controls, cohort-prior value analysis, bootstrap confidence intervals, equity-aware subgroup analysis, and external ACS/BTS/PSRC validation.
5. We extend the behavior system beyond total trip generation to mode composition, mode-specific trip volume, and exploratory purpose composition while keeping trip generation as the main contribution.

## 2. Problem Definition

Let `X` denote harmonized household features and `y` denote household diary-day trip count `CNTTDHH`. Historical NHTS waves are available for training, and the target wave is 2022. The strict setting forbids using 2022 `CNTTDHH` labels for training, calibration, or parameter selection. The labels are used only for final evaluation.

The task is:

```text
Given historical labeled NHTS waves and unlabeled 2022 household profiles,
predict 2022 household travel behavior under post-pandemic event shift.
```

The main target is household trip generation. Additional behavior outputs include household mode-share vector, household purpose-share vector, and derived mode-specific trip volume:

```text
predicted trips by mode = predicted total trips * predicted mode share
```

## 3. Method

### 3.1 Historical Routine Predictor

We train a historical routine-mobility model `f_hist(X)` on pre-2022 NHTS household records. The final ordinary supervised baseline is a CUDA XGBoost model trained on 2001, 2009, and 2017 and evaluated on 2022. This branch captures household heterogeneity but lacks post-pandemic event semantics.

### 3.2 LLM Event-Prior Generation

The LLM receives cohort-level household profiles rather than individual household targets. Cohorts summarize safe household attributes and exclude `HOUSEID`, `CNTTDHH`, survey weights, and target-derived aggregate statistics. The output schema contains numeric event priors:

- `trip_suppression`
- `remote_work`
- `transit_avoidance`
- `delivery_substitution`
- `recovery_sensitivity`
- `confidence`

The 2022 households are compressed from `7,893` rows to `1,327` cohorts, then batch prompting with batch size `15` reduces event-prior generation to `89` prompts. After schema and range validation, inference uses a deterministic adapter rather than per-household online LLM calls.

### 3.3 Fixed No-Label Adapter

The primary prediction is:

```text
y_hat_2022 = f_hist(X) * clip(1 - alpha * s_event, min_factor, 1)
```

where `s_event` is the distilled event pressure and the reported primary rule is `gated_trip_suppression_a1_d0p15`. Low-confidence or weakly differentiated cohort priors fall back to a global event pressure. This keeps the main experiment fixed and label-free.

### 3.4 What the Method Is Not

The method is not pure LLM regression, not an LLM-generated 2022 decision tree, and not a generic mobility foundation model. It is a hybrid structured-data system: historical labels provide the numerical household baseline, and LLM event priors provide semantic adaptation under post-pandemic shift.

## 4. Experimental Setup

### Data

The main dataset is the NHTS public-use household data for 2001, 2009, 2017, and 2022, with `357,553` harmonized household records and 28 common household variables. The target 2022 evaluation set contains `7,893` households. Weighted metrics use the released household final weights, following the documented 2022 NHTS weighting plan.

### Metrics

For trip generation, we report weighted MAE, weighted RMSE, weighted bias, weighted R2, and household-level tolerance accuracy. Weighted bias is central because historical transfer overpredicts total demand. For mode and purpose composition, we report weighted total variation and component MAE.

### Baselines and Ablations

The comparison includes:

- historical mean;
- ordinary historical XGBoost;
- historical trend shift;
- transparent Poisson/Tweedie/negative-binomial count models;
- CatBoost GPU and stronger XGBoost/LightGBM tabular baselines;
- LLM-only pressure;
- global event prior;
- random pressure control;
- zero-shot LLM rule tree and pseudo-label tree;
- LLM rule plus historical calibration;
- primary gated LLM event adapter.

## 5. Results

### 5.1 Main Trip-Generation Result

| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 |
|---|---:|---:|---:|---:|
| Historical mean | 5.5116 | 6.1251 | +4.4995 | -1.1722 |
| Ordinary XGBoost | 4.3377 | 5.3169 | +3.6052 | -0.6367 |
| Poisson GLM | 4.3368 | 5.4383 | +3.6178 | -0.7123 |
| Tweedie GLM (p=1.5) | 4.3761 | 5.5757 | +3.6677 | -0.7999 |
| Negative-binomial GLM (alpha=0.5) | 6.1229 | 15.3423 | -1.1091 | -12.6282 |
| Historical trend shift | 4.1504 | 5.1463 | +3.3485 | -0.5334 |
| LLM-only pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 |
| Zero-shot LLM rule tree | 2.6019 | 3.8300 | -0.4072 | 0.1507 |
| LLM rule + 500-history calibration | 2.7723 | 3.8226 | +0.4627 | 0.1538 |
| Primary gated adapter | 2.5023 | 3.6038 | -0.0230 | 0.2480 |

The ordinary XGBoost baseline overpredicts strongly. Transparent Poisson/Tweedie count-model baselines also overpredict by more than `+3.6` weighted trips, while the negative-binomial GLM reduces positive bias only by becoming unstable and much less accurate. This suggests that the core failure is not only high-capacity tabular overfitting but a broader historical-transfer mismatch under the 2022 event shift. The primary gated adapter reduces weighted MAE by `42.31%` relative to ordinary XGBoost, by `42.30%` relative to the best transparent count model, and by `59.13%` relative to the negative-binomial GLM; it also reduces weighted RMSE by `32.22%` and absolute weighted bias by `99.36%`. The LLM-only pressure baseline is directionally useful but less calibrated, supporting the hypothesis that LLM priors need a historical household baseline.

### 5.2 Statistical Validation

Household bootstrap resampling gives 95% confidence intervals:

- ordinary XGBoost weighted MAE: `[4.2418, 4.4312]`;
- gated adapter weighted MAE: `[2.4292, 2.5808]`;
- paired weighted-MAE reduction: `[1.7422, 1.9267]`;
- paired absolute-bias reduction: `[3.3582, 3.6696]`.

These intervals support that the improvement is not a single point-estimate artifact.

### 5.3 Controls and Robustness

The global event prior is strong, so the correct interpretation is event-level adaptation plus incremental cohort refinement, not "LLM ranking explains everything." Placebo and robustness checks help constrain this claim:

- 500-run permutation control for the primary gated rule: actual weighted MAE `2.5023`, random-permutation mean `2.5808`, empirical p-value `0.0020`.
- Irrelevant pseudo-event controls are weaker: best ranked pseudo-event weighted MAE `2.6274`, best gated pseudo-event weighted MAE `2.5610`.
- Zero-shot LLM rule tree reaches weighted MAE `2.6019`, and a pseudo-label tree distilled from it reaches `2.6040`; both are weaker and more biased than the primary adapter.
- LLM rule plus 500 historical calibration reaches weighted MAE `2.7723`, showing that rule distillation plus small historical calibration is coherent but not the best solution for the 2022 event-shift task.
- CatBoost GPU is the strongest extra non-LLM baseline with weighted MAE `4.2196` and weighted bias `+3.4873`, still far from the primary adapter.
- Poisson GLM reaches weighted MAE `4.3368` and Tweedie GLM reaches `4.3761`; a stable negative-binomial GLM check reaches weighted MAE `6.1229` with alpha `0.5`, while alpha `1.0` is solver-infeasible during historical validation. These are grounded in standard count-data regression practice, and solver diagnostics are recorded so they should be read as transparent count-model baselines rather than exhaustively optimized count-model state of the art.
- Cohort-prior value analysis shows that the primary gated adapter is `0.0508` weighted-MAE lower than a same-alpha global prior and `0.0201` lower than the reported low-cost global prior; it beats same-alpha global prior in `77.8%` of evaluated subgroup cells while using cohort-specific pressure for only `17.5%` of survey-weighted households.

### 5.4 Behavior-System Outputs

The method extends beyond total trips:

- transit-share weighted MAE improves from `0.0325` to `0.0269`;
- total mode-trip MAE improves from `4.8467` to `3.2802` when combining gated trip count with LLM-adjusted mode shares;
- purpose composition is implemented as an exploratory third dimension, but the LLM prior is not the overall best purpose row.

These results support a broader household behavior-system framing, while trip generation remains the main contribution.

### 5.5 External and Temporal Validation

Temporal transfer checks show that 2022 is more difficult than pre-COVID transfer: pre-COVID mean absolute weighted bias is `0.9024`, while a diagnostic 2022 transfer check has absolute weighted bias `3.7202`. External evidence supports the event mechanisms:

- ACS commuting statistics show worked-from-home share `5.7% -> 15.2%` from 2019 to 2022.
- ACS public-transportation commute share changes `5.0% -> 3.1%`.
- BTS/UMD daily mobility is retained as a guardrail because device trip counts should not be treated as direct NHTS household labels.
- PSRC household survey replication shows a pre/post improvement from weighted MAE `3.4271` to `3.2498` and weighted bias `+1.0532` to `+0.2605` using an ACS-derived remote-work suppression factor without target-year PSRC label calibration.

The PSRC result is an external regional replication of the event-adaptation principle, not direct numerical validation of NHTS 2022.

## 6. Discussion

The main finding is role separation. Historical tabular models supply calibrated household heterogeneity but fail under post-pandemic event shift. LLM priors supply event semantics but are under-calibrated as standalone numerical predictors. The hybrid adapter works because it constrains the LLM to a narrow semantic role and leaves numerical scale to historical survey data.

The strongest alternative explanation is that a global post-pandemic downscaling scalar captures most of the benefit. The experiments partly confirm this: the global event prior is strong. The cohort-prior value analysis refines the interpretation rather than overturning it. The final gated adapter uses cohort-specific pressure selectively and provides small but consistent MAE/calibration gains over global controls, especially in lower-income, zero-vehicle, and no-worker subgroup cells. The contribution should therefore be framed carefully as label-free event adaptation with selective auditable cohort refinement, not as evidence that rich LLM reasoning alone dominates. Placebo controls, zero-shot rule-tree baselines, small-calibration baselines, and subgroup checks provide evidence that the event prior is meaningful, but additional regional and shock replications would strengthen the paper.

## 7. Limitations

1. The main NHTS evaluation focuses on one shock year, 2022.
2. The LLM may contain broad post-pandemic mobility knowledge from pretraining; the prospective event-context file reduces but does not fully eliminate retrospective knowledge concerns.
3. The global event prior is strong, so cohort-specific LLM ranking should be described as incremental.
4. Mode and purpose extensions are useful for planning relevance but are not yet full mode-choice or purpose-choice models.
5. PSRC is a regional external survey with different sampling and diary protocols, so it validates transfer of the principle rather than exact NHTS numerical accuracy.
6. The current count-model baselines include Poisson, Tweedie, and a stable negative-binomial GLM check with solver diagnostics recorded; a submission version should still add zero-inflated or more carefully regularized count variants if time permits.
7. A local/open-source LLM prior-replication protocol and environment audit are implemented, but the current Python environment has CPU-only PyTorch despite an available RTX 4090; therefore, local LLM replication should not be presented as completed evidence until CUDA-enabled generation outputs exist.

## 8. Conclusion

This project shows that post-pandemic household travel prediction can be formulated as label-free event adaptation. A historical household model alone overpredicts 2022 travel, while a pure LLM-style predictor is directionally useful but under-calibrated. Distilling LLM event semantics into a fixed adapter substantially reduces trip-count error and nearly removes systematic bias without using 2022 trip-count labels for calibration. The result suggests a practical path for using LLMs in survey-based mobility modeling: constrain them to produce auditable event priors, combine them with historically calibrated models, and evaluate them with causal guardrails, robust baselines, and planning-oriented multi-objective metrics.

## Verified References To Cite

- ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework. ICLR 2026 Poster. https://openreview.net/forum?id=MPYsaBgZIT
- CausalMob: Causal Human Mobility Prediction with LLMs-derived Human Intentions toward Public Events. KDD 2025. https://doi.org/10.1145/3690624.3709231 and https://arxiv.org/abs/2412.02155
- AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction. NAACL 2025. https://aclanthology.org/2025.naacl-long.61/
- AgentMob: Towards Efficient and Evidence-grounded Mobility Prediction with LLM-Driven Agent. 2026 preprint. https://arxiv.org/abs/2606.05130
- ELP-Mob: Building Efficient LLM Pipeline for Human Mobility Prediction. SIGSPATIAL/GIS 2025. https://github.com/chwang0721/ELP-Mob
- UniMob: A Universal Model for Human Mobility Prediction. KDD 2025 / arXiv. https://arxiv.org/abs/2412.15294
- Foundation Models for Spatio-Temporal Data Science: A Tutorial and Survey. 2025. https://arxiv.org/abs/2503.13502
- McFadden, D. Conditional Logit Analysis of Qualitative Choice Behavior. 1974.
- Ben-Akiva and Lerman. Discrete Choice Analysis: Theory and Application to Travel Demand. MIT Press, 1985.
- Cameron and Trivedi. Regression Analysis of Count Data. Cambridge University Press, 2nd edition, 2013.
- 2022 NHTS Address-Based Sample Weighting Plan. FHWA/Ipsos, 2022.
