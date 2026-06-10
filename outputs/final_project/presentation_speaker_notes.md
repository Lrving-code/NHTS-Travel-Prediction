# Presentation Speaker Notes

## Core Message
We formulate 2022 NHTS household travel prediction as event-driven temporal adaptation. Historical models capture routine household mobility, while LLM event priors encode pandemic mechanisms that are weakly represented in household covariates: remote work, transit avoidance, delivery substitution, and uneven recovery.

The main method is not pure LLM prediction. It is a hybrid gated adapter: historical household baseline times an event correction factor. The LLM outputs structured event pressure, not household trip-count labels.

## One-Minute Version
Historical prediction overestimates 2022 trips. The primary fixed no-label gated rule reduces weighted MAE from `4.3377` to `2.5023` and moves weighted bias to `-0.0230`. The LLM-only pressure baseline reaches `2.7175`. The LLM is not used as a direct trip-count predictor; it provides pandemic-event semantics that modify a historical routine-mobility predictor.

## 10-Minute Talk Path

The first 23 slides are the main talk path. Slides B1-B6 after END are backup Q&A slides and should only be used during discussion.

Suggested timing:

- Background and research question: 1.5 minutes.
- Data, technical route, and LLM event priors: 2 minutes.
- Method comparison and main results: 2.5 minutes.
- Robustness, heterogeneity, multi-objective analysis, and mode/purpose extension: 2 minutes.
- Conclusion and limitations: 2 minutes.

If time is tight, keep the problem, route, method comparison, main results, robustness framing, and conclusion; skim the error-distribution, subgroup, and mode/purpose details.

## Method Comparison Logic

- Ordinary historical prediction has household grounding but lacks event semantics: wMAE `4.3377`, wBias `3.6052`.
- Pure LLM pressure has the right downward direction but weaker calibration: wMAE `2.7175`, wBias `-0.3732`.
- Zero-shot LLM rule tree is now an explicit ablation: wMAE `2.6019`, wBias `-0.4072`. It is useful for cold-start discussion but weaker than the hybrid adapter.
- LLM rule + 500 historical calibration reaches wMAE `2.7723`, wBias `0.4627`. It is the bridge baseline for the collaborator's route, but it still overpredicts 2022.
- Global event prior is a strong low-cost control: wMAE `2.5223`, wBias `-0.7477`. Do not overclaim that cohort-specific LLM ranking explains the whole gain.
- Primary hybrid gated adapter is the balanced operating point: wMAE `2.5023`, wBias `-0.0230`, wR2 `0.2480`.
- Stronger non-LLM tabular baselines still overpredict. Best extra baseline `catboost_gpu` has wMAE `4.2196` and wBias `3.4873`.

## Metric Language
- Weighted MAE/RMSE are survey-weighted trip-count errors.
- Weighted bias tells whether we systematically overpredict or underpredict.
- Within-k accuracy is used only to make regression error intuitive.
- Weighted total variation is the whole mode-share vector error.
- Transit-share weighted MAE is the public-transit component error.

## Mode Extension
Mode composition is the second household-level output. XGBoost + LLM reduces transit-share weighted MAE from `0.0325` to `0.0269`, while the overall mode-composition improvement is small.

Mode-specific trip volume is the derived planning output: predicted total trips multiplied by predicted mode shares. It is more interpretable for planning than a standalone mode-share number.

## Questions To Expect
- Why use two metric families? Trip generation is count regression, while mode composition is a share-vector prediction problem.
- Is this pure LLM? No. Pure LLM-like correction is weaker than XGBoost + LLM.
- Why not let the LLM build a zero-shot 2022 decision tree? We now include that ablation. The zero-shot rule tree reaches wMAE `2.6019`, still weaker and more biased than the primary hybrid rule. Without labels, the output is closer to an LLM belief tree than a data-fitted decision tree.
- Did 2022 labels enter training? Not in the main label-free setting; 2022 targets are used for evaluation.
- What should we not claim? Do not claim causal effects or direct LLM trip-count prediction; claim causal guardrails and event-prior adaptation.

## Backup Slide Map

- B1: why not direct LLM rule-tree prediction.
- B2: no 2022-label training and leakage guardrails.
- B3: what the LLM adds beyond a global event prior.
- B4: relation to 2025-2026 LLM mobility and foundation-model work.
- B5: limits of mode and purpose outputs.
- B6: plain-language interpretation of weighted MAE, bias, and within-k accuracy.
