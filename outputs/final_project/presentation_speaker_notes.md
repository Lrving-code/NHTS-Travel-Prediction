# Presentation Speaker Notes

## Core Message
We formulate 2022 NHTS household travel prediction as event-driven temporal adaptation. The paper story is that a traditional supervised baseline captures routine mobility, while LLM event priors encode pandemic mechanisms that are weakly represented in household covariates.

## One-Minute Version
The traditional supervised baseline overestimates 2022 trips. The primary fixed no-label gated rule reduces weighted MAE from `4.3377` to `2.5023` and moves weighted bias to `-0.0230`. The LLM-only pressure baseline reaches `2.7175`, while the best-MAE hybrid sensitivity row reaches `2.4820`. The LLM is not used as a direct trip-count predictor; it provides pandemic-event semantics that modify a routine predictor trained on pre-2022 NHTS data.

## Metric Language
- Weighted MAE/RMSE are survey-weighted trip-count errors.
- Weighted bias tells whether we systematically overpredict or underpredict.
- Within-k accuracy is used only to make regression error intuitive.
- Weighted total variation is the whole mode-share vector error.
- Transit-share weighted MAE is the public-transit component error.

## Mode Extension
Mode composition is the second household-level output. XGBoost + LLM reduces transit-share weighted MAE from `0.0325` to `0.0269`, while the overall mode-composition improvement is small.

## Questions To Expect
- Why use two metric families? Trip generation is count regression, while mode composition is a share-vector prediction problem.
- Is this pure LLM? No. Pure LLM-like correction is weaker than XGBoost + LLM.
- Did 2022 labels enter training? Not in the main label-free setting; 2022 targets are used for evaluation.
