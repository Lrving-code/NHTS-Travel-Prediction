# Presentation Speaker Notes

## Core Message
We predict 2022 household mobility under post-pandemic distribution shift. The main result is a label-free LLM event-prior correction for household trip counts.

## One-Minute Version
Historical prediction overestimates 2022 trips. The primary fixed no-label gated rule reduces weighted MAE from `4.3377` to `2.5023` and moves weighted bias to `-0.0230`. The best-MAE sensitivity row reaches `2.4820`. The LLM is not used as a direct trip-count predictor; it provides pandemic-event semantics that modify a historical routine-mobility predictor.

## Metric Language
- Weighted MAE/RMSE are survey-weighted trip-count errors.
- Weighted bias tells whether we systematically overpredict or underpredict.
- Within-k accuracy is used only to make regression error intuitive.
- Weighted total variation is the whole mode-share vector error.
- Transit-share weighted MAE is the public-transit component error.

## Mode Extension
Mode composition is supporting evidence. XGBoost + LLM reduces transit-share weighted MAE from `0.0325` to `0.0269`, but the overall mode-composition improvement is small.

## Questions To Expect
- Why not compare with your classmate's accuracy? Because that is trip-level classification, while our main task is household-level regression.
- Is this pure LLM? No. Pure LLM-like correction is weaker than XGBoost + LLM.
- Did 2022 labels enter training? Not in the main label-free setting; 2022 targets are used for evaluation.