# Pre-COVID Placebo Event-Correction Report

## Scope

This placebo applies the 2022 LLM-derived global trip-suppression prior to a pre-COVID transfer task: train on 2001+2009 and predict 2017. It tests whether pandemic event correction behaves like a generic error fix or a context-specific shock adapter.

- Global 2022 trip-suppression pressure used in placebo: `0.4559`.
- 2017 labels are used only for evaluation.

## Results

| Method | Weighted MAE | Weighted bias | Weighted R2 | Prediction mean |
|---|---:|---:|---:|---:|
| placebo_global_2022_suppression_a0p5 | 3.8181 | -0.8340 | 0.2847 | 6.9663 |
| placebo_global_2022_suppression_a0p25 | 3.8867 | +0.1945 | 0.2996 | 7.9948 |
| placebo_global_2022_suppression_a0p75 | 3.9352 | -1.8625 | 0.2057 | 5.9379 |
| routine_2001_2009_to_2017 | 4.1272 | +1.2230 | 0.2505 | 9.0233 |
| placebo_global_2022_suppression_a1 | 4.2497 | -2.8910 | 0.0627 | 4.9094 |
| placebo_global_2022_suppression_a1p25 | 4.7632 | -3.9194 | -0.1443 | 3.8809 |

## Interpretation

The routine pre-COVID baseline has weighted MAE `4.1272` and weighted bias `+1.2230`. Small global downshifts may compensate for this ordinary positive transfer bias, but stronger pandemic-style suppression should not be treated as a universal correction.

For the paper, this placebo should be reported as a guardrail rather than a proof of causality: event priors need context and strength discipline, and 2022 target labels must remain outside the adaptation step.
