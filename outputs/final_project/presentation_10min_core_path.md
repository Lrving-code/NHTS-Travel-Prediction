# 10-Minute Core Talk Path

## Goal

The 10-minute defense should not try to explain every experiment. It should make one argument:

> 2022 NHTS is an event-shift target. A routine household model overpredicts post-pandemic travel. LLM-derived event priors provide label-free mechanism adaptation, and the final operating point is selected under planning objectives.

## Core Slides

| Time | Slide | Message |
|---:|---|---|
| 0:00-0:40 | 1 Title | We predict post-pandemic household travel behavior with routine models plus LLM event priors. |
| 0:40-1:30 | 3 Research Gap | 2022 is not ordinary temporal transfer; historical models overpredict. |
| 1:30-2:10 | 5 Data and Target System | The task is household-level behavior: trips, mode shares, purpose shares, and mode-specific volume. |
| 2:10-3:10 | 6 Technical Route | XGBoost learns routine mobility; LLM outputs event priors; adapter applies a no-label correction. |
| 3:10-3:50 | 7 LLM Event Prior | The LLM predicts structured mechanisms, not trip counts. |
| 3:50-4:30 | 8 Evaluation Design | Compare traditional, global event rule, random control, LLM-only, and hybrid. |
| 4:30-5:30 | 9 Main Results | Primary gated adapter reduces weighted MAE by 42.31% and nearly removes weighted bias. |
| 5:30-6:10 | 10 Statistical Validation | Bootstrap CI supports that the gain is not just a point-estimate accident. |
| 6:10-7:10 | 12 Multi-Objective Selection | Main method is selected for balanced accuracy/calibration; low-cost deployment can choose global prior. |
| 7:10-8:00 | 18 LLM Role | Pure LLM pressure captures the downward direction but is less calibrated; XGBoost grounding makes the hybrid adapter stable. |
| 8:00-9:05 | 19 Behavior Outputs | The method extends to mode/transit and mode-specific trip volume; purpose is exploratory. |
| 9:05-10:00 | 23 Conclusion | Summarize contribution, limitations, and next paper-level upgrades. |

## Backup Slides

Use only if asked:

- Slide 4: Related work.
- Slide 11: Error-bias tradeoff.
- Slide 13: Error distribution and tolerance.
- Slide 14: Event heterogeneity.
- Slide 15: Robustness check.
- Slide 16: Subgroup check.
- Slide 17: Error insight.
- Slide 20: Purpose composition.
- Slide 21: LLM scaling.
- Slide 22: Research logic.
- B7: External mechanism validation and BTS compatibility guardrail.

## Lines to Avoid

- Avoid: “LLM predicts 2022 household trips.”
- Avoid: “We prove causal effects.”
- Avoid: “Purpose composition is solved.”
- Avoid: “Cohort-specific LLM ranking is the only reason for the gain.”
- Avoid: “BTS device trips prove external household-level trip-count accuracy.”

## Lines to Use

- “LLM outputs event priors, not target values.”
- “The main result is label-free event adaptation under a post-pandemic distribution shift.”
- “The global event prior is a strong low-cost baseline; the gated LLM adapter is the balanced reporting operating point.”
- “Pure LLM pressure has the right direction but weaker calibration; the hybrid design is why the final bias is near zero.”
- “Causal guardrails and negative controls make the LLM component auditable.”
- “External data supports the event mechanisms; PSRC household microdata provides recovery-transfer evidence, while direct NHTS-style external replication remains future work.”
