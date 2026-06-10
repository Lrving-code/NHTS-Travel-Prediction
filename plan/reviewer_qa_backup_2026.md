# Reviewer and Instructor Q&A Backup

Last updated: 2026-06-10

Use this document to answer discussion questions after the 10-minute presentation. The main talk should stay on the 23-slide path; these answers are backup material.

## Q1. Why not let the LLM directly build a 2022 decision tree?

Short answer: we implemented this baseline, and it is useful but less calibrated.

Evidence:

- Zero-shot LLM rule tree weighted MAE: `2.6019`, weighted bias: `-0.4072`.
- Pseudo-label tree distilled from that rule tree weighted MAE: `2.6040`.
- Primary hybrid gated adapter weighted MAE: `2.5023`, weighted bias: `-0.0230`.

Response:

Without 2022 trip-count labels, a decision tree cannot learn data-fitted split thresholds or leaf predictions for the target year. The LLM can provide a qualitative belief tree, but the numerical calibration is weaker than a historical household baseline plus an event adapter.

## Q2. Did the method use 2022 labels or retrospective leakage?

Short answer: 2022 `CNTTDHH` labels are used only for final evaluation.

Evidence:

- The main method is fixed before evaluation: `gated_trip_suppression_a1_d0p15`.
- LLM inputs exclude `HOUSEID`, `CNTTDHH`, and `WTHHFIN`.
- Leakage audit passes for LLM-facing profile and feature files.
- `plan/prospective_event_context_2022.md` separates event-context assumptions from observed 2022 outcomes.

Response:

The LLM sees cohort-level household profiles and generic post-pandemic event context, not target labels or sample weights. We also report zero-shot rule-tree, small-data calibration, placebo, and leakage checks to make the no-label claim auditable.

## Q3. If the global event prior is already strong, what does the LLM add?

Short answer: the dominant signal is event-level suppression; the cohort-specific LLM ranking is an incremental refinement.

Evidence:

- Global event prior weighted MAE: `2.5223`.
- Primary gated adapter weighted MAE: `2.5023`.
- Primary bias: `-0.0230`, compared with global event prior bias `-0.7477`.
- Same-alpha decomposition: primary gated is `0.0508` weighted-MAE lower than `global_trip_suppression_a1`, and it beats same-alpha global prior in `77.8%` of evaluated subgroup cells.
- The gate uses cohort-specific pressure for `17.5%` of survey-weighted households; the rest falls back to the global event prior.
- Irrelevant pseudo-event controls are weaker: best ranked pseudo-event weighted MAE `2.6274`; best gated pseudo-event weighted MAE `2.5610`.

Response:

We should not overclaim that cohort ranking explains all gains. The clean claim is that LLM event semantics give a label-free event correction, while the gated cohort refinement selectively improves calibration and subgroup robustness where cohort priors differ meaningfully from the global event pressure.

## Q4. How is this different from 2025-2026 mobility foundation-model work?

Short answer: those works mostly target trajectories, flows, traffic sensors, or public-event time series; this project targets household survey prediction under post-pandemic shift.

Evidence:

- Literature anchor: `plan/literature_grounding_2026.md`.
- ELLMob and CausalMob motivate event-aware LLM mobility modeling.
- AgentMove motivates zero-shot LLM mobility agents; ELP-Mob motivates efficient LLM mobility pipelines; AgentMob motivates a fast-path plus selective LLM/tool-reasoning design for ambiguous mobility cases.
- UniMob and STFM surveys motivate foundation-model generalization.

Response:

We borrow the event-generalization idea, but do not train a universal mobility foundation model. The unit is an NHTS household survey record, and the LLM is used only as a structured event-prior generator.

## Q5. Are mode and purpose choice solved?

Short answer: no. Trip generation is the main contribution; mode and purpose are behavior-system extensions.

Evidence:

- Transit-share weighted MAE improves from `0.0325` to `0.0269`.
- Mode-specific trip-volume MAE improves from `4.8467` to `3.2802`.
- Purpose composition is exploratory; the LLM purpose prior is not the overall best row.

Response:

The project predicts a broader household behavior system, but the strongest and most defensible result is event-driven trip generation. Mode/purpose outputs support planning relevance and define future work.

## Q6. What do the metrics mean in plain language?

Short answer: weighted MAE measures average trip-count error under survey weights; weighted bias measures systematic overprediction or underprediction.

Evidence:

- Historical XGBoost weighted MAE: `4.3377`, weighted bias `+3.6052`.
- Primary gated weighted MAE: `2.5023`, weighted bias `-0.0230`.
- Within 2 trips: `54.5%`; within 3 trips: `71.2%`.

Response:

MAE tells us how far predictions are from household trip counts. Bias tells us whether the model systematically overestimates total travel demand. The primary gain is not just lower error; it nearly removes the systematic 2022 overprediction.
