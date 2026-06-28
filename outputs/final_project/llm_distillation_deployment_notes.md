# LLM Distillation Deployment Notes

## Design Role

The LLM component is not used as a per-household trip-count predictor. It is used as an event-prior generator:

```text
historical household baseline
    + LLM event-prior distillation
    + fixed no-label adapter
    + optional low-confidence fallback
```

This keeps the quantitative prediction anchored in observed NHTS household behavior while using LLM generalization to represent post-pandemic mechanisms such as remote work, transit avoidance, delivery substitution, and recovery sensitivity.

## Efficiency Logic

- Direct household-level LLM calls scale linearly with the number of households.
- The implemented pipeline aggregates 7,893 households into 1,327 cohorts.
- Batch prompting with batch size 15 reduces the LLM workload to 89 prompts.
- After event priors are generated and validated, inference uses a deterministic adapter, so the main reported experiment does not call the LLM per household.

## Confidence Handling

The main no-label experiment uses fixed rules and a conservative fallback:

- High-confidence cohort priors are used by the event adapter.
- Low-confidence cohort priors fall back to a global event pressure.
- Online LLM fallback is a deployable extension for future systems, but is not mixed into the reported 2022 evaluation.

## Reporting Boundary

The safe claim is:

> LLM knowledge is distilled into structured event priors, and these priors adapt a historical household model under a post-pandemic distribution shift.

The unsafe claim is:

> The LLM directly predicts household trips or learns a 2022 decision tree without calibration.
