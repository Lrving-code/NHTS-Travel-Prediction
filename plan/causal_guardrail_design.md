# Causal Guardrail Design

## Purpose

This project should use causal language only as a guardrail, not as a causal-effect claim.

Allowed framing:

> LLM priors are structured mechanism proxies for post-pandemic event response.

Not allowed framing:

> LLM priors identify the causal effect of COVID-19 on household travel.

## Causal DAG

```mermaid
flowchart LR
    H[Household attributes<br/>income, vehicles, workers, urban form] --> R[Routine travel demand]
    R --> Y[Observed 2022 household travel behavior]

    E[Societal event context<br/>COVID recovery, restrictions, telework, transit perception] --> M[Event response mechanisms]
    H --> M
    M --> Y

    H --> P[Cohort profile]
    E --> L[LLM event-prior generator]
    P --> L
    L --> S[Structured event priors<br/>trip suppression, remote work, transit avoidance]
    S --> A[No-label adapter]
    R --> A
    A --> Yhat[Predicted 2022 behavior]

    Y -. evaluation only .-> Eval[Final evaluation]
    Yhat --> Eval
```

## Interpretation

- `H -> R`: traditional supervised learning captures stable household heterogeneity.
- `E -> M -> Y`: the pandemic recovery changes the behavioral mechanism.
- `H -> M`: event response is heterogeneous across households.
- `E, P -> L -> S`: the LLM converts external event context and household cohort profiles into structured event priors.
- `S -> A`: the adapter uses priors through pre-declared correction rules.
- `Y` should not enter `L` or `A` during training/calibration.

## Negative Controls

Already available:

- Random pressure assignment.
- Global pressure baseline.
- Temporal transfer validation.
- Pre-COVID placebo event correction:
  - `outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md`
  - Full 2022-strength suppression (`alpha=1`) worsens 2017 weighted MAE from `4.1272` to `4.2497`, while stronger `alpha=1.25` worsens it to `4.7632`.
  - This shows event priors need context and strength discipline rather than being treated as a universal downshift.
- LLM input leakage audit:
  - `outputs/leakage_audit/llm_input_leakage_audit_report.md`
  - 1327 cohort profiles and 89 batch prompt payloads have zero forbidden field violations.
  - Required guardrail instructions explicitly forbid `CNTTDHH`, `WTHHFIN`, `HOUSEID`, and aggregate 2022 target outcomes.

Add next:

- Irrelevant pseudo-event prior:
  - Ask/generate an event prior unrelated to travel suppression, such as generic “digital service adoption pressure,” and verify it should not improve trip-count prediction.
- Feature leakage audit:
  - Extend the current schema audit into a full prompt provenance appendix if writing the paper.

## Claim Discipline

| Phrase | Use? | Reason |
|---|---|---|
| causal guardrails | Yes | Describes audit logic |
| mechanism proxy | Yes | Accurate for LLM priors |
| causal plausibility | Yes | Properly scoped |
| causal effect | No | Not identified |
| treatment effect | No | No formal treatment assignment |
| confounding-aware diagnostic | Yes, if explained | We use controls/placebos, not full identification |

## PPT Integration

Best location:

- After the technical route slide if causal guardrails are emphasized.
- Or as a backup slide after robustness.

Spoken message:

> “We use causal structure as a guardrail: target labels do not enter the LLM branch, and negative controls test whether the event prior behaves like a meaningful mechanism rather than arbitrary prompt noise.”
