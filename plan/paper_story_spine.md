# Paper Story Spine

## Working Title

**Event-Generalizable LLM Priors for Label-Free Adaptation of Household Travel Behavior under Societal Shocks**

## One-Sentence Claim

When a societal shock breaks historical travel regularities and target-year labels are unavailable, an LLM can convert external event knowledge into structured mechanism priors that adapt a routine household travel model under causal guardrails and planning-oriented multi-objective evaluation.

## What This Paper Is Not

- Not “LLM directly predicts household trip counts.”
- Not “XGBoost plus prompt engineering.”
- Not “COVID-specific post-hoc curve fitting.”
- Not “causal effect estimation” unless a formal identification design is added.

## What This Paper Is

A label-free event-adaptation framework:

```text
historical household data
  -> routine mobility predictor

external event context + cohort profile
  -> LLM structured event priors

routine prediction + event priors + no-label adapter
  -> 2022 household behavior prediction

negative controls + uncertainty + Pareto objectives
  -> auditable planning interpretation
```

## Core Abstraction

Let `X_h` be household covariates, `E` be external event context, and `Y_2022` be the target behavior.

Historical supervised learning estimates:

```text
f_routine(X_h) ≈ expected travel under routine conditions
```

The LLM does not estimate `Y_2022`. It estimates a structured mechanism proxy:

```text
s_event = g_LLM(profile(X_h), E)
```

The adapter applies a pre-declared correction:

```text
ŷ_2022 = f_routine(X_h) × A(s_event)
```

The paper’s empirical question is whether `s_event` carries useful event-generalization information beyond household covariates and simple global shifts.

## Research Questions and Evidence

| RQ | Question | Evidence already available | Needed for paper-grade version |
|---|---|---|---|
| RQ1 | Is 2022 an event-shift target? | Temporal transfer report; XGBoost overprediction | More formal placebo framing |
| RQ2 | Does label-free LLM adaptation help? | `42.31%` weighted MAE reduction; near-zero bias; bootstrap CI | Stronger non-LLM baselines |
| RQ3 | Is the LLM role bounded and auditable? | Structured priors; no target labels in prompts; random/global controls | Frozen RAG context and prompt audit |
| RQ4 | Does the result support planning objectives? | Pareto analysis; mode/transit/mode-volume outputs | Equity objective and subgroup CIs |
| RQ5 | Does the behavior-system framing generalize beyond counts? | Mode and mode-volume gains; purpose extension exists | Stronger purpose-specific adapter or scope limitation |

## Main Figures for a Paper

1. Problem figure: routine temporal transfer fails in 2022.
2. Method figure: routine predictor + event prior + adapter + guardrails.
3. Main result: trip-count weighted MAE / bias / CI.
4. Control result: global, random, LLM-only, hybrid comparison.
5. Pareto result: operating points under planning objectives.
6. Behavior-system result: trip count, transit share, mode-specific trip volume.
7. Limitation/future: external validation and frozen event-context RAG.

## Claim Discipline

Allowed:

- “LLM-derived event priors improve label-free adaptation under a post-pandemic shift.”
- “The primary value is event-mechanism generalization, not direct numerical prediction.”
- “The global event baseline is strong; cohort-specific priors add auditable refinement.”
- “Causal guardrails improve plausibility and reduce leakage risk.”

Not allowed yet:

- “The LLM identifies causal effects.”
- “The method generalizes to all shocks.”
- “Purpose composition is solved.”
- “The LLM ranking is the dominant source of improvement.”

## Presentation Translation

For a 10-minute defense, the talk should use this path:

1. 2022 is not ordinary temporal transfer.
2. Routine household models overpredict after COVID.
3. LLMs are used only to produce event priors.
4. The hybrid adapter improves trip count with uncertainty.
5. Pareto analysis explains why the main method is chosen.
6. Mode/transit outputs connect to sustainable mobility.
7. Limitations and future work show how to reach paper level.
