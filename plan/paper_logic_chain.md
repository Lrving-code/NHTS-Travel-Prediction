# Paper Logic Chain: Event-Aware LLM-Guided Travel Demand Adaptation

## One-Sentence Contribution

We study post-pandemic household travel demand prediction as an event-driven temporal adaptation problem and propose using LLM-derived event semantics to correct the systematic bias of historical tabular models under rare societal shocks.

## Working Title

**Event-Aware LLM-Guided Temporal Adaptation for Post-Pandemic Household Travel Demand Prediction**

## Core Narrative

### 1. Routine travel-demand models assume historical continuity

Household travel demand models trained on historical survey waves learn routine relationships between household attributes and daily travel behavior. For example, household size, income, vehicle ownership, urban/rural status, census region, weekday, and rail availability are normally informative predictors of household trip counts.

In our current NHTS baseline, the historical training waves are 2001, 2009, and 2017, while the target wave is 2022.

### 2. The 2022 target year violates routine continuity

The 2022 NHTS is a post-pandemic recovery survey wave. COVID-19 changed travel behavior through multiple mechanisms:

- remote work and reduced commuting
- school and activity disruption
- public transit avoidance
- health-risk concerns
- online shopping and delivery substitution
- higher immobility for some household groups

These mechanisms are only weakly encoded, or not encoded at all, in standard NHTS household variables.

### 3. Our baseline evidence shows systematic transfer failure

The first household baseline predicts `CNTTDHH`, the household travel-day trip count.

Current CUDA XGBoost results:

| Experiment | Train Years | Test Year | Weighted MAE | Weighted RMSE | Weighted Bias |
|---|---|---:|---:|---:|---:|
| Direct transfer | 2017 | 2022 | 4.5921 | 5.6292 | +3.9598 |
| Pooled history | 2001+2009+2017 | 2022 | 4.7163 | 5.8269 | +4.1255 |
| Pooled history with year | 2001+2009+2017 | 2022 | 4.4062 | 5.4076 | +3.7202 |

The key observation is not only high error, but positive bias: historical models consistently overpredict 2022 household trip counts.

This supports the claim that 2022 behavior is not simply a smooth continuation of pre-pandemic travel patterns.

### 4. Why traditional tabular adaptation is insufficient

Standard domain adaptation can reweight historical samples to match the 2022 covariate distribution. This may help when the problem is mostly covariate shift.

However, the pandemic setting contains event-driven behavioral mechanisms that are not fully represented by observed tabular covariates. Two households with similar historical covariates may respond differently if their household profile implies different exposure to event constraints, such as commute substitution, transit avoidance, or delivery adoption.

Therefore, we need a mechanism that can represent event semantics and behavioral constraints.

### 5. Why LLMs are relevant

The 2026 literature suggests that LLMs are useful in mobility research when the task requires reasoning about:

- activity semantics
- traveler intentions and preferences
- real-world constraints
- rare events that deviate from routine mobility
- cross-scenario generalization

The closest methodological inspiration is ELLMob, which frames event-driven mobility as a conflict between habitual patterns and event constraints. Other 2026 work points toward structured spatio-temporal steering, gated fusion, trend enhancement, and LLM modules rather than direct prompt-only prediction.

### 6. Our method hypothesis

LLMs should not directly predict household trip counts. Instead, they should act as event-semantic adapters.

For each household profile, an LLM estimates latent event-response factors:

- `trip_suppression_risk`
- `remote_work_substitution_likelihood`
- `transit_avoidance_likelihood`
- `online_delivery_substitution_likelihood`
- `post_pandemic_recovery_sensitivity`

These factors encode how pandemic-era constraints plausibly affect household travel demand beyond what routine tabular features capture.

### 7. Proposed model structure

Let:

- `X`: harmonized household tabular features
- `y`: household trip count `CNTTDHH`
- `f_tabular(X)`: historical tabular predictor
- `z_llm(X, event_context)`: LLM-derived event semantic features
- `g_residual(X, z_llm)`: residual adapter

The final prediction is:

```text
y_hat = f_tabular(X) + g_residual(X, z_llm)
```

The LLM is used to produce structured event-response features, while a smaller supervised residual model learns how to combine those features with tabular data.

### 8. Main research questions

**RQ1: How severe is post-pandemic temporal shift in NHTS household travel demand prediction?**

Evidence:

- Compare historical-to-2022 transfer baselines.
- Report MAE, RMSE, bias, weighted MAE, weighted RMSE, and subgroup errors.

**RQ2: Can LLM-derived event semantic features reduce systematic 2022 overprediction?**

Evidence:

- Compare tabular baseline vs LLM-feature-augmented residual adaptation.
- Primary target: reduce weighted bias while preserving or improving weighted MAE/RMSE.

**RQ3: Which household groups benefit most from event-aware adaptation?**

Evidence:

- Subgroup error analysis by income, vehicle ownership, worker count, urban/rural status, rail availability, and census region.

**RQ4: Is LLM reasoning adding event semantics rather than just model capacity?**

Evidence:

- Ablate LLM features.
- Compare against random features, hand-coded pandemic features, and target-mean calibration.
- Check whether LLM features correlate with interpretable pandemic mechanisms.

## Expected Paper Contributions

1. **Problem framing:** Post-pandemic NHTS household travel demand prediction is formulated as event-driven temporal adaptation rather than ordinary cross-year prediction.
2. **Empirical evidence:** Historical tabular models systematically overpredict 2022 household trip counts, revealing a strong post-pandemic shift.
3. **Method:** An LLM-guided residual adaptation framework injects event semantics into structured travel-demand prediction.
4. **Analysis:** Subgroup and ablation studies identify where event-aware LLM guidance helps and where it fails.

## Method Positioning

This is not:

- LLM direct numerical regression
- generic prompt engineering
- replacing transportation models with a chatbot

This is:

- structured event feature generation
- residual adaptation under temporal shift
- semantic prior injection for rare societal shocks
- hybrid tabular + LLM modeling

## Leakage Rules

The official 2022 NHTS report can be used for motivation and post-hoc discussion, but not as model input if we claim forecasting or pre-2022 adaptation.

Allowed retrieval/model context:

- pre-2022 or early-2022 public information about COVID travel behavior
- general pandemic behavior mechanisms
- transportation-domain knowledge available before the target labels

Not allowed as model input:

- aggregate 2022 NHTS target statistics
- post-hoc summaries of 2022 NHTS trip reductions
- any information that directly reveals the target outcome distribution

## Figure 1 Concept

Figure 1 should show:

1. Historical NHTS waves learn routine household mobility.
2. COVID/post-pandemic event constraints shift 2022 behavior downward.
3. Historical tabular predictor overestimates 2022 trips.
4. LLM event-semantic adapter generates household-specific shock features.
5. Residual adapter corrects predictions and reduces bias.

## Related Work Buckets

1. Household travel demand and NHTS-based mobility modeling
2. Temporal domain adaptation and covariate shift in tabular prediction
3. LLMs for human mobility and transportation forecasting
4. Event-driven mobility modeling under societal shocks
5. Hybrid structured-data and LLM systems

## Current Evidence Snapshot

Data:

- NHTS 2001, 2009, 2017, 2022 public-use household data
- 357,553 harmonized household records
- 28 common household variables across all four waves

Baseline:

- CUDA XGBoost on RTX 4090
- Best current weighted MAE: 4.4062
- Best current weighted RMSE: 5.4076
- Best current weighted bias: +3.7202

Interpretation:

- The model is learning historical trip-generation behavior, but the 2022 target distribution is shifted downward.
- The positive bias creates a clear measurable target for event-aware adaptation.

## Immediate Next Experiment

Build an LLM event-feature generation module with a controlled JSON schema, generate features for household cohorts or individual households, and train a residual correction model against a small 2022 validation split.
