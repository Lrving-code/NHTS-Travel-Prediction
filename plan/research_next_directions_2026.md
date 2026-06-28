# 2025-2026 Research Directions: LLM, Causality, and Multi-Objective Mobility Modeling

## Why Update the Story

The current project is strong as a course project because it already shows:

- 2022 NHTS is an event-shift target rather than ordinary temporal transfer.
- A traditional supervised baseline captures routine household heterogeneity but overpredicts post-pandemic travel.
- LLM event priors reduce trip-count error and systematic bias without using 2022 trip-count labels for calibration.
- The output scope has expanded to trip generation, mode composition, purpose composition, and derived mode-specific trip volumes.

To make the story more research-grade, the next step should not be “ask the LLM more directly.” The better direction is:

```text
LLM as event-generalization module
+ causal guardrails
+ multi-objective planning/evaluation
+ structured spatio-temporal/foundation-model baselines
```

## Recent Top-Tier Signals

### 1. Event-driven mobility: routine pattern vs event constraint

**ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework**  
ICLR 2026 Poster.  
Links: https://openreview.net/forum?id=MPYsaBgZIT and https://arxiv.org/abs/2603.07946

Key idea: LLM mobility models handle routine trajectories better than event-deviated mobility. ELLMob explicitly models the competition between habitual patterns and event-imposed constraints and evaluates on Typhoon Hagibis, COVID-19, and Tokyo Olympics event settings.

Relevance: This is the closest conceptual support for our framing. Our project also separates routine household mobility from pandemic-event constraints.

### 2. LLMs need structured steering for spatio-temporal data

**U-STS-LLM: A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation**  
arXiv 2026.  
Link: https://arxiv.org/abs/2605.11735

Key idea: LLMs need explicit spatio-temporal structural guidance, such as dynamic attention bias, graph state, LoRA tuning, and gated fusion. Prompt-only prediction is not enough.

Relevance: Supports our use of a fixed adapter and cohort-level structured features rather than direct LLM regression.

### 3. LLM city-wide traffic foundation models

**A scalable and generic framework for city-wide traffic prediction with large language model**  
Nature Communications 2026.  
Link: https://www.nature.com/articles/s41467-026-73610-2

Key idea: LLM-based traffic prediction can be made scalable and generic across modes, contexts, and time granularities, evaluated across 11 datasets and 29 cities/areas.

Relevance: Raises the bar for paper-level validation. A publishable version of our work should eventually validate beyond one NHTS shock year.

### 4. Time-series foundation models as strong transportation baselines

**Time Series Foundation Models as Strong Baselines in Transportation Forecasting**  
arXiv 2026.  
Link: https://arxiv.org/html/2602.24238v2

Key idea: Chronos-2 zero-shot forecasts are competitive across transportation datasets and provide uncertainty estimates.

Relevance: If we frame any part of our work as temporal forecasting, we need foundation-model baselines. For NHTS, this is less direct because our task is cross-sectional household prediction, but it motivates stronger baseline discussion.

### 5. Causal intervention for LLM spatio-temporal forecasting

**Causal Intervention Is What Large Language Models Need for Spatio-Temporal Forecasting**  
IEEE Transactions on Cybernetics 2025.  
Link: https://www.semanticscholar.org/paper/Causal-Intervention-Is-What-Large-Language-Models-Li-Li/f4a79745a54801b220633964a01b195ae7e8783c

Key idea: adaptive graphs and LLMs can suffer from spurious spatial associations and hallucinations. The paper proposes a spatio-temporal causal intervention LLM with a causal intervention encoder and chain-of-action prompting.

Relevance: Gives us a rigorous angle: LLM priors should be constrained by causal structure and negative controls, not treated as free predictors.

### 6. LLM-as-parser for multi-objective route planning

**LLMAP: LLM-Assisted Multi-Objective Route Planning with User Preferences**  
Findings of EMNLP 2025.  
Links: https://arxiv.org/abs/2509.12273 and https://liangqiy.com/publication/llmap_llm-assisted_multi-objective_route_planning_with_user_preferences/LLMAP_LLM-Assisted_Multi-Objective_Route_Planning_with_User_Preferences.pdf

Key idea: The LLM should parse preferences and task dependencies, while a graph search/optimization solver handles constrained multi-objective routing.

Relevance: Directly supports our preferred LLM role: the LLM interprets goals and event context; the structured solver/model performs prediction and trade-off optimization.

### 7. Agentic and benchmarked urban LLMs

**AgentMove: A Large Language Model based Agentic Framework for Zero-shot Next Location Prediction**  
NAACL 2025.  
Link: https://github.com/tsinghua-fib-lab/AgentMove

Key idea: Decompose mobility prediction into memory, world knowledge, collective pattern extraction, and reasoning rather than directly asking the LLM for an answer.

**CityGPT / CityBench**  
KDD 2025.  
Links: https://github.com/tsinghua-fib-lab/CityGPT and https://github.com/tsinghua-fib-lab/CityBench

Key idea: Urban LLMs need domain-specific spatial cognition and systematic task benchmarks.

Relevance: Supports a decomposed design: traditional model + cohort profile + event priors + validation, instead of direct LLM output.

### 8. Urban spatio-temporal foundation models

**UrbanDiT: A Foundation Model for Open-World Urban Spatio-Temporal Learning**  
NeurIPS 2025.  
Links: https://arxiv.org/abs/2411.12164 and https://github.com/tsinghua-fib-lab/UrbanDiT

**TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration**  
NeurIPS 2025.  
Links: https://openreview.net/forum?id=9Ook5bXnPr and https://github.com/tsinghua-fib-lab/TrajAgent

Relevance: The field is moving toward foundation models and large/small model collaboration. Our project should describe XGBoost as a small structured predictor and the LLM as an event reasoning module.

## Better Research Ideas for This Project

### Direction A: Causal Event Adaptation

Research question:

> Can LLM-generated event priors serve as causal mechanism proxies for post-pandemic travel suppression, while avoiding target leakage and spurious correlations?

Design:

1. Build a causal graph:
   - household attributes -> routine travel demand
   - event mechanisms -> remote work, transit avoidance, delivery substitution
   - event mechanisms + household attributes -> trip generation / mode / purpose
2. Treat LLM priors as mechanism proxies, not treatment effects.
3. Add negative controls:
   - random pressure
   - global pressure
   - irrelevant pseudo-event prior
   - pre-COVID placebo year
4. Estimate heterogeneous event response:
   - subgroup gain by income, vehicle ownership, worker count, urban/rural, rail access
   - optionally causal forest / double ML as a supplementary diagnostic
5. Report “causal plausibility” rather than claiming causal identification.

What to implement next:

- Add a causal DAG figure in the PPT.
- Add placebo event priors for 2017 and show they should not improve much.
- Add subgroup effect tables with confidence intervals.

### Direction B: Multi-Objective Evaluation and Pareto Tuning

Current evaluation optimizes mainly wMAE and bias. For mobility inequality and sustainable development, a better objective is multi-dimensional:

```text
minimize prediction error
minimize systematic bias
minimize subgroup disparity
preserve transit / active mobility signals
avoid over-suppressing low-mobility households
minimize LLM request cost
```

Possible composite metrics:

- Accuracy: weighted MAE / RMSE
- Calibration: absolute weighted bias
- Equity: worst-subgroup MAE or subgroup-gain variance
- Sustainability: transit-share MAE, active-mode MAE
- Robustness: permutation p-value or CI width
- Efficiency: number of LLM requests / tokens

LLM role:

- LLM parses stakeholder priorities:
  - “equity-first”
  - “accuracy-first”
  - “transit-planning-first”
  - “low-cost deployment”
- The solver chooses Pareto-optimal adapter settings from pre-declared alpha/gating candidates.

This follows the LLMAP pattern: LLM-as-parser, structured solver as optimizer.

What to implement next:

- Build a Pareto frontier over existing methods and alpha/gate settings.
- Add one PPT page: “No single best model; choose operating point by planning objective.”

### Direction C: Event-Context RAG Instead of Free LLM Memory

Current setup allows generic pandemic knowledge. A stronger paper version should freeze event context:

1. Build a small RAG corpus from pre-2022 or during-2022 documents:
   - remote work reports
   - transit ridership recovery
   - public health restrictions
   - e-commerce/delivery substitution reports
   - Census / BLS telework statistics
2. Retrieve context per cohort:
   - urban/rural
   - rail availability
   - worker count
   - income category
3. Ask the LLM to output event priors with citations/snippets.
4. Validate that no NHTS 2022 target outcomes enter the corpus.

This makes the “LLM generalization” story stronger and more defensible.

### Direction D: Large-Small Model Collaboration

Use the TrajAgent / U-STS-LLM idea:

- Small model:
  - XGBoost / LightGBM / CatBoost predicts routine household behavior.
- LLM:
  - produces event priors, explanations, and scenario variables.
- Adapter:
  - combines routine prediction and event priors through a fixed or learned gated rule.
- Validator:
  - checks leakage, schema, confidence, negative controls, uncertainty.

This is exactly aligned with our current architecture and should be the main story.

### Direction E: From Prediction to Planning Simulation

Once trip generation + mode + purpose outputs exist, we can create counterfactual scenarios:

- stronger remote-work persistence
- faster transit recovery
- fuel price shock
- transit safety improvement
- targeted subsidy for zero-vehicle households

The model can output:

- total trip change
- transit trip change
- active-mode trip change
- work/shopping/social purpose change
- subgroup disparity change

This would connect directly to “mobility inequality and sustainable development.”

## Recommended Ranking

For the current course project:

1. **Add multi-objective/Pareto evaluation**: easiest, directly improves story.
2. **Add causal DAG + placebo/negative controls**: high conceptual value, modest implementation.
3. **Frame LLM as event-generalization module**: already supported, should be emphasized in PPT.

For a paper submission:

1. Event-context RAG with frozen corpus.
2. External validation across another shock or city/region.
3. Stronger baselines: CatBoost/LightGBM, causal DiD-style correction, foundation-model baseline where applicable.
4. Multi-objective and subgroup uncertainty analysis.

## Revised Paper Story

Title candidate:

**Event-Generalizable LLM Priors for Causal and Multi-Objective Adaptation of Household Travel Behavior under Societal Shocks**

Cleaner contribution statement:

> We formulate post-pandemic household travel prediction as event-driven adaptation. A traditional supervised model learns routine household mobility, while an LLM generalizes event mechanisms into structured priors. Causal guardrails, negative controls, uncertainty analysis, and multi-objective evaluation turn the LLM from a black-box predictor into an auditable event-adaptation module.
