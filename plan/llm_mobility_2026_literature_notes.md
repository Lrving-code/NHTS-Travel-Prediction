# 2026 LLM Mobility Literature Notes

## Scope

This note summarizes 2026-era directions for using large language models in mobility, transportation forecasting, and event-aware spatio-temporal modeling. The goal is to inform the NHTS post-pandemic household travel demand project.

## Key 2026 Papers

### ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework

- Venue/status: ICLR 2026.
- URL: https://arxiv.org/abs/2603.07946
- Main idea: LLM-based mobility models handle routine trajectories better than event-deviated mobility, so the paper explicitly models the competition between habitual mobility and event constraints.
- Method pattern:
  - Construct event-annotated mobility datasets covering Typhoon Hagibis, COVID-19, and Tokyo 2021 Olympics.
  - Extract competing rationales between habit patterns and event constraints.
  - Iteratively self-align the LLM output so generated mobility remains both habitually grounded and event-responsive.
- Relevance to this project:
  - This is the closest conceptual match. Our 2022 NHTS problem is also a habitual-pattern vs pandemic-constraint adaptation problem.

### LLMs for Human Mobility: Opportunities, Challenges, and Future Directions

- Status: arXiv 2026 survey.
- URL: https://arxiv.org/abs/2603.12420
- Main idea: LLMs are useful in mobility because many tasks require place semantics, activity intentions, preferences, and real-world constraints that numerical coordinates or tabular attributes do not capture well.
- Task taxonomy:
  - Travel itinerary planning
  - Trajectory generation
  - Mobility simulation
  - Mobility prediction
  - Mobility semantics and understanding
- Relevance to this project:
  - Supports framing the LLM as a semantic/event-reasoning module rather than a black-box regressor.

### A Scalable and Generic Framework for City-Wide Traffic Prediction with Large Language Model

- Venue/status: Nature Communications, published May 26, 2026.
- URL: https://www.nature.com/articles/s41467-026-73610-2
- Main idea: LLM-based city-wide traffic prediction can improve generality across modes, scenarios, and time granularities.
- Method pattern:
  - Trend data enhancement module
  - Spatiotemporal feature encoding module
  - LLM module
  - Evaluation across 11 large-scale datasets from 29 cities/areas.
- Relevance to this project:
  - Suggests a hybrid architecture: first encode structured tabular/spatiotemporal signals, then use LLM/foundation-model components to improve cross-scenario generalization.

### U-STS-LLM: A Unified Spatio-Temporal Steered LLM

- Status: arXiv 2026.
- URL: https://arxiv.org/abs/2605.11735
- Main idea: LLMs need explicit structural guidance for non-text spatio-temporal data.
- Method pattern:
  - Dynamic spatio-temporal attention bias generator
  - Persistent functional graph + transient nodal states
  - LoRA for parameter-efficient tuning
  - Gated adaptive fusion
  - Unified objective for forecasting and imputation
- Relevance to this project:
  - Supports steering or gating the LLM with structured survey features instead of relying on prompt-only prediction.

### Time Series Foundation Models as Strong Baselines in Transportation Forecasting

- Status: arXiv 2026, revised May 18, 2026.
- URL: https://arxiv.org/abs/2602.24238
- Main idea: Chronos-2 zero-shot forecasts are competitive across transportation datasets and should be used as strong baselines.
- Relevance to this project:
  - If we frame the task as temporal forecasting, we need a foundation-model baseline, not only XGBoost.
  - For NHTS household survey data, this may be less directly applicable than event-aware LLM adaptation, because the task is cross-sectional household prediction rather than dense time series.

## Design Implications for Our Project

The best direction is not direct LLM regression. It should be:

**Event-aware LLM-guided residual adaptation for post-pandemic household travel demand prediction.**

Recommended architecture:

1. Train tabular baseline on historical NHTS waves.
2. Represent each household as a structured natural-language profile.
3. Use an LLM/RAG module to infer pandemic shock factors:
   - trip suppression risk
   - remote work substitution likelihood
   - transit avoidance likelihood
   - delivery/substitution likelihood
   - recovery sensitivity
4. Train a small residual model using tabular features plus LLM-derived event features.
5. Evaluate whether the LLM-guided features reduce the positive 2022 bias found in current baselines.

## Leakage Control

The 2022 NHTS Summary of Travel Trends should be used for motivation and evaluation discussion, not as model input. If the model is framed as forecasting 2022 from pre-2022 information, the retrieval corpus should be limited to information that would have been available before or during 2022, such as public COVID mobility restrictions, remote-work reports, and pre-2022 travel behavior evidence.

## Proposed Research Framing

Title candidate:

**Event-Aware LLM-Guided Temporal Adaptation for Post-Pandemic Household Travel Demand Prediction**

Core claim:

Historical tabular travel-demand models learn routine household mobility patterns but fail under rare societal shocks. LLMs can provide structured semantic priors about event constraints, enabling better residual adaptation under temporal distribution shift.
