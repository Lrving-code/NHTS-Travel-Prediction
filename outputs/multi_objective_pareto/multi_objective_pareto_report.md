# Multi-Objective Pareto Analysis

## Scope

This analysis turns the existing adapter grid into a planning-oriented multi-objective evaluation. It does not retrain any model. It asks which operating point should be selected when accuracy, calibration, behavior-system fidelity, and LLM request cost are considered together.

## LLM Cost Proxy

- Traditional and random-control methods: `0` LLM requests.
- Global event prior methods: `1` event-level request proxy.
- Cohort-aware LLM adapter methods: `89` batched cohort requests with batch size 15.

## Key Trip-Generation Trade-Off

| Method | Weighted MAE | Abs weighted bias | Weighted RMSE | Weighted R2 | Request cost |
|---|---:|---:|---:|---:|---:|
| historical_xgboost | 4.3377 | 3.6052 | 5.3169 | -0.6367 | 0 |
| llm_trip_suppression_a1p25 | 2.4820 | 0.5404 | 3.6601 | 0.2244 | 89 |
| gated_trip_suppression_a1_d0p15 | 2.5023 | 0.0230 | 3.6038 | 0.2480 | 89 |

Interpretation: the single-objective sensitivity candidate is slightly more accurate on MAE, but the primary gated adapter has much lower aggregate bias and better weighted R2. This is why the project should present the result as a multi-objective operating-point choice rather than a single leaderboard.

## Preference-Based Operating Points

| Profile | Selected method | Family | Score | Weighted MAE | Abs bias | Request cost |
|---|---|---|---:|---:|---:|---:|
| minimum_mae_sensitivity | llm_trip_suppression_a1p25 | llm_event_adapter | 0.0000 | 2.4820 | 0.5404 | 89 |
| calibration_first | gated_trip_suppression_a1_d0p15 | gated_llm_adapter | 0.0023 | 2.5023 | 0.0230 | 89 |
| balanced_course_report | gated_trip_suppression_a1_d0p15 | gated_llm_adapter | 0.0037 | 2.5023 | 0.0230 | 89 |
| low_cost_deployment | global_trip_suppression_a1 | global_event_prior | 0.0189 | 2.5531 | 0.1229 | 1 |

## Multi-Output Behavior Summary

| Objective | Metric | Traditional | Event-adapted | Reduction |
|---|---|---:|---:|---:|
| Trip generation | trip_count_weighted_mae | 4.3377 | 2.5023 | 42.31% |
| Trip generation calibration | trip_count_abs_weighted_bias | 3.6052 | 0.0230 | 99.36% |
| Mode composition | mode_weighted_total_variation | 0.2008 | 0.1985 | 1.13% |
| Sustainable mode signal | transit_share_weighted_mae | 0.0325 | 0.0269 | 17.35% |
| Mode-specific trip volume | mode_specific_trip_volume_mae | 4.8467 | 3.2802 | 32.32% |

## Paper Story Implication

The stronger claim is not that an LLM directly predicts travel better. The stronger claim is that a large-small model system can expose a Pareto set of event-adapted predictions: the small structured model learns routine household heterogeneity, the LLM supplies event mechanisms, and the solver selects an auditable operating point for the planning objective.

This matches recent LLM-assisted optimization work: use the LLM to parse event context and stakeholder priorities, then use a deterministic solver or adapter grid to make the numerical decision.
