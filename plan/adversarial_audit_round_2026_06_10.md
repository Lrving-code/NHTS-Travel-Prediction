# Adversarial Audit Round: 2026-06-10

## Review Stance

Assume the reviewer is skeptical of LLM-for-mobility papers and will reject any claim that looks like:

- post-hoc prompt engineering,
- target leakage,
- weak baselines,
- one-dataset overclaiming,
- decorative LLM usage,
- or unclear contribution boundaries.

## Summary Judgment

The current project is solid for a 10-minute course defense. It is not yet top-journal-ready. The strongest publishable core is:

> **Label-free event adaptation for household travel behavior under societal shocks using LLM-derived structured event priors with causal guardrails and multi-objective planning evaluation.**

The project should not claim that LLMs directly predict household trips. It should claim that LLMs provide event-mechanism priors that help structured models adapt when labels are not available.

## Severity-Ranked Issues

### Critical 1: One-shock validation is not enough for top-journal generalization

Risk: A reviewer can argue that the method is tuned to 2022 NHTS and does not demonstrate event generalization.

Current evidence:

- Strong 2022 NHTS trip-count result.
- Pre-COVID temporal transfer validation shows 2022 is a stronger shift.

Required fix:

- Add external validation or at least an out-of-domain stress test.
- Candidate: use external aggregate mobility/telework/transit recovery statistics as validation targets or context-only checks.
- Minimum course-level fix: clearly state this limitation and show the temporal-transfer placebo.

### Critical 2: LLM priors may contain retrospective world knowledge

Risk: Even without 2022 NHTS labels, GPT-5.5 could know broad COVID-era mobility outcomes.

Current evidence:

- Inputs exclude 2022 target labels.
- Prospective event-context plan exists.

Required fix:

- Implement event-context RAG with frozen source documents.
- Audit prompts to ensure the model is asked for mechanism priors, not target outcomes.
- For the current PPT, state this as a limitation and future paper requirement.

### Major 3: Global event prior is a strong baseline

Risk: If global pressure nearly matches cohort-aware LLM, reviewers may ask why the LLM is necessary.

Current evidence:

- Global event prior is very strong.
- Pareto analysis shows low-cost deployment selects `global_trip_suppression_a1`.
- Cohort-aware gated method has much lower bias than minimum-MAE sensitivity and better R2.

Required fix:

- Reframe contribution as event-level label-free adaptation plus auditable cohort refinement.
- Add subgroup/heterogeneity analysis to show where cohort-aware priors matter.
- Avoid saying individualized LLM ranking is the dominant source of gain unless evidence supports it.

### Major 4: Stronger non-LLM baselines are needed

Risk: Reviewers may expect CatBoost, LightGBM, reweighting, and simple domain adaptation before accepting LLM claims.

Current evidence:

- Historical mean, XGBoost, trend shift, global event rule, random pressure, LLM-only, hybrid.
- Added stronger baselines:
  - CatBoost GPU: weighted MAE `4.2196`, weighted bias `+3.4873`.
  - stronger XGBoost CUDA: weighted MAE `4.4173`, weighted bias `+3.7364`.
  - covariate-shift reweighted XGBoost CUDA: weighted MAE `4.4304`, weighted bias `+3.7523`.
  - LightGBM CPU: weighted MAE `4.4430`, weighted bias `+3.7646`.
- Primary gated event adapter remains lower than best stronger non-LLM baseline by `1.7173` weighted MAE, a relative reduction of `40.70%`.

Required fix:

- Keep these baselines in the main report/PPT.
- Add constrained historical shift rules selected without target labels if time permits.
- Make clear that covariate-shift reweighting does not solve mechanism shift.

### Major 5: Purpose-composition result is weak

Risk: Multi-output story may look overextended if one output does not improve.

Current evidence:

- Purpose output exists but LLM purpose prior is not overall best.

Required fix:

- Present purpose as an exploratory extension and boundary condition.
- Keep main claims on trip generation, transit share, and mode-specific trip volume.
- Only make purpose a headline if a stronger purpose-specific adapter is implemented.

### Major 6: Causal language must be disciplined

Risk: “Causal” without identification will be criticized.

Current evidence:

- Negative controls and temporal validation exist.
- No causal identification design yet.

Required fix:

- Use “causal guardrails,” “mechanism proxy,” and “causal plausibility.”
- Add DAG and negative controls.
- Do not claim estimated causal effects unless a formal design is added.

### Moderate 7: PPT still has too many result slides for 10 minutes

Risk: The audience may miss the main story.

Required fix:

- Keep the 10-minute talk to around 12-14 core slides and treat the rest as backup if needed.
- Mark a “talk path” in speaker notes.
- Each result slide should have exactly one spoken message.

### Moderate 8: Multi-objective evaluation needs equity dimension

Risk: The course theme includes mobility inequality; current Pareto objectives focus on error, bias, mode fidelity, and cost.

Required fix:

- Add worst-subgroup MAE or subgroup gain variance.
- Use bootstrap or subgroup CIs if possible.

## Fixes Implemented in This Round

- Added a multi-objective Pareto experiment:
  - `src/run_multi_objective_pareto_analysis.py`
  - `outputs/multi_objective_pareto/multi_objective_pareto_report.md`
  - `outputs/multi_objective_pareto/trip_pareto_frontier.png`
  - `outputs/multi_objective_pareto/behavior_system_improvement.png`
- Added a PPT page for multi-objective operating-point selection.
- Updated README with Pareto outputs and reproduction command.
- Added a top-journal upgrade roadmap:
  - `plan/top_journal_upgrade_roadmap.md`
- Added stronger non-LLM baselines:
  - `src/run_stronger_tabular_baselines.py`
  - `outputs/strong_baselines/strong_tabular_baseline_report.md`
  - Main PPT result table now includes CatBoost GPU.

## Next Audit Gate

The next round should not add more slides first. It should strengthen the empirical core:

1. Verify local GPU environment and package support.
2. Add equity-aware Pareto objective from subgroup metrics.
3. Add causal DAG and pseudo-event placebo artifact.
4. Rebuild PPT with a cleaner 10-minute talk path and backup-slide separation.
