# Top-Journal Upgrade Roadmap

## One Main Spine

This project should be framed as:

> **Event-generalizable household travel behavior adaptation under societal shocks.**

The central argument is not “LLM improves XGBoost.” The central argument is:

1. A historical structured model learns **routine household mobility**.
2. A societal event, such as COVID-19 recovery, changes the behavioral mechanism through remote work, transit avoidance, shopping substitution, and heterogeneous recovery.
3. LLMs are useful because they can translate external event knowledge into **structured event priors**.
4. Causal guardrails and negative controls keep those priors auditable.
5. Multi-objective evaluation selects an operating point for planning, not just a single leaderboard winner.

This keeps the paper from becoming a stitched-together project. Every component answers the same question:

> How can a travel behavior model adapt to a rare societal event when target-year labels are not available for calibration?

## Recent Literature Signals to Anchor the Story

| Signal | Verified source | What it supports in our project |
|---|---|---|
| Event-driven mobility needs event constraints beyond routine trajectories | ELLMob, ICLR 2026 Poster: https://openreview.net/forum?id=MPYsaBgZIT | Our routine model + event prior separation |
| LLM route/planning systems should let LLMs parse preferences while solvers handle constrained optimization | LLMAP, Findings of EMNLP 2025: https://aclanthology.org/2025.findings-emnlp.416/ | Our Pareto operating-point selection |
| LLMs for trajectory modeling work better as large-small model collaboration | TrajAgent, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/1f170d33c836cbebde939fb49c4e7d6e-Abstract-Conference.html | Our XGBoost small model + LLM event module |
| Urban LLMs need spatial/domain grounding and benchmarked urban tasks | CityGPT / CityBench, KDD 2025: https://kdd2025.kdd.org/research-track-papers-2/ | Our structured cohort prompts and leakage checks |
| Causal spatio-temporal prediction is becoming a top-conference concern | Causal Spatio-Temporal Prediction, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b71169b4db5448c494c91db848f060be-Abstract-Conference.html | Our causal guardrail and negative-control agenda |

## Paper-Level Research Questions

**RQ1: Event shift.** How large is the failure of routine household travel models under a post-pandemic event shift?

**RQ2: Label-free adaptation.** Can LLM-derived event priors improve 2022 household travel prediction without using 2022 target labels for calibration?

**RQ3: Mechanism and guardrails.** Do the gains survive negative controls, pre-COVID placebo checks, subgroup diagnostics, and leakage audits?

**RQ4: Planning objectives.** How should the model choose among accuracy, calibration, transit fidelity, equity, uncertainty, and LLM request cost?

**RQ5: Multi-output behavior system.** Does the event-adaptation idea transfer from trip generation to mode composition, purpose composition, and mode-specific trip volume?

## Current Evidence Strength

Current status after the 2026-06-29 audit:

- The project is strong enough for the course presentation and has a coherent paper-grade artifact package.
- 2022 is empirically different from pre-COVID transfer years.
- Traditional XGBoost strongly overpredicts 2022 household trips.
- The fixed no-label gated adapter reduces weighted MAE by about `42.3%` and nearly removes aggregate weighted bias.
- Bootstrap confidence intervals support the trip-count gain.
- Mode-specific trip volume and transit-share metrics improve under the behavior-system extension.
- Trip-level harmonized mode-choice transfer is now survey-weighted and reports weighted accuracy, weighted balanced accuracy, and weighted macro F1.
- Batch prompting reduces cohort requests from `1327` to `89`.
- A frozen event-context corpus now records ACS/BTS source facts, PSRC validation-only evidence, candidate sources not used, and the 89 frozen-context batch prompts; the prompt leakage audit checks all 1,327 cohort records with `0` violations, and an integrity manifest hashes 16 frozen-context artifacts.
- Pareto analysis now shows different operating points for minimum error, balanced reporting, uncertainty-aware reporting, and low-cost deployment.
- The LaTeX manuscript now has a successful manual `pdflatex -> bibtex -> pdflatex -> pdflatex -> pdflatex` compile record, with no undefined citations and no overfull hbox in the main results table.

Remaining top-tier risks:

- Only one target shock year and one national survey system.
- External validation has improved through PSRC household microdata replication, but it is still a principle-level external replication rather than a direct NHTS numerical validation.
- Purpose composition remains an exploratory output rather than a strong positive result.
- LLM priors are not yet fully regenerated from the frozen event-context corpus. The replay code exists, but execution needs either `CURSOR_API_AUTH_TOKEN` or a cached/downloadable local instruction model.
- Causal claims are currently plausibility claims, not identified treatment effects.
- More non-COVID or non-recovery shock replications would be needed before claiming broad event generalization.

## Required Upgrade Workstreams

### W1. Stronger Baselines - mostly complete

Completed baselines now include:

- Strong tabular baselines.
- Covariate-shift reweighted XGBoost using 2022 covariates without 2022 trip-count labels.
- CatBoost GPU and LightGBM CPU fallback baselines.
- Transparent Poisson, Tweedie, negative-binomial, and zero-inflated count-model baselines.
- Historical trend shift and global event-prior baselines.
- Zero-shot LLM rule tree, pseudo-label tree, and small historical calibration variants.
- Random pressure, irrelevant pseudo-event, and pre-COVID placebo controls.

Still useful if time permits:

- Add a difference-in-differences-style correction or external-feature-only model if a defensible external event feature panel is available.
- Re-run LightGBM with GPU only if the local build can support it cleanly; the current artifact records CPU fallback.

### W2. Causal Guardrails - complete for course and artifact gate

Implemented:

- Causal guardrail design and evidence pack.
- Negative controls: random pressure, irrelevant pseudo-event prior, global pressure, pre-COVID placebo.
- Subgroup robustness and worst-subgroup reporting.
- Claim discipline: use “causal plausibility” and “mechanism proxy,” not causal identification.

### W3. Event-Context RAG - prompt provenance complete, replay pending

Partly implemented as a frozen provenance package under `outputs/event_context_corpus/`. A full paper-grade RAG system would replace free-form LLM pandemic memory with a frozen external context:

- Telework reports.
- Transit ridership recovery.
- Public-health restrictions.
- E-commerce/delivery substitution.
- Fuel price or travel cost context if used.

Each LLM prior should be traceable to retrieved snippets. 2022 NHTS targets must not enter the corpus. The current artifact freezes source-level context and batch prompts, but it does not yet regenerate all priors from a stronger open-source model under retrieval-only constraints.

Current execution blockers:

- Batched API replay needs `CURSOR_API_AUTH_TOKEN`.
- Local replay needs a cached or downloadable instruction model, because `Qwen/Qwen2.5-1.5B-Instruct` was not available offline.

### W4. Multi-Objective Mobility Evaluation - complete for current scope

Implemented:

- Worst-subgroup MAE.
- Transit/active mode preservation.
- CI width or bootstrap robustness.
- LLM token/request cost.
- Stakeholder profiles: accuracy-first, equity-first, transit-planning-first, low-cost deployment.

### W5. Multi-Output Behavior System - implemented with scoped claims

Keep count, mode, purpose, and mode-specific volume under one behavior-system framing.

Implemented or available:

- Active mobility share.
- Transit/private mode ratio.
- Work-trip share.
- Low-carbon trip proxy.

Only foreground outputs that serve the event-adaptation story. Purpose composition should remain exploratory unless a stronger purpose-specific adapter is added.

### W6. Presentation and Paper Audit

Every PPT round should answer:

- What is the one message of this slide?
- Does the slide support one of the RQs?
- Is the LLM role clear and bounded?
- Are claims scoped to evidence?
- Does every table/figure have a reason to exist?
- Are weak results labeled as exploratory?

## GPU Execution Policy

Use the 4090 for all model experiments that support it:

- XGBoost: `device=cuda`.
- CatBoost: `task_type=GPU` if installed.
- LightGBM: GPU only if the local build supports it; otherwise document CPU fallback.
- Record `nvidia-smi`, Python version, package versions, and script arguments for each major run.

## Immediate Next Actions

1. If an LLM endpoint or local model is available, run frozen-context prior replay and compare regenerated priors against current GPT-reference priors.
2. If time permits, add a difference-in-differences-style or external-feature-only robustness baseline; covariate reweighting, CatBoost, and LightGBM have already been checked.
3. Keep the PPT and report story focused on: event shift -> label-free event priors -> causal/leakage guardrails -> multi-objective planning choice -> behavior-system outputs.
4. Do not upgrade the claim to broad causal identification or general mobility foundation modeling.
5. For a real venue submission, move the LaTeX draft into the target template and get advisor feedback on the external validation scope.
