# Top-Venue Adversarial Audit, Round 2

Date: 2026-06-10

Scope: paper package after adding the LaTeX manuscript skeleton, verified references, and core figures.

## Reviewer-Level Verdict

The project is strong enough for a course final presentation and has a coherent workshop / applied-ML paper story. It is not yet a guaranteed top-tier conference or journal submission because the current strongest evidence is still one primary shock year in one national survey dataset. The contribution is defensible only if framed as **label-free event adaptation for survey-based household mobility**, not as a general mobility foundation model or as proof that LLMs directly predict travel better.

## What Is Now Strong

| Criterion | Current evidence | Verdict |
|---|---|---|
| Single main story | `plan/paper_logic_chain.md`; LaTeX title/abstract/method | Strong |
| Label-free target-year setting | leakage audit, prospective context, fixed adapter wording | Strong for target labels |
| Main empirical result | wMAE `4.3377 -> 2.5023`; bias `+3.6052 -> -0.0230` | Strong |
| Method spectrum | pure LLM, zero-shot rule tree, rule+history, global prior, random controls, Poisson/Tweedie/negative-binomial count models | Strong for course/paper rebuttal |
| Statistical validation | bootstrap CIs and paired improvements | Good |
| Robustness controls | permutation null, irrelevant pseudo-event, pre-COVID placebo | Good |
| Global-vs-cohort decomposition | cohort-prior value analysis: primary beats same-alpha global prior by `0.0508` wMAE and wins in `77.8%` of subgroup cells | Good and appropriately scoped |
| External evidence | ACS mechanism evidence; PSRC household microdata replication | Good but not definitive |
| Presentation readiness | 33-slide deck with method-spectrum, cohort-prior defense, and B1-B9 backup Q&A | Strong |
| Paper package | Markdown draft + LaTeX skeleton + verified citations + figures | Good first submission draft |
| Classical grounding | McFadden, Ben-Akiva/Lerman, Cameron/Trivedi, NHTS weighting report | Improved |

## Remaining Top-Tier Risks

| Risk | Why a strict reviewer may object | Required improvement |
|---|---|---|
| One main shock year | A single 2022 NHTS target can look like a hand-tuned event correction | Add at least one more exogenous mobility shock or regional shock replication with the same no-label protocol |
| Strong global prior | If global downscaling is already close, the LLM-specific contribution may look incremental | Current cohort-prior analysis addresses this for subgroup cells; still frame as selective refinement, not dominant source of gain |
| LLM pretraining leakage | GPT-style models may know post-pandemic mobility facts | Run or document a local open-source LLM control with explicit pretraining cutoff if feasible |
| No full causal identification | Current causal guardrails are negative controls, not causal effect estimates | Avoid causal-effect claims; keep causal language to guardrails and mechanism plausibility |
| Mode/purpose are exploratory | Main gain is trip generation; mode/purpose improvements are uneven | Keep mode/purpose as behavior-system extensions, not primary solved tasks |
| LaTeX compile environment | MiKTeX BibTeX cannot run in the current elevated shell | Compile in a normal user shell or Overleaf before sending to collaborators/advisor |
| Count-model depth | Poisson/Tweedie/negative-binomial baselines are transparent checks, but not fully optimized zero-inflated or regularized travel-demand count models | Keep the solver caveat; add zero-inflated or more carefully regularized count variants for a submission version if time permits |

## Current Round Fixes

- Added four core figures to `outputs/paper_draft/latex/main.tex`:
  - framework overview;
  - main metric comparison;
  - permutation robustness check;
  - multi-objective / behavior-system evaluation.
- Added citation-verification and build notes in the LaTeX package.
- Updated readiness audits so the LaTeX manuscript and verified references are part of the checked artifact set.
- Added transparent Poisson/Tweedie count-model baselines with solver diagnostics under `outputs/count_model_baselines/`.
- Added a stable negative-binomial GLM check with alpha-selection diagnostics under `outputs/negative_binomial_baseline/`.
- Added cohort-prior value analysis under `outputs/cohort_prior_value_analysis/`, showing selective subgroup gains over same-alpha and reported global priors.
- Added verified classical travel-demand, count-data, and NHTS survey-weighting citations to `references.bib`, `main.tex`, and `citation_verification_log.md`.

## Recommended Next Experiments

1. **Open-source LLM prior replication.** Use a local model on the RTX 4090 for a smaller cohort subset, then compare event-prior ranking with GPT-5.5-generated priors.
2. **Cohort-value extension.** The core subgroup analysis is now present; a paper version should add confidence intervals for the subgroup gains and cleaner codebook labels.
3. **External shock replication.** Use PSRC or another household survey to replicate the same no-label adapter around a non-COVID shock or recovery period.
4. **Count-model extension.** The transparent Poisson/Tweedie/negative-binomial checks are now present; a submission version should add zero-inflated or more carefully regularized count variants if feasible.
5. **Advisor-facing paper version.** Convert `main.tex` into a venue template only after the story and experiments are stable.

## Safe Top-Line Claim

Use:

> LLM event priors can act as a label-free semantic adapter for historically grounded household travel models under post-pandemic event shift.

Avoid:

> LLMs solve post-pandemic household travel prediction or outperform transportation models in general.
