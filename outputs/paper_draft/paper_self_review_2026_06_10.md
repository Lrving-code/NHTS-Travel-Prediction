# Paper Self-Review: 2026-06-10

## Verdict

The project is now a strong course project and a credible workshop or short-paper seed. It has a coherent research spine, strong primary trip-generation result, extensive baselines, leakage controls, external mechanism evidence, a 10-minute defense deck, and a paper draft. It should not yet be represented as a guaranteed top-tier full-paper submission because the main NHTS evaluation covers one shock year and the strongest alternative explanation remains a global post-pandemic downscaling effect.

## Structure Review

| Section | Status | Notes |
|---|---|---|
| Abstract | PASS | Draft includes problem, method, results, and scoped contribution. |
| Introduction | PASS | Clearly separates routine mobility and event response. |
| Related work | PARTIAL | Current anchors are verified, but a full paper still needs conventional household travel-demand and domain-adaptation citations. |
| Method | PASS | Fixed no-label adapter, LLM schema, and leakage exclusions are explainable. |
| Experiments | PASS | Broad baseline spectrum and multiple controls are documented. |
| Results | PASS | Main trip-generation results are strong and statistically supported. |
| Discussion | PASS | Global prior strength and cohort-ranking limits are acknowledged. |
| Limitations | PASS | External validation and mode/purpose boundaries are explicit. |

## Logic Consistency

| Question | Status | Evidence |
|---|---|---|
| Does the method match the research question? | PASS | The strict setting forbids 2022 target-label calibration and uses fixed event priors. |
| Are conclusions supported by results? | PASS | The primary claim is trip-generation event adaptation, supported by wMAE and bias improvements. |
| Is the LLM role overstated? | PASS | Draft and matrix state that the LLM is an event-prior generator, not a direct predictor. |
| Is causal language controlled? | PASS | Documents use "causal guardrails" and mechanism evidence, not causal-effect proof. |
| Are external validations scoped correctly? | PASS | ACS/BTS/PSRC are described as mechanism/principle support, not direct NHTS numeric validation. |

## Result Credibility

Strengths:

- Primary weighted MAE improves from `4.3377` to `2.5023`.
- Weighted bias moves from `+3.6052` to `-0.0230`.
- Bootstrap paired confidence intervals remain positive.
- Strong non-LLM baselines still overpredict.
- Zero-shot rule-tree and small-history calibration ablations address instructor/reviewer questions.
- Permutation and irrelevant pseudo-event controls reduce the risk that arbitrary cohort scores explain the result.
- External PSRC household microdata provides an independent regional pre/post replication of the event-adaptation principle.

Remaining risks:

- The global event prior is nearly as strong as the gated adapter, so the paper must not overclaim cohort-specific LLM reasoning.
- NHTS 2022 is one target shock year. A full top-tier submission would benefit from another region, shock, or household survey.
- Mode and purpose results are not yet strong enough to be central contributions.
- LLM pretraining may contain retrospective pandemic knowledge; the current design controls label leakage but cannot fully control model pretraining knowledge.

## Citation Review

Verified anchors:

- ELLMob, ICLR 2026, OpenReview.
- CausalMob, KDD 2025, arXiv and ACM DOI.
- AgentMove, NAACL 2025, ACL Anthology.
- AgentMob, 2026 preprint/OpenReview.
- ELP-Mob, SIGSPATIAL/GIS 2025, project repository and ACM DOI.
- UniMob, KDD 2025 / arXiv.
- STFM survey, arXiv 2025.

Still needed for a full paper:

- Classic household travel demand / trip-generation modeling references.
- NHTS methodology citation from official documentation.
- Domain adaptation / covariate shift baselines for tabular data.
- Transportation planning literature on post-pandemic mobility behavior.

## Figure and Table Review

Ready:

- Main method comparison table.
- Error/bias tradeoff.
- Bootstrap/statistical validation.
- Event heterogeneity and robustness controls.
- Mode/purpose extension figures.
- 31-slide defense deck with backup Q&A.

Recommended before paper submission:

- A single polished Figure 1 showing routine model, LLM event-prior branch, fixed adapter, and leakage guardrails.
- A compact method-spectrum table that groups direct LLM, rule tree, rule calibration, global prior, and gated adapter.
- A figure comparing global prior vs gated adapter to visually communicate the limited but real cohort-refinement gain.

## Next Paper-Level Work

1. Convert the Markdown draft into a venue-ready LaTeX skeleton.
2. Add a verified BibTeX file only after each citation is fetched from DOI/arXiv/ACL/OpenReview.
3. Add one more external survey/region if accessible, or state clearly that PSRC is the only external household replication.
4. Strengthen mode-choice experiments only if the paper title promises full behavior prediction; otherwise keep mode/purpose as extensions.
5. Ask an advisor/TA whether the course paper should emphasize algorithmic novelty, planning insight, or empirical auditability.

## Current Safe Submission Position

Safe title direction:

> Label-Free Event-Aware LLM Adaptation for Post-Pandemic Household Travel Demand Prediction

Safe thesis:

> Under a post-pandemic survey shift, historical household models overpredict routine travel. LLM-derived event priors, when distilled into fixed and auditable correction rules, can improve label-free transfer while preserving household-level numerical grounding.

Unsafe thesis:

> LLMs directly solve household travel prediction, replace transportation models, or prove causal COVID effects.
