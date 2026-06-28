# 2025-2026 Literature Grounding

Last verified: 2026-06-10

## One-Sentence Position

This project sits at the intersection of event-driven mobility prediction, LLM-derived event semantics, and causal/multi-objective evaluation. The defensible novelty is not that an LLM directly predicts household trips; it is that an LLM supplies label-free event priors that correct a historically grounded household travel model under a rare post-pandemic temporal shift.

## 2025-2026 Signals To Cite

| Direction | Representative work | What it shows | How this project differs |
|---|---|---|---|
| Event-driven LLM mobility | ELLMob, ICLR 2026: https://openreview.net/forum?id=MPYsaBgZIT | Routine mobility and event constraints can conflict; LLMs are useful for event-responsive mobility generation. | We study survey-based household travel demand prediction rather than trajectory generation, and we keep the numerical predictor grounded in NHTS labels from pre-2022 years. |
| Causal event features | CausalMob, KDD 2025: https://doi.org/10.1145/3690624.3709231 and code page https://github.com/YangXiaojie1998/CausalMob | LLMs can transform public-event text into structured intention/treatment features for mobility prediction with causal framing. | We do not have target-year event articles at household level, so we use prospective event priors plus leakage/placebo/negative-control guardrails. |
| Zero-shot LLM mobility agents | AgentMove, NAACL 2025: https://aclanthology.org/2025.naacl-long.61/ | Direct LLM-based agents can support zero-shot next-location prediction but need systematic design. | Our ablation shows direct LLM/rule-tree prediction is directionally useful but less calibrated than the hybrid adapter. |
| Efficient LLM mobility pipeline | ELP-Mob, SIGSPATIAL/GIS 2025: https://github.com/chwang0721/ELP-Mob | LLM mobility prediction pipelines need batching/efficiency to be usable. | We use cohort-level and batch prompting to reduce calls from household-level prompting to 89 batch prompts. |
| Evidence-grounded LLM mobility agents | AgentMob, arXiv/OpenReview 2026: https://arxiv.org/abs/2606.05130 | The newest agentic direction uses a fast path for routine cases and invokes extra LLM/tool reasoning only for ambiguous mobility cases. | Our method makes the same efficiency principle survey-compatible: routine household demand comes from the historical model, and LLM reasoning is distilled into auditable event priors rather than used on every household. |
| Universal mobility foundation models | UniMob, KDD 2025: https://doi.org/10.1145/3690624.3709236 and arXiv https://arxiv.org/html/2412.15294v1 | The field is moving from task-specific mobility models toward universal trajectory/flow modeling. | Our target is not universal mobility sequence generation; it is event-shift adaptation for NHTS household survey prediction. |
| Spatiotemporal foundation models | STFM survey, 2025: https://arxiv.org/html/2503.13502v1 | Foundation models are increasingly framed around generalization, reasoning, and efficiency in ST data science. | We use a lightweight LLM event-prior adapter rather than training a large spatiotemporal foundation model. |
| LLM traffic/time-series adaptation | ST-LLM+ / TimeCMA / TimeKD references in the ST-LLM repo: https://github.com/ChenxiLiu-HNU/ST-LLM | Recent work adapts LLMs to traffic and time-series forecasting through spatial-temporal embeddings, cross-modality alignment, or distillation. | Our data are tabular household surveys; event semantics enter as structured priors, not as tokenized traffic sequences. |
| Urban mobility language-modeling survey | Language Modeling for Urban Mobility, 2025: https://github.com/urban-mobility-generation/Language-Modeling-for-Urban-Mobility | Mobility data can be transformed into language-model-like representations through tokenization, encoding, and prompting. | We choose structured cohort prompts because NHTS is survey-tabular rather than GPS trajectory data. |

## Gap Statement

Recent work shows that LLMs can help mobility modeling through event reasoning, zero-shot mobility agents, and spatiotemporal foundation-model adaptation. However, most of these studies focus on trajectories, flows, traffic sensors, or public-event time series. They do not directly address the NHTS setting where:

- the prediction unit is a household survey record rather than a trajectory or road node;
- the target year is a rare post-pandemic distribution shift;
- target-year trip labels should not be used for calibration in the intended prospective setting;
- the model must be interpretable enough for mobility inequality and planning discussion.

This motivates our framework: historical NHTS waves learn routine household mobility, while an LLM supplies event-generalizable priors that are converted into fixed, auditable correction rules.

## Verification Notes

- ELLMob was checked against OpenReview, which lists it as an ICLR 2026 Poster and describes the routine-pattern/event-constraint conflict.
- CausalMob was checked through arXiv `2412.02155`, which lists the KDD 2025 acceptance and DOI `10.1145/3690624.3709231`.
- AgentMove was checked through ACL Anthology `2025.naacl-long.61`, which lists the NAACL 2025 venue and DOI `10.18653/v1/2025.naacl-long.61`.
- UniMob was checked through arXiv `2412.15294`; the ACM DOI page may block automated fetches, so arXiv is the reproducible metadata fallback.
- ELP-Mob was checked through the public GitHub repository, whose README includes the SIGSPATIAL/GIS 2025 citation and DOI.
- AgentMob was checked through arXiv `2606.05130` and OpenReview search results; treat it as a very recent 2026 preprint rather than a peer-reviewed anchor.

## Reviewer-Facing Claims

- Safe claim: The method is an event-adaptation framework for survey-based household mobility prediction.
- Safe claim: LLM event priors help mainly by correcting post-pandemic overprediction; cohort-specific ranking adds incremental signal.
- Safe claim: Pure LLM and zero-shot rule-tree baselines are useful controls, but they are less numerically calibrated than the hybrid adapter.
- Unsafe claim: The method is a general mobility foundation model.
- Unsafe claim: The LLM alone understands household travel demand better than supervised historical data.
- Unsafe claim: The current mode/purpose modules solve full mode choice or purpose choice.

## Paper Story To Use

1. Mobility prediction is moving toward foundation and language-model paradigms, but survey-based household travel demand remains strongly tied to calibrated historical labels.
2. Public shocks such as COVID-19 break the routine relationship learned from pre-event data.
3. LLMs are useful precisely where tabular covariates are weak: representing event mechanisms such as remote work, transit avoidance, delivery substitution, and uneven recovery.
4. The proposed adapter separates routine mobility from event response, avoiding both historical overprediction and uncalibrated LLM direct prediction.
5. Causal guardrails, zero-shot rule-tree baselines, irrelevant pseudo-event placebo, and multi-objective evaluation make the claim auditable rather than purely prompt-driven.
