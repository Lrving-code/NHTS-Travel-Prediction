# Project Quality Assessment

## Bottom Line

作为“大数据与城市规划”课程大作业，本项目已经足够拿出来汇报，而且亮点比较清楚：它不是普通的跨年监督预测，而是把 2022 NHTS 预测定义为 post-pandemic distribution shift，再用 LLM event prior 做 label-free correction。

作为论文，目前还不够。更准确的判断是：可以发展成 workshop / course-paper / short-paper 级别的研究原型；如果要投正式交通规划或数据挖掘论文，需要补 household-level external microdata validation、更加严格的前瞻式事件上下文和更强的 mode-choice 实验。

一句话判断：**大作业可以讲，论文还需要补证据链**。当前最适合的题目不是“LLM 提高 NHTS 预测准确率”，而是“在目标年份标签不可用时，如何用 LLM 事件先验修正历史出行模型的 distribution shift”。

## Evidence from Our Results

- Ordinary historical XGBoost weighted MAE: `4.3377`.
- Primary hybrid gated weighted MAE: `2.5023`, reduction `42.31%`.
- Ordinary historical XGBoost weighted bias: `+3.6052`.
- Primary hybrid gated weighted bias: `-0.0230`, almost removing systematic overprediction.
- LLM-only pressure weighted MAE: `2.7175`, weaker than hybrid. This supports the claim that LLM should be an event-prior adapter, not a standalone predictor.
- Zero-shot LLM rule tree weighted MAE: `2.6019`; pseudo-label tree distilled from it: `2.6040`. These baselines answer the direct "why not let the LLM build a 2022 tree" question and remain weaker than the hybrid adapter.
- LLM rule + 500 historical calibration weighted MAE: `2.7723`; full-history rule calibration: `2.7870`. This integrates the collaborator's rule-distillation + small-data calibration idea, but it is not the strongest setting for the 2022 event-shift task.
- Irrelevant pseudo-event controls do not reproduce the main method: best ranked pseudo-event weighted MAE `2.6274`, best gated pseudo-event weighted MAE `2.5610`.
- Transit-share weighted MAE improves from `0.0325` to `0.0269`, but full mode composition improves only modestly.
- Household exact rounded accuracy improves from `5.84%` to `17.72%`; weighted within-2-trips coverage improves from `23.39%` to `53.95%`.
- External mechanism validation supports the event-prior direction: ACS worked-from-home commute share changes from `5.7%` in 2019 to `15.2%` in 2022, while public-transportation commute share changes from `5.0%` to `3.1%`.
- BTS daily mobility is useful as a compatibility guardrail but not as an external numeric target: its device-based trips/person do not align with NHTS travel-diary `CNTTDHH`.

These numbers are strong enough for a course report because the baseline failure, correction mechanism, and external mechanism support are visible. They are not enough for a paper if framed as pure prediction SOTA, because exact household-level prediction remains hard, the mode-composition extension is still exploratory, and external household-level microdata validation is not yet available.

## External Positioning

The 2022 NHTS official trend report supports our problem framing. It documents post-pandemic travel changes, including pandemic-related reasons for fewer trips, reduced public transit use, more online shopping, and much higher household immobility in 2022 than in previous waves. This means 2022 is not a normal cross-year transfer target: [2022 NHTS Summary of Travel Trends](https://nhts.ornl.gov/assets/2022/pub/2022_NHTS_Summary_Travel_Trends.pdf?_cl=ROJh03eGaVX88DooPy4JXwQB).

Recent NHTS household trip-generation work treats gradient boosting and related ML methods as strong baselines for nonlinear household travel modeling. This makes ordinary XGBoost a reasonable traditional baseline rather than a weak strawman: [Machine Learning Modeling of Household Trip Generation by State Using NHTS Data](https://www.mdpi.com/2413-8851/9/9/353).

Recent LLM mobility work also motivates our design. LLM-MPE argues that public-event mobility prediction benefits from textual event information and a separation between regular mobility and event-related deviations: [Exploring Large Language Models for Human Mobility Prediction under Public Events](https://arxiv.org/abs/2311.17351). A broader review also notes that LLM applications in transportation grew rapidly from 2023 to 2025 and include travel behavior prediction, prompt engineering, and zero/few-shot setups: [Large Language Models for Traffic and Transportation Research](https://arxiv.org/abs/2503.21330).

The current 2026 frontier raises the bar. A Nature Communications paper on LLM-based city-wide traffic prediction validates across 11 real-world datasets and 29 cities/areas, which implies that a full paper needs broader external validation than one NHTS shock year: [A scalable and generic framework for city-wide traffic prediction with large language model](https://www.nature.com/articles/s41467-026-73610-2).

There is also a 2025 daily travel behavior paper that uses the 2022 NHTS dataset in a supervised hybrid recurrent-network plus fine-tuned LLM pipeline, reporting over 80% mode/purpose accuracy and trip-count MAE around 0.77: [Integrating hybrid recurrent neural networks and large language models for daily travel behavior prediction](https://doi.org/10.1016/j.trip.2025.101793). This is not directly comparable to our label-free transfer setup, but it means we should not pitch our result as supervised NHTS SOTA. Our defensible angle is label-free pandemic adaptation before 2022 labels are available.

## Course Project Judgment

For a course project, the story is strong because it has:

1. A real planning problem: 2022 post-pandemic mobility is a distribution-shift year.
2. A clear baseline failure: historical XGBoost overpredicts household trips.
3. A concrete LLM role: generate `s_event`, not direct labels.
4. Multiple controls: historical mean, XGBoost, trend rule, global event rule, random pressure, LLM-only, zero-shot LLM rule tree, pseudo-label tree, LLM rule + small historical calibration, irrelevant pseudo-event placebo, hybrid.
5. A guarded conclusion: event-level correction is strong; cohort ranking is incremental; mode composition remains exploratory.

The main presentation should emphasize the first three points and use the controls as credibility support.

## Paper Readiness Judgment

Current version is not ready for a full paper because:

1. It validates one shock year only: 2022 NHTS.
2. The LLM context is retrospective; GPT-5.5 may already know COVID-era mobility facts.
3. The strongest improvement is close to global event downscaling, so cohort-specific LLM ranking should not be overclaimed.
4. Mode composition improvement is concentrated in transit and does not yet form a complete mode-choice model.
5. External validation is mechanism-level, not household-level external MAE.

Adversarial reviewer concern: the strongest alternative explanation is that most of the gain comes from a global 2022 shock correction, not from rich LLM reasoning. The current version now addresses this with LLM-only, zero-shot rule-tree, random-prior, irrelevant pseudo-event, and global-prior controls. A paper would still need stronger external evidence that cohort-specific semantic priors add value beyond a calibrated shock scalar across multiple shocks or regions.

To make it paper-ready, the next experiments should be:

1. Prospective event-context freezing: use only documents/data available before 2022 NHTS release.
2. Household-level external validation: find a compatible post-pandemic household travel survey with trip diaries or region-level add-on microdata.
3. Cross-region or multi-shock validation: at least one additional event or held-out region.
4. Stronger baselines: calibrated global shock model, difference-in-differences style correction, CatBoost/LightGBM, and a simple external-feature model.
5. Mode-choice strengthening: move from transit-share correction toward a complete mode/purpose event adapter.

## Presentation Recommendation

Do not pitch this as “LLM beats transportation models.” The defensible claim is:

> Historical NHTS models learn household-level routine mobility; LLM event priors encode post-pandemic travel suppression mechanisms; a fixed label-free adapter combines the two and substantially reduces 2022 overprediction.

This is a good course-project claim and a plausible seed for a paper after stronger validation.

Suggested report stance:

1. Course project: present as a complete empirical pipeline with a clear research question, baselines, leakage guardrails, and interpretable results.
2. Paper seed: present as a label-free event adaptation idea, not as final SOTA.
3. Main insight: LLMs are useful when they are constrained to produce event priors, while the traditional supervised baseline remains responsible for household heterogeneity.
