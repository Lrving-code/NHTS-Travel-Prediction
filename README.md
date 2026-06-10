# Pandemic-Aware Household Travel Behavior Prediction

本项目研究 2022 年疫情后分布变化下的 NHTS household travel behavior prediction。核心思路是用 NHTS 历史数据训练传统监督模型来学习 routine mobility，再用 LLM 生成的疫情事件先验修正 post-pandemic shift。项目以家庭为单位预测一个综合出行行为系统：出行强度、出行方式结构、出行目的结构，以及派生的各方式出行量。

## 当前结论

行为目标 1 是 household-level trip-count regression：

- Temporal transfer validation: pre-COVID mean absolute weighted bias `0.9024` vs 2022 absolute weighted bias `3.7202`
- Traditional supervised baseline weighted MAE: `4.3377`
- Primary fixed no-label gated correction weighted MAE: `2.5023`
- Primary fixed no-label gated correction weighted bias: `-0.0230`
- Primary fixed no-label gated correction weighted R2: `0.2480`
- Primary gated weighted MAE reduction: `42.31%`
- LLM-only pressure weighted MAE: `2.7175`
- Compared with LLM-only pressure, the primary hybrid gated adapter reduces weighted MAE from `2.7175` to `2.5023` and reduces absolute weighted bias from `0.3732` to `0.0230`.
- Zero-shot LLM-authored rule tree weighted MAE: `2.6019`; pseudo-label DecisionTree distilled from that rule tree weighted MAE: `2.6040`.
- Compared with the zero-shot LLM rule tree, the primary hybrid gated adapter is lower by `0.0997` weighted MAE and has much smaller weighted bias (`-0.4072 -> -0.0230`).
- LLM rule + small historical calibration is implemented as a bridge baseline: 500 historical calibration rows give weighted MAE `2.7723`, and full-history rule calibration gives `2.7870`. This is useful for data-sparse/cold-start framing, but it is not the best setting for the 2022 event-shift task.
- Gated household accuracy: exact `17.7%`, within 2 trips `54.5%`, within 3 trips `71.2%`
- Stronger non-LLM baselines still overpredict 2022: best extra baseline is CatBoost GPU with weighted MAE `4.2196` and weighted bias `+3.4873`; the primary gated event adapter is `40.70%` lower in weighted MAE.
- Irrelevant pseudo-event placebo controls do not reproduce the main result: the best ranked pseudo-event weighted MAE is `2.6274`, and the best gated pseudo-event weighted MAE is `2.5610`.
- Equity-aware subgroup evaluation: worst-subgroup weighted MAE improves from `8.3778` under historical XGBoost to `4.7091` under the primary gated event adapter; all evaluated subgroups improve relative to historical XGBoost.
- Multi-objective analysis: calibration-first and balanced profiles select the primary `gated_trip_suppression_a1_d0p15`; low-cost deployment selects `global_trip_suppression_a1`.
- External mechanism validation: ACS commute statistics show worked-from-home share stayed much higher in 2022 than 2019 (`5.7% -> 15.2%`) and public-transportation commute share stayed lower (`5.0% -> 3.1%`), supporting the remote-work and transit-avoidance event priors. BTS daily mobility is retained as a compatibility guardrail: its device-based trip counts should not be used as direct numeric labels for NHTS household `CNTTDHH`.
- External household microdata validation: PSRC 2017+2019->2023 pre/post replication improves weighted MAE (`3.4271 -> 3.2498`) and weighted bias (`+1.0532 -> +0.2605`) when an ACS-derived remote-work suppression factor is applied without target-year PSRC label calibration. A BTS device-mobility recovery factor is retained as a negative compatibility guardrail because it worsens the household-survey transfer.

行为目标 2 是 household-level mode composition：

- Historical XGBoost weighted total variation: `0.2008`
- XGBoost + LLM transit prior weighted total variation: `0.1985`
- Transit-share weighted MAE: `0.0325 -> 0.0269`
- 该目标用于补充“怎么出行”的方式结构维度，和 `CNTTDHH` 一起构成 household travel behavior prediction。

行为目标 3 是 household-level purpose composition：

- `TRIPPURP` 被聚合为 work、shopping、social/recreation、other home-based、non-home-based 五类目的结构。
- Traditional XGBoost purpose weighted TV: `0.5661`。
- LLM purpose prior 目前不是整体最优；它作为第三个行为维度和边界实验，用于说明未来需要更强的 purpose-specific event adapter。

派生目标是 mode-specific trip volume：

- `predicted trips by mode = predicted total trips × predicted mode share`
- Traditional count × traditional mode total mode-trip MAE: `4.8467`
- Gated count × LLM mode total mode-trip MAE: `3.2802`

## 方法一句话

传统监督模型学习正常时期的 routine mobility，LLM 生成 2022 疫情后 event priors，例如 trip suppression、remote work substitution、transit avoidance、online delivery substitution 和 recovery sensitivity。最终用这些先验修正 routine prediction：

```text
2022 prediction = routine prediction * event correction factor
```

注意：LLM 不是直接预测 household trip count，也不是替代 XGBoost。LLM 的角色是 event-prior adapter。

和 pure LLM pressure 相比，hybrid gated 方法更稳：pure LLM 能抓到疫情后出行下降方向，但缺少 household-level routine baseline，容易把总量压得过低；hybrid 方法保留传统模型学到的家庭基础出行需求，再用 LLM event priors 做机制修正。

为什么不直接让 LLM 零样本构建 2022 决策树：决策树需要标签来学习 split threshold 和叶节点数值。没有 2022 `CNTTDHH` 标签时，LLM 只能生成 belief tree 或 synthetic labels，优化目标会变成拟合 LLM 自己的假设，而不是拟合真实 NHTS 行为。仓库已经把这个问题做成补充 ablation：zero-shot LLM rule tree 的 weighted MAE 是 `2.6019`，pseudo-label tree 是 `2.6040`，均弱于 primary hybrid gated adapter 的 `2.5023`，尤其 weighted bias 更不稳。因此本项目把 LLM 限定为 event-prior generator，把数值预测锚定在真实历史 NHTS 训练出的 household model 上。

## 学术问题与贡献

本项目把 2022 NHTS 预测定义为 **event-driven temporal adaptation** 问题，而不是普通 cross-year prediction。核心观察是：传统监督模型能学习家庭属性和 routine mobility 的关系，但 2022 疫情恢复期引入了 remote work、transit avoidance、online delivery substitution 等事件机制，这些机制没有被传统 household covariates 充分表示。

研究问题：

- **RQ1**：历史 household travel model 在 2022 post-pandemic shift 下会出现多大系统性偏差？
- **RQ2**：不使用 2022 `CNTTDHH` 标签训练/校准时，LLM 生成的 event priors 能否修正这种偏差？
- **RQ3**：LLM 的作用是替代传统监督模型，还是作为 event-semantic adapter 与传统监督模型互补？

方法贡献：

- 提出一个 label-free hybrid adaptation 结构：`routine predictor + LLM event prior + fixed correction rule`。
- 将 LLM 限定为结构化事件先验生成器，而不是直接数值预测器，降低 hallucination 和 target leakage 风险。
- 使用 cohort-level prompting，把 7,893 个家庭压缩为 1,327 个 cohort；进一步用 batch prompting 将请求数压缩到 89 个 batch prompts（batch size 15），降低请求成本并提高可审计性。
- 对比 historical mean、ordinary historical XGBoost、historical trend shift、global event rule、random pressure、LLM-only pressure 和 hybrid LLM adapter，说明 hybrid 设计的必要性。
- 额外实现 zero-shot LLM rule tree / pseudo-label tree baseline，回应“为什么不直接让 LLM 构建 2022 决策树”的审稿问题。
- 实现 LLM rule + small historical calibration 中间 baseline，回应“先由 LLM 提取规则、再用少量历史数据校准”的合作方案。
- 通过 irrelevant pseudo-event placebo 检验机制相关性：任意 distribution-matched cohort score 不能替代 post-pandemic event prior。

最新文献定位：

- `plan/literature_grounding_2026.md` 汇总了截至 2026-06-10 核查的相关方向，包括 ELLMob（ICLR 2026）、CausalMob（KDD 2025）、AgentMove（NAACL 2025）、UniMob（KDD 2025）、ELP-Mob（SIGSPATIAL/GIS 2025）和 spatiotemporal foundation model survey。
- 我们的定位不是 generic mobility foundation model，而是 **survey-based household mobility 的 event-driven label-free adaptation**：用 NHTS 历史标签锚定 routine mobility，用 LLM event priors 表达疫情冲击机制。

## 整体任务设计

本项目按家庭颗粒度组织成一个两阶段预测框架：

- **Trip generation**：预测家庭 travel day 总出行次数 `CNTTDHH`。
- **Mode composition**：把 trip-level `TRPTRANS` 聚合为家庭层面的 mode-share 向量，预测 private vehicle、walk、bike、transit、taxi/ridehail、other 的占比。
- **Purpose composition**：把 trip-level `TRIPPURP` 聚合为 household purpose-share 向量，预测 work、shopping、social/recreation、other home-based、non-home-based。
- **Mode-specific trip count**：把 trip generation 和 mode composition 组合为各方式出行次数：

```text
predicted trips by mode = predicted total trips * predicted mode share
```

因此仓库对外表述为一个整体：预测疫情后家庭层面的完整出行行为，包括“出行多少次”“用什么方式出行”“为什么出行”和“每种方式多少次”。

## 主要输出

汇报材料：

- `outputs/final_project/NHTS_Travel_Behavior_Template_Presentation.pptx`（当前主汇报版本，含稳健性检验）
- `outputs/final_project/NHTS_Travel_Behavior_Template_Presentation_v2.pptx`
- `outputs/final_project/NHTS_LLM_Event_Adaptation_Optimized_Presentation.pptx`
- `outputs/final_project/presentation_speaker_notes_zh.md`
- `outputs/final_project/project_quality_assessment.md`（课程汇报与论文潜力判断）
- `outputs/final_project/method_comparison_report_zh.md`
- `outputs/final_project/method_comparison_report.md`
- `outputs/final_project/figures/presentation_figures/`
- `plan/reviewer_qa_backup_2026.md`（答辩追问与 backup slides 口径）
- `outputs/submission_readiness/submission_readiness_audit.md`（投稿/答辩证据链审计）

主实验：

- `outputs/final_project/final_project_report.md`
- `outputs/final_project/final_metrics_summary.csv`
- `outputs/final_project/household_accuracy_summary.csv`
- `outputs/final_project/figures/`

稳健性检验：

- `outputs/robustness_checks/robustness_check_report.md`
- `outputs/robustness_checks/permutation_pressure_controls.csv`
- `outputs/robustness_checks/permutation_null_mae.png`
- `outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_report_zh.md`
- `outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv`
- `outputs/llm_rule_small_data_calibration/llm_rule_small_data_calibration_report_zh.md`
- `outputs/llm_rule_small_data_calibration/method_spectrum_metrics.csv`
- `outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_report.md`
- `outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv`
- `outputs/temporal_transfer_validation/temporal_transfer_validation_report.md`
- `outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md`
- `outputs/leakage_audit/llm_input_leakage_audit_report.md`
- `outputs/causal_guardrails/causal_guardrail_evidence_report.md`
- `outputs/causal_guardrails/causal_dag.png`

更强非 LLM baseline：

- `src/run_stronger_tabular_baselines.py`
- `outputs/strong_baselines/strong_tabular_baseline_metrics.csv`
- `outputs/strong_baselines/strong_tabular_baseline_report.md`

统计验证：

- `outputs/statistical_validation/confidence_interval_report.md`
- `outputs/statistical_validation/trip_metric_confidence_intervals.csv`
- `outputs/statistical_validation/paired_improvement_confidence_intervals.csv`

多目标与 Pareto 分析：

- `outputs/multi_objective_pareto/multi_objective_pareto_report.md`
- `outputs/multi_objective_pareto/preference_operating_points.csv`
- `outputs/multi_objective_pareto/trip_pareto_frontier.png`
- `outputs/multi_objective_pareto/behavior_system_improvement.png`

移动不平等与分组稳健性：

- `outputs/equity_aware_evaluation/equity_aware_evaluation_report.md`
- `outputs/equity_aware_evaluation/equity_summary.csv`
- `outputs/equity_aware_evaluation/equity_operating_points.csv`

出行方式结构实验：

- `outputs/mode_composition_extension/mode_composition_report.md`
- `outputs/mode_composition_extension/mode_composition_metrics.csv`
- `outputs/mode_composition_extension/mode_specific_trip_count_metrics.csv`
- `outputs/mode_composition_extension/figures/`

出行目的结构实验：

- `outputs/purpose_composition_extension/purpose_composition_report.md`
- `outputs/purpose_composition_extension/purpose_composition_metrics.csv`
- `outputs/purpose_composition_extension/figures/`

LLM 批量化与泛化设计：

- `plan/llm_generalization_strategy.md`
- `plan/prospective_event_context_2022.md`
- `outputs/llm_event_features/household_cohort_batch_prompt_b15_summary.md`

## 项目结构

```text
data/
  raw/                         # 本地原始 NHTS 数据，git ignored
  processed/
    household_harmonized.csv   # harmonized household dataset，git ignored

src/
  build_household_dataset.py
  build_llm_household_profiles.py
  generate_llm_event_features.py
  run_household_baseline.py
  run_label_free_llm_adaptation.py
  run_mode_composition_extension.py
  run_purpose_composition_extension.py
  run_statistical_confidence_intervals.py
  run_multi_objective_pareto_analysis.py
  run_equity_aware_evaluation.py
  run_pre_covid_placebo_event_correction.py
  run_zero_shot_llm_rule_tree_baseline.py
  run_llm_rule_small_data_calibration.py
  run_irrelevant_pseudo_event_placebo.py
  run_llm_input_leakage_audit.py
  build_causal_guardrail_evidence_pack.py
  run_external_aggregate_validation.py
  run_psrc_external_household_validation.py
  run_stronger_tabular_baselines.py
  build_batched_llm_event_prompts.py
  generate_presentation_figures.py
  run_robustness_checks.py
  generate_final_project_assets.py
  build_optimized_presentation.py
  build_template_presentation.py

plan/
  current_research_design.md
  paper_logic_chain.md
  llm_event_adaptation_plan.md
  mode_composition_extension_plan.md
  presentation_optimization_plan.md

outputs/
  robustness_checks/
  final_project/
  label_free_llm_adaptation/
  mode_composition_extension/
  purpose_composition_extension/
  external_validation/
  statistical_validation/
```

## 复现方式

需要本地已有 NHTS 数据和 LLM event features。原始数据和部分中间文件体积较大，默认不提交到 git。

主实验：

```powershell
python src\run_label_free_llm_adaptation.py --device cuda
python src\generate_final_project_assets.py
```

最终可复现产物：

```powershell
python src\export_final_repro_artifacts.py --device cuda
```

该脚本会在 `outputs/models/final_label_free_2022/` 保存 routine predictor、feature list、2022 prediction CSV、metrics、metadata、数据 hash 和环境信息。该目录默认不提交到 git。

mode-composition 实验：

```powershell
python src\run_mode_composition_extension.py --device cuda
```

purpose-composition 与统计验证：

```powershell
python src\run_purpose_composition_extension.py --device cuda
python src\run_statistical_confidence_intervals.py --bootstrap-runs 1000
```

稳健性与 placebo guardrail：

```powershell
python src\run_robustness_checks.py
python src\run_temporal_transfer_validation.py --device cuda
python src\run_pre_covid_placebo_event_correction.py --device cuda
python src\run_llm_input_leakage_audit.py
python src\build_causal_guardrail_evidence_pack.py
python src\run_external_aggregate_validation.py --device cuda
python src\run_psrc_external_household_validation.py --device cuda
```

多目标/Pareto 分析：

```powershell
python src\run_multi_objective_pareto_analysis.py
python src\run_equity_aware_evaluation.py
```

更强非 LLM baseline。建议先跑 XGBoost 与 covariate-shift reweighted XGBoost，确认 GPU 通道稳定后再单独跑 CatBoost/LightGBM：

```powershell
python src\run_stronger_tabular_baselines.py --device cuda --methods xgboost,reweighted_xgboost --n-estimators 200 --domain-n-estimators 120
python src\run_stronger_tabular_baselines.py --device cuda --methods catboost --n-estimators 200
python src\run_stronger_tabular_baselines.py --device cuda --methods lightgbm --n-estimators 200
```

LLM batch prompt 准备：

```powershell
python src\build_batched_llm_event_prompts.py --batch-size 15 --max-batch-chars 64000 --prospective-context-path plan\prospective_event_context_2022.md --output-jsonl outputs\llm_event_features\household_cohort_batch_prompts_b15.jsonl --summary-path outputs\llm_event_features\household_cohort_batch_prompt_b15_summary.md
```

优化汇报材料：

```powershell
python src\generate_presentation_figures.py
python src\run_robustness_checks.py
python src\run_multi_objective_pareto_analysis.py
python src\build_template_presentation.py
python src\build_optimized_presentation.py
```

其中 `build_template_presentation.py` 会读取 `temp/ppt_template_reference.pptx` 作为样式参考，生成中英双语、章节化、包含分析图和稳健性检验的主汇报 PPT。

## 关键文件依赖

复现完整实验至少需要：

```text
data/processed/household_harmonized.csv
outputs/llm_event_features/household_cohort_profiles.csv
outputs/llm_event_features/cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv
```

mode-composition 扩展还需要：

```text
data/raw/nhts_2017/csv/trippub.csv
data/raw/nhts_2022/csv/tripv2pub.csv
```

purpose-composition 扩展使用同一组 trip-level files，并额外依赖跨年可对齐的 `TRIPPURP` 字段。

## 重要注意事项

- 当前版本足够作为“大数据与城市规划”课程大作业汇报；如果要发展成论文，应按 `outputs/final_project/project_quality_assessment.md` 继续打磨 paper writing、advisor feedback、额外地区复刻和更强的 mode-choice 实验。
- 主实验不使用 2022 `CNTTDHH` 标签训练或校准，2022 标签只用于最终 evaluation。
- 汇报主口径使用固定 `gated_trip_suppression_a1_d0p15` no-label rule；参数扫描结果只放内部附录，不进入公开方法比较主表。
- 稳健性检验显示 global event pressure 是很强的 baseline；应把贡献表述为 event-level label-free adaptation，cohort-specific LLM ranking 是增量证据，不是唯一或主导来源。
- 外部验证现在包括 mechanism-level ACS/BTS 证据和 PSRC household-level direct pre/post microdata。PSRC 的结果支持 event-adaptation principle，但不要把它表述为 NHTS 2022 数值预测的直接外部验证。
- mode-composition 是探索性扩展：总体 weighted TV 改善较小，最清楚的结果是 transit-share weighted MAE 和 mode-specific trip volume 改善。
- purpose-composition 是综合目标扩展和边界实验：它说明项目可以预测“为什么出行”，但当前 LLM purpose prior 还不是整体最优。
- LLM 的核心价值应表述为 event-generalizable priors，而不是逐户直接预测；batch prompting 是工程加速，prospective event context 是防止 retrospective leakage 的论文级改进方向。
- `TRPTRANS` 编码在 2017 和 2022 不能直接按数字对齐，mode-composition 扩展使用 year-specific official codebook mapping。
- `outputs/share_package/` 是本地分享包目录，默认 git ignored。
- 不要提交 `.env`、API key、原始 LLM JSONL 请求日志或原始 NHTS 大文件。
