# Pandemic-Aware Household Travel Behavior Prediction

本项目研究 2022 年疫情后分布变化下的 NHTS household travel behavior prediction。核心思路是用历史 NHTS 学习 routine mobility，再用 LLM 生成的疫情事件先验修正 post-pandemic shift。项目以家庭为单位，同时预测出行强度和出行方式结构。

## 当前结论

行为目标 1 是 household-level trip-count regression：

- Historical predictor weighted MAE: `4.3377`
- Primary fixed no-label gated correction weighted MAE: `2.5023`
- Primary fixed no-label gated correction weighted bias: `-0.0230`
- Primary fixed no-label gated correction weighted R2: `0.2480`
- Primary gated weighted MAE reduction: `42.31%`
- Best-MAE sensitivity row weighted MAE: `2.4820`
- Best-MAE sensitivity reduction: `42.78%`
- LLM-only pressure weighted MAE: `2.7175`
- Gated household accuracy: exact `17.7%`, within 2 trips `54.5%`, within 3 trips `71.2%`

行为目标 2 是 household-level mode composition：

- Historical XGBoost weighted total variation: `0.2008`
- XGBoost + LLM transit prior weighted total variation: `0.1985`
- Transit-share weighted MAE: `0.0325 -> 0.0269`
- 该目标用于补充“怎么出行”的方式结构维度，和 `CNTTDHH` 一起构成 household travel behavior prediction。

## 方法一句话

历史预测器学习正常时期的 routine mobility，LLM 生成 2022 疫情后 event priors，例如 trip suppression、remote work substitution、transit avoidance、online delivery substitution 和 recovery sensitivity。最终用这些先验修正 historical prediction：

```text
2022 prediction = historical routine prediction * event correction factor
```

注意：LLM 不是直接预测 household trip count，也不是替代 XGBoost。LLM 的角色是 event-prior adapter。

## 学术问题与贡献

本项目把 2022 NHTS 预测定义为 **event-driven temporal adaptation** 问题，而不是普通 cross-year prediction。核心观察是：历史模型能学习家庭属性和 routine mobility 的关系，但 2022 疫情恢复期引入了 remote work、transit avoidance、online delivery substitution 等事件机制，这些机制没有被传统 household covariates 充分表示。

研究问题：

- **RQ1**：历史 household travel model 在 2022 post-pandemic shift 下会出现多大系统性偏差？
- **RQ2**：不使用 2022 `CNTTDHH` 标签训练/校准时，LLM 生成的 event priors 能否修正这种偏差？
- **RQ3**：LLM 的作用是替代历史模型，还是作为 event-semantic adapter 与历史模型互补？

方法贡献：

- 提出一个 label-free hybrid adaptation 结构：`routine predictor + LLM event prior + fixed correction rule`。
- 将 LLM 限定为结构化事件先验生成器，而不是直接数值预测器，降低 hallucination 和 target leakage 风险。
- 使用 cohort-level prompting，把 7,893 个家庭压缩为 1,327 个 cohort 请求，降低请求成本并提高可审计性。
- 对比 historical mean、ordinary historical XGBoost、historical trend shift、global event rule、random pressure、LLM-only pressure 和 hybrid LLM adapter，说明 hybrid 设计的必要性。

## 整体任务设计

本项目按家庭颗粒度组织成一个两阶段预测框架：

- **Trip generation**：预测家庭 travel day 总出行次数 `CNTTDHH`。
- **Mode composition**：把 trip-level `TRPTRANS` 聚合为家庭层面的 mode-share 向量，预测 private vehicle、walk、bike、transit、taxi/ridehail、other 的占比。
- **Mode-specific trip count**：把前两步组合为各方式出行次数：

```text
predicted trips by mode = predicted total trips * predicted mode share
```

因此仓库对外表述为一个整体：预测疫情后家庭层面的完整出行行为，包括“出行多少次”和“用什么方式出行”。

## 主要输出

汇报材料：

- `outputs/final_project/NHTS_Travel_Behavior_Template_Presentation_v2.pptx`（当前主汇报版本，含 adversarial audit）
- `outputs/final_project/NHTS_Travel_Behavior_Template_Presentation.pptx`
- `outputs/final_project/NHTS_LLM_Event_Adaptation_Optimized_Presentation.pptx`
- `outputs/final_project/presentation_speaker_notes_zh.md`
- `outputs/final_project/method_comparison_report_zh.md`
- `outputs/final_project/method_comparison_report.md`
- `outputs/final_project/figures/paper_style/`

主实验：

- `outputs/final_project/final_project_report.md`
- `outputs/final_project/final_metrics_summary.csv`
- `outputs/final_project/household_accuracy_summary.csv`
- `outputs/final_project/figures/`

对抗性审计：

- `outputs/adversarial_audit/adversarial_audit_report.md`
- `outputs/adversarial_audit/permutation_pressure_controls.csv`
- `outputs/adversarial_audit/permutation_null_mae.png`

出行方式结构实验：

- `outputs/mode_composition_extension/mode_composition_report.md`
- `outputs/mode_composition_extension/mode_composition_metrics.csv`
- `outputs/mode_composition_extension/figures/`

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
  generate_paper_style_figures.py
  run_adversarial_audit.py
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
  adversarial_audit/
  final_project/
  label_free_llm_adaptation/
  mode_composition_extension/
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

该脚本会在 `outputs/models/final_label_free_2022/` 保存 historical routine model、feature list、2022 prediction CSV、metrics、metadata、数据 hash 和环境信息。该目录默认不提交到 git。

mode-composition 实验：

```powershell
python src\run_mode_composition_extension.py --device cuda
```

优化汇报材料：

```powershell
python src\generate_paper_style_figures.py
python src\run_adversarial_audit.py
python src\build_template_presentation.py
python src\build_optimized_presentation.py
```

其中 `build_template_presentation.py` 会读取 `temp/ppt_template_reference.pptx` 作为样式参考，生成中英双语、章节化、包含 paper-style analysis figures 的主汇报 PPT。

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

## 重要注意事项

- 主实验不使用 2022 `CNTTDHH` 标签训练或校准，2022 标签只用于最终 evaluation。
- 汇报主口径使用固定 `gated_trip_suppression_a1_d0p15` no-label rule；`llm_trip_suppression_a1p25` 是 best-MAE sensitivity row，不应表述为严格无标签参数选择的主方法。
- 对抗性审计显示 global event pressure 是很强的 baseline；应把贡献表述为 event-level label-free adaptation，cohort-specific LLM ranking 是增量证据，不是唯一或主导来源。
- mode-composition 是探索性扩展：总体 weighted TV 改善较小，最清楚的结果是 transit-share weighted MAE 改善。
- `TRPTRANS` 编码在 2017 和 2022 不能直接按数字对齐，mode-composition 扩展使用 year-specific official codebook mapping。
- `outputs/share_package/` 是本地分享包目录，默认 git ignored。
- 不要提交 `.env`、API key、原始 LLM JSONL 请求日志或原始 NHTS 大文件。
