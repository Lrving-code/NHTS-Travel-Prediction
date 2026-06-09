# NHTS Travel Prediction

本项目研究 2022 年疫情后分布变化下的 NHTS household mobility prediction。主线任务是预测家庭每日出行次数 `CNTTDHH`，并验证 LLM 生成的疫情事件先验能否在不使用 2022 目标标签训练/校准的情况下，修正历史出行预测器的系统性偏差。

## 当前结论

主任务是 household-level trip-count regression：

- Historical predictor weighted MAE: `4.3377`
- LLM trip-suppression correction weighted MAE: `2.4820`
- Weighted MAE reduction: `42.78%`
- Weighted RMSE reduction: `31.16%`
- Gated LLM correction weighted bias: `-0.0230`
- Gated household accuracy: exact `17.7%`, within 2 trips `54.5%`, within 3 trips `71.2%`

辅助任务是 household-level mode composition：

- Historical XGBoost weighted total variation: `0.2008`
- XGBoost + LLM transit prior weighted total variation: `0.1985`
- Transit-share weighted MAE: `0.0325 -> 0.0269`
- 该任务作为 supplementary / backup evidence；主贡献仍是 `CNTTDHH` 的 label-free event adaptation。

## 方法一句话

历史预测器学习正常时期的 routine mobility，LLM 生成 2022 疫情后 event priors，例如 trip suppression、remote work substitution、transit avoidance、online delivery substitution 和 recovery sensitivity。最终用这些先验修正 historical prediction：

```text
2022 prediction = historical routine prediction * event correction factor
```

注意：LLM 不是直接预测 household trip count，也不是替代 XGBoost。LLM 的角色是 event-prior adapter。

## 与参考资料/同学版本的区别

同学参考资料更接近 trip-level `TRPTRANS` mode classification：

- 输入：一条 trip 的距离、时长、目的、车辆/个人/地区信息
- 输出：这条 trip 的交通方式
- 指标：accuracy / F1 等分类指标

本项目主线是 household-level `CNTTDHH` regression：

- 输入：家庭和地区属性
- 输出：这个家庭 travel day 总出行次数
- 指标：MAE / RMSE / Bias / R2 / within-k accuracy

所以两者不能直接比较 accuracy 或 MAE。可以作为互补模块：我们预测出行强度，mode-composition 扩展预测出行方式结构。

## 主要输出

汇报材料：

- `outputs/final_project/NHTS_LLM_Event_Adaptation_Optimized_Presentation.pptx`
- `outputs/final_project/presentation_speaker_notes_zh.md`
- `outputs/final_project/method_comparison_report_zh.md`
- `outputs/final_project/method_comparison_report.md`

主实验：

- `outputs/final_project/final_project_report.md`
- `outputs/final_project/final_metrics_summary.csv`
- `outputs/final_project/household_accuracy_summary.csv`
- `outputs/final_project/figures/`

辅助实验：

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
  generate_final_project_assets.py
  build_optimized_presentation.py

plan/
  current_research_design.md
  paper_logic_chain.md
  llm_event_adaptation_plan.md
  mode_composition_extension_plan.md
  presentation_optimization_plan.md

outputs/
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

mode-composition 辅助实验：

```powershell
python src\run_mode_composition_extension.py --device cuda
```

优化汇报材料：

```powershell
python src\build_optimized_presentation.py
```

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
- `TRPTRANS` 编码在 2017 和 2022 不能直接按数字对齐，mode-composition 扩展使用 year-specific official codebook mapping。
- `outputs/share_package/` 是本地分享包目录，默认 git ignored。
- `refs/` 是本地参考资料目录，默认 git ignored。
- 不要提交 `.env`、API key、原始 LLM JSONL 请求日志或原始 NHTS 大文件。
