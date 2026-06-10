# Zero-Shot LLM Rule-Tree Baseline

## 这个实验回答什么问题

老师追问的是：为什么不直接让 LLM 在零样本情况下构建 2022 年决策树？

这个补充实验把该问题落成一个 ablation：我们允许 rule tree 使用 household covariates 和已有 GPT-5.5 event-prior features，但不允许它读取 2022 `CNTTDHH` 标签、survey weight 或 2022 aggregate target statistics 来训练 split threshold 或 leaf value。

## Baseline 定义

- Baseline 名称：`zero_shot_llm_semantic_rule_tree`。
- 标签使用：No 2022 CNTTDHH labels are used for split selection, leaf values, or calibration.
- 关键限制：This is a deliberately direct LLM-style baseline. It represents a qualitative belief tree rather than a statistically trained decision tree, because true decision-tree thresholds and leaf values require labels.
- 额外实现：把 zero-shot rule tree 输出的 pseudo labels 蒸馏成一棵 sklearn `DecisionTreeRegressor`，得到 `zero_shot_pseudo_label_tree`。这模拟“LLM 先造 pseudo labels，再训练一棵树”的方案，但仍不使用真实 2022 labels。

## 结果

| 方法 | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | Within 2 trips | MAE vs primary |
|---|---:|---:|---:|---:|---:|---:|
| historical_xgboost | 4.3377 | 5.3169 | +3.6052 | -0.6367 | 26.5% | +1.8354 |
| llm_only_pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 | 51.0% | +0.2152 |
| zero_shot_llm_rule_tree | 2.6019 | 3.8300 | -0.4072 | 0.1507 | 52.8% | +0.0997 |
| zero_shot_pseudo_label_tree | 2.6040 | 3.8368 | -0.4072 | 0.1477 | 52.8% | +0.1017 |
| primary_hybrid_gated_adapter | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 54.5% | +0.0000 |

## 结论

- Primary hybrid gated adapter 相对 historical XGBoost 的 weighted MAE 降低 `42.31%`。
- Zero-shot rule tree 的 weighted MAE 比 primary hybrid 高 `3.98%`。
- Pseudo-label tree 的 weighted MAE 比 primary hybrid 高 `4.07%`。
- LLM-only pressure 的 weighted MAE 比 primary hybrid 高 `8.60%`。

这支持我们的主张：LLM 直接生成规则树可以作为 qualitative prior 或 ablation，但它没有真实历史 NHTS 监督模型提供的 household-level numerical grounding，因此数值校准不如 hybrid adapter 稳定。

## 答辩口径

可以做这个 baseline，而且我们已经把它作为补充实验实现了。结果表明，zero-shot LLM rule tree 能够表达疫情机制方向，但它更像 belief tree；没有标签时，split threshold 和 leaf value 缺少 empirical risk minimization 支撑。主方法把历史监督模型作为 routine mobility anchor，只让 LLM 提供 event prior，因此在误差、bias 和可审计性上更适合作为论文主线。

## Distilled pseudo-label tree

下面这棵树只拟合 zero-shot rule labels，不拟合真实 `CNTTDHH`：

```text
|--- llm_trip_suppression_pressure <= 0.455
|   |--- HHSIZE <= 3.500
|   |   |--- WRKCOUNT <= 1.500
|   |   |   |--- NUMADLT <= 1.500
|   |   |   |   |--- online_delivery_substitution_likelihood <= 0.575
|   |   |   |   |   |--- value: [3.450]
|   |   |   |   |--- online_delivery_substitution_likelihood >  0.575
|   |   |   |   |   |--- value: [3.229]
|   |   |   |--- NUMADLT >  1.500
|   |   |   |   |--- URBAN <= 1.500
|   |   |   |   |   |--- value: [3.774]
|   |   |   |   |--- URBAN >  1.500
|   |   |   |   |   |--- value: [3.449]
|   |   |--- WRKCOUNT >  1.500
|   |   |   |--- online_delivery_substitution_likelihood <= 0.595
|   |   |   |   |--- remote_work_substitution_likelihood <= 0.635
|   |   |   |   |   |--- value: [4.700]
|   |   |   |   |--- remote_work_substitution_likelihood >  0.635
|   |   |   |   |   |--- value: [4.636]
|   |   |   |--- online_delivery_substitution_likelihood >  0.595
|   |   |   |   |--- value: [4.246]
|   |--- HHSIZE >  3.500
|   |   |--- online_delivery_substitution_likelihood <= 0.615
|   |   |   |--- post_pandemic_recovery_sensitivity <= 0.515
|   |   |   |   |--- llm_trip_suppression_pressure <= 0.415
|   |   |   |   |   |--- value: [5.496]
|   |   |   |   |--- llm_trip_suppression_pressure >  0.415
|   |   |   |   |   |--- value: [5.461]
|   |   |   |--- post_pandemic_recovery_sensitivity >  0.515
|   |   |   |   |--- value: [5.560]
|   |   |--- online_delivery_substitution_likelihood >  0.615
|   |   |   |--- value: [5.299]
|--- llm_trip_suppression_pressure >  0.455
|   |--- llm_trip_suppression_pressure <= 0.590
|   |   |--- WRKCOUNT <= 0.500
|   |   |   |--- HHSIZE <= 2.500
|   |   |   |   |--- HHFAMINC <= 7.500
|   |   |   |   |   |--- value: [2.200]
|   |   |   |   |--- HHFAMINC >  7.500
|   |   |   |   |   |--- value: [2.134]
|   |   |   |--- HHSIZE >  2.500
|   |   |   |   |--- value: [2.812]
|   |   |--- WRKCOUNT >  0.500
|   |   |   |--- HHSIZE <= 3.500
|   |   |   |   |--- online_delivery_substitution_likelihood <= 0.595
|   |   |   |   |   |--- value: [3.128]
|   |   |   |   |--- online_delivery_substitution_likelihood >  0.595
|   |   |   |   |   |--- value: [2.833]
|   |   |   |--- HHSIZE >  3.500
|   |   |   |   |--- WRKCOUNT <= 1.500
|   |   |   |   |   |--- value: [3.400]
|   |   |   |   |--- WRKCOUNT >  1.500
|   |   |   |   |   |--- value: [4.174]
|   |--- llm_trip_suppression_pressure >  0.590
|   |   |--- remote_work_substitution_likelihood <= 0.230
|   |   |   |--- HHSIZE <= 1.500
|   |   |   |   |--- online_delivery_substitution_likelihood <= 0.570
|   |   |   |   |   |--- value: [1.124]
|   |   |   |   |--- online_delivery_substitution_likelihood >  0.570
|   |   |   |   |   |--- value: [1.004]
|   |   |   |--- HHSIZE >  1.500
|   |   |   |   |--- HHFAMINC <= 2.500
|   |   |   |   |   |--- value: [1.415]
|   |   |   |   |--- HHFAMINC >  2.500
|   |   |   |   |   |--- value: [1.320]
|   |   |--- remote_work_substitution_likelihood >  0.230
|   |   |   |--- remote_work_substitution_likelihood <= 0.565
|   |   |   |   |--- value: [2.402]
|   |   |   |--- remote_work_substitution_likelihood >  0.565
|   |   |   |   |--- value: [1.823]

```

## Artifacts

- Metrics: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_metrics.csv`
- Predictions: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_predictions.csv`
- Figure: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_rule_tree_metric_comparison.png`
- Rule spec: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_spec.json`
