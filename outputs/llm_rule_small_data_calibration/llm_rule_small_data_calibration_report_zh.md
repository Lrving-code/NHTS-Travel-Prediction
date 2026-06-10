# LLM Rule + Small Historical Calibration

## 这个实验回答什么问题

这个实验回应队友提出的融合路线：先让 LLM 提供可解释的 rule structure，再用少量历史 NHTS 样本校准规则叶节点，而不是先训练完整历史 XGBoost 再用 LLM 修正。

它和主方法的关系是中间 baseline，而不是替代主方法：

```text
data-only fitting -> LLM-only -> zero-shot LLM rule -> LLM rule + small historical calibration -> historical model + LLM event adapter
```

## 设计

- LLM-style routine rule：按 household size、vehicle ownership、worker count、urban context 分出 routine demand leaves。
- Small historical calibration：只用 2001/2009/2017 的少量样本校准每个 leaf 的 base demand。
- Event adaptation：2022 不使用 `CNTTDHH` 训练或校准，只用已有 LLM trip-suppression pressure 作为固定 event factor。
- 每个 sample size 重复 `20` 次；event alpha 固定为 `1.0`，不根据 2022 labels 调参。

## 主要结果

| 方法 | Weighted MAE | Weighted Bias | Weighted R2 | Delta vs primary |
|---|---:|---:|---:|---:|
| historical_xgboost | 4.3377 | +3.6052 | -0.6367 | +1.8354 |
| llm_only_pressure | 2.7175 | -0.3732 | 0.0794 | +0.2152 |
| zero_shot_llm_rule_tree | 2.6019 | -0.4072 | 0.1507 | +0.0997 |
| llm_rule_small_hist_calibrated_n100 | 3.0012 | +0.7405 | 0.0551 | +0.4989 |
| llm_rule_small_hist_calibrated_n500 | 2.7723 | +0.4627 | 0.1538 | +0.2700 |
| llm_rule_small_hist_calibrated_n1000 | 2.7905 | +0.5725 | 0.1445 | +0.2882 |
| llm_rule_full_history_calibrated | 2.7870 | +0.6068 | 0.1455 | +0.2848 |
| primary_hybrid_gated_adapter | 2.5023 | -0.0230 | 0.2480 | +0.0000 |

## 结论

- Zero-shot LLM rule tree wMAE `2.6019`，主方法 `2.5023`。
- Rule + 500 historical samples wMAE `2.7723`，比 zero-shot rule tree 更弱。
- Rule + 1000 historical samples wMAE `2.7905`。
- Full historical calibration wMAE `2.7870`。
- 最好的 small-data calibrated rule 是 `llm_rule_small_hist_calibrated_n500`，平均 wMAE `2.7723`。

这个结果把两条路线统一了：LLM 规则 + 少量历史校准是合理的 data-sparse/cold-start 中间方法，但在当前 NHTS 2022 event-shift 主任务上，完整历史模型提供的 household grounding 仍然更强。因此论文/汇报可以把它作为方法谱系中的桥梁，而不是和 hybrid adapter 对立。

## Artifacts

- Run metrics: `outputs\llm_rule_small_data_calibration\small_data_calibration_metrics_by_run.csv`
- Summary: `outputs\llm_rule_small_data_calibration\small_data_calibration_summary.csv`
- Method spectrum: `outputs\llm_rule_small_data_calibration\method_spectrum_metrics.csv`
- Figure: `outputs\llm_rule_small_data_calibration\llm_rule_small_data_calibration_spectrum.png`
- Leaf values: `outputs\llm_rule_small_data_calibration\small_data_leaf_values.csv`
