# Zero-Shot LLM Rule-Tree Baseline

## 这个实验回答什么问题

这个 ablation 回答老师/审稿人可能追问的问题：为什么不直接让 LLM 在零样本情况下构建 2022 年决策树？

## 简短回答

可以构建 zero-shot rule tree，但它更像 qualitative prior，而不是经过真实 NHTS outcome 校准的统计模型。它能表达目标年事件方向，但 split threshold 和 leaf value 并不是从真实标签中估计出来的。主方法更稳健，因为数值上的 routine demand 来自历史 NHTS 标签，LLM 只负责提供 event semantics。

## 结果

- Historical XGBoost wMAE: `4.3377`.
- Zero-shot rule tree wMAE: `2.7508`; weighted bias: `+0.5000`.
- Pseudo-label decision tree wMAE: `2.7767`; weighted bias: `+0.5000`.
- LLM-only pressure wMAE: `2.7175`; weighted bias: `-0.3732`.
- Primary hybrid gated wMAE: `2.5023`; weighted bias: `-0.0230`.
- Zero-shot rule tree is `+0.2485` wMAE relative to the primary method.
- Pseudo-label decision tree is `+0.2744` wMAE relative to the primary method.

## 结论

这个 ablation 支持当前设计：LLM 不应该直接替代监督 household model。LLM 适合作为 event-generalization module，但直接 zero-shot tree 缺少数据估计的 threshold 和校准后的 leaf value。在论文或 PPT 中，它可以解释为什么我们采用 hybrid adapter，而不是 pure LLM rule induction。

## 完整指标表

| Method | Family | wMAE | wRMSE | wBias | wR2 | Exact | Within 2 | Delta vs primary |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| historical_xgboost | traditional_supervised | 4.3377 | 5.3169 | +3.6052 | -0.6367 | 0.058 | 0.349 | +1.8354 |
| zero_shot_llm_rule_tree | zero_shot_llm_tree | 2.7508 | 3.7834 | +0.5000 | 0.1712 | 0.133 | 0.580 | +0.2485 |
| zero_shot_pseudo_label_tree | zero_shot_llm_tree | 2.7767 | 3.8197 | +0.5000 | 0.1553 | 0.132 | 0.569 | +0.2744 |
| llm_only_pressure | llm_only_control | 2.7175 | 3.9876 | -0.3732 | 0.0794 | 0.142 | 0.614 | +0.2152 |
| primary_hybrid_gated_adapter | main_method | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 0.177 | 0.656 | +0.0000 |

## Artifacts

- Figure: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_rule_tree_metric_comparison.png`
- Metrics: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_metrics.csv`
- Rule traces: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_predictions.csv`
- Rule specification: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_llm_rule_tree_spec.json`
- Markdown rule specification: `outputs\zero_shot_llm_rule_tree_baseline\zero_shot_rule_tree_spec.md`
