# 10-Minute Final Defense Talk Plan

## Timing

| Section | Slides | Time | Main message |
|---|---:|---:|---|
| Opening and motivation | 1-3 | 1.2 min | 2022 is a post-pandemic behavioral shift, not normal year-to-year prediction. |
| Related work and problem setup | 4-5 | 1.2 min | Existing travel-demand models learn routine mobility; LLM work motivates event-generalizable priors. |
| Method design | 6-8 | 2.0 min | Traditional supervised baseline provides household grounding; LLM provides event priors; 2022 labels are evaluation-only. |
| Main evaluation | 9-12 | 2.0 min | Hybrid correction reduces wMAE and removes overprediction bias; bootstrap CI supports the gain. |
| Insights and robustness | 13-17 | 1.6 min | Event pressure behaves plausibly; random controls and global baseline define the claim boundary. |
| Behavior-system extension | 18-19 | 1.0 min | The project predicts trip generation, mode composition, mode-specific volumes, and purpose composition. |
| LLM scaling and conclusion | 20-22 | 1.0 min | Batch prompting uses LLM resources better; the LLM's key value is generalizing event mechanisms. |

## Key Sentences

1. 我们的问题不是普通回归，而是疫情后分布变化下的 household travel behavior prediction。
2. 传统监督模型负责家庭层面的 routine mobility，LLM 负责把疫情事件机制泛化到未标注 cohort。
3. 主实验不使用 2022 `CNTTDHH` 标签训练或调参；2022 标签只用于最终评估。
4. 主方法把 weighted MAE 从 `4.3377` 降到 `2.5023`，paired bootstrap MAE reduction 的 95% CI 为 `[1.7422, 1.9267]`。
5. 我们不把 LLM 说成直接预测器，而是 event-prior adapter。
6. 综合输出包括出行次数、方式结构、目的结构和各方式出行量；purpose 结果同时展示了当前方法边界。
7. Batch size 15 可把 1327 个 cohort 请求压缩为 89 个 batch prompts，减少 `93.29%` 请求。

## Discussion Boundaries

- Do not claim supervised NHTS SOTA.
- Do not claim purpose composition is already solved.
- Do not claim cohort-specific LLM ranking is the only source of improvement; global event correction is also strong.
- Emphasize the course-project contribution: a complete, leakage-controlled, event-aware mobility adaptation pipeline.
