# 中文汇报讲稿

## 核心主线

本项目预测的是 2022 年 NHTS 家庭层面的出行行为。难点不在于普通回归，而在于 2022 是疫情后恢复期，历史年份学到的 routine mobility 会系统性高估出行次数。

我们的做法是把任务拆成两部分：历史模型学习“正常情况下这个家庭大概会出行多少”，LLM 根据 cohort profile 和 2022 疫情语境生成 event pressure，再用固定规则把 event pressure 转成出行折减系数。

最终预测形式可以概括为：

```text
2022 prediction = historical household baseline × event correction
```

LLM 不直接输出 trip count，也不替代 XGBoost。它输出的是 `s_event`，也就是疫情对某类家庭出行的压低程度。

## 一分钟版本

普通 XGBoost 使用历史 NHTS 学到了 household-level 差异，但会把 2022 当作常态延续，因此 weighted MAE 为 `4.3377`，weighted bias 为 `+3.6052`，明显高估出行。

加入 LLM event pressure 后，主口径 `gated_trip_suppression_a1_d0p15` 的 weighted MAE 降到 `2.5023`，相对普通 XGBoost 降低 `42.31%`；weighted bias 变为 `-0.0230`，基本消除了系统性高估。

对比结果也说明 pure LLM 不是最优方案。LLM-only pressure 的 weighted MAE 是 `2.7175`，弱于 hybrid 方法。也就是说，LLM 提供疫情事件方向，历史模型提供家庭数值基线，二者结合才稳定。

## 图表讲法

- 误差-偏差图：横轴是系统性偏差，纵轴是 weighted MAE，越靠左下越好。Hybrid 位于低误差、低偏差区域。
- 误差分布图：历史模型误差整体偏正，说明它系统性高估 2022 出行；事件修正后误差更靠近 0。
- 容忍曲线：在允许 1、2、3 次 trip 误差时，hybrid 方法覆盖更多家庭。
- 异质性图：LLM pressure 与 remote work、transit avoidance 等机制有一致关系，说明它不是随意生成的装饰变量。
- 稳健性图：真实 LLM pressure 优于随机置换；但 global pressure 也很强，所以贡献应表述为疫情事件层面修正，cohort ranking 是增量信息。

## Mode Extension 怎么讲

出行方式结构是扩展任务。总体 mode-share 向量的提升不大，但 transit-share weighted MAE 从 `0.0325` 降到 `0.0269`，改善 `17.4%`。这与疫情期间公共交通规避机制一致。

因此方式结构部分定位为“事件先验在 transit 维度有合理信号”，不表述为完整 mode-choice model 已经解决。

## 可能被问到的问题

- 为什么 LLM 比传统方法有用？传统模型只看到家庭属性和历史规律，缺少 2022 疫情事件机制；LLM 可以把 remote work、transit avoidance、delivery substitution 等语境转成结构化先验。
- 为什么不能只用 LLM？纯 LLM pressure 缺少家庭层面的数值基线，误差高于 hybrid。
- 有没有用 2022 标签训练？主实验没有。2022 `CNTTDHH` 只用于最终 evaluation。
- global pressure 已经很强，LLM 还有意义吗？有，但需要明确：主信号是疫情层面的出行折减，LLM cohort ranking 提供增量证据，不是唯一来源。
- 现在还能怎么扩展？可以接入政策、新闻、mobility index 等外部事件数据，让 event context 在预测前就固定下来。
