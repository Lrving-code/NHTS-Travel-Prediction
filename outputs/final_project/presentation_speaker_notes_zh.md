# 中文汇报讲稿

## 核心信息

我们把 2022 NHTS household travel behavior prediction 定义成 event-driven temporal adaptation 问题。历史模型负责学习 routine mobility，LLM event priors 负责表达疫情带来的 remote work、transit avoidance、delivery substitution 等事件机制。

## 一分钟版本

传统历史预测器会明显高估 2022 年家庭出行次数。主口径使用固定 no-label gated rule，weighted MAE 从 `4.3377` 降到 `2.5023`，weighted bias 变成 `-0.0230`，几乎消除了系统性高估。LLM-only pressure baseline 的 weighted MAE 是 `2.7175`，最低 MAE 的 hybrid sensitivity row 可以到 `2.4820`。这说明 LLM 不是直接预测 trip count，而是提供疫情事件语义先验，再去修正历史 routine-mobility predictor。

## 指标解释

- Weighted MAE/RMSE：加权后的出行次数误差。
- Weighted Bias：系统性高估或低估，越接近 0 越好。
- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。
- Weighted total variation：家庭 mode-share 向量整体误差。
- Transit-share weighted MAE：公共交通占比这一项的误差。

## Mode Extension 怎么讲

出行方式结构是第二个家庭层面的预测输出。XGBoost + LLM 把 transit-share weighted MAE 从 `0.0325` 降到 `0.0269`，说明 LLM 的 transit avoidance prior 对公共交通这一项有帮助；但总体 mode composition 改善不大，所以汇报时要把它讲成 mode-structure 维度，而不是夸大成主要增益。

## 可能被问到的问题

- 为什么有两套指标？因为 trip generation 是 count regression，mode composition 是 share-vector prediction。
- 这是 pure LLM 吗？不是。纯 LLM-style correction 比 XGBoost + LLM 弱，说明 LLM 适合作为 event-prior adapter。
- 有没有用 2022 标签训练？主实验没有。2022 `CNTTDHH` 只在最终 evaluation 中使用。