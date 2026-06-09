# 中文汇报讲稿

## 核心信息

我们研究的是 2022 年疫情后分布变化下的 household mobility prediction。主任务是预测一个家庭在 travel day 的总出行次数 `CNTTDHH`。

## 一分钟版本

传统历史预测器会明显高估 2022 年家庭出行次数。加入 LLM trip-suppression event prior 后，weighted MAE 从 `4.3377` 降到 `2.4820`。这里 LLM 不是直接预测 trip count，而是提供疫情事件语义先验，再去修正历史 routine-mobility predictor。

## 指标解释

- Weighted MAE/RMSE：加权后的出行次数误差。
- Weighted Bias：系统性高估或低估，越接近 0 越好。
- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。
- Weighted total variation：家庭 mode-share 向量整体误差。
- Transit-share weighted MAE：公共交通占比这一项的误差。

## Mode Extension 怎么讲

出行方式结构是辅助实验。XGBoost + LLM 把 transit-share weighted MAE 从 `0.0325` 降到 `0.0269`，说明 LLM 的 transit avoidance prior 对公共交通这一项有帮助。但总体 mode composition 改善不大，所以它应该作为 backup 或 supplementary evidence。

## 可能被问到的问题

- 为什么不能和同学的 accuracy 直接比？因为同学那版更像 trip-level `TRPTRANS` 分类，我们主任务是 household-level `CNTTDHH` 回归。
- 这是 pure LLM 吗？不是。纯 LLM-style correction 比 XGBoost + LLM 弱，说明 LLM 适合作为 event-prior adapter。
- 有没有用 2022 标签训练？主实验没有。2022 `CNTTDHH` 只在最终 evaluation 中使用。