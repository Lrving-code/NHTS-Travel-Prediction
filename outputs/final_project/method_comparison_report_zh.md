# 方法比较说明

## 比较对象

本项目现在有两个任务，但主次不同：

- 主任务：家庭每日出行次数预测，目标变量是 `CNTTDHH`。
- 辅助任务：家庭层面的出行方式结构预测，由 trip-level `TRPTRANS` 聚合得到。
- 同学参考资料里的 trip-level `TRPTRANS` 分类是另一个任务，不能直接和我们的 `CNTTDHH` 回归指标比较。

## 指标怎么读

- Weighted MAE：按 NHTS survey weight 加权后的平均绝对误差，越低越好。
- Weighted RMSE：加权均方根误差，对大误差更敏感，越低越好。
- Weighted Bias：加权有符号误差，越接近 0 越说明没有系统性高估/低估。
- Weighted R2：相对加权均值预测的拟合提升，越高越好。
- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。
- Weighted total variation：mode share 向量整体误差，公式是 `0.5 * sum(abs(pred - true))` 后再加权，越低越好。
- Transit-share weighted MAE：公共交通 share 这一项的加权绝对误差，越低越好。

## 出行次数预测结果

| 方法 | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | 相对历史预测 MAE 提升 |
|---|---:|---:|---:|---:|---:|
| Historical predictor | 4.3377 | 5.3169 | 3.6052 | -0.6367 | 0.00% |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 | 4.32% |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 | 41.85% |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 | 38.99% |
| LLM trip suppression | 2.4820 | 3.6601 | -0.5404 | 0.2244 | 42.78% |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 42.31% |

主任务结论：

- 最低 MAE 是 `llm_trip_suppression_a1p25`：weighted MAE 从 `4.3377` 降到 `2.4820`，降低 `42.78%`。
- 最好的 bias/R2 折中是 `gated_trip_suppression_a1_d0p15`：weighted bias `-0.0230`，weighted R2 `0.2480`。
- 家庭颗粒度上，gated correction exact hit `17.7%`，within 2 trips `54.5%`，within 3 trips `71.2%`。

## 出行方式结构结果

| 方法 | Weighted TV | Mean Share MAE | Dominant Accuracy | Transit MAE | Transit Bias |
|---|---:|---:|---:|---:|---:|
| llm_only_2017_mean_no_adapt | 0.2647 | 0.0882 | 0.8933 | 0.0567 | 0.0220 |
| llm_only_2017_mean_transit_a1.25 | 0.2570 | 0.0857 | 0.8933 | 0.0451 | 0.0087 |
| historical_mean_2017 | 0.2647 | 0.0882 | 0.8933 | 0.0567 | 0.0220 |
| historical_xgboost | 0.2008 | 0.0669 | 0.9029 | 0.0325 | 0.0035 |
| global_transit_avoidance_a1 | 0.1989 | 0.0663 | 0.9042 | 0.0289 | -0.0018 |
| llm_transit_avoidance_a1 | 0.1985 | 0.0662 | 0.9046 | 0.0269 | -0.0070 |

辅助任务结论：

- 纯 LLM-style correction 相比 2017 mean prior 有改善，但仍弱于 XGBoost。
- XGBoost + LLM transit prior 把 weighted TV 从 `0.2008` 降到 `0.1985`，总体提升较小。
- 但公共交通这一项改善明显：transit-share weighted MAE 从 `0.0325` 降到 `0.0269`。
- LLM-only corrected transit MAE 是 `0.0451`，说明 LLM 知道方向，但需要历史预测器提供 household-specific baseline。

## 最终口径

不要把这个项目讲成“LLM 替代传统模型”。更准确的说法是：历史预测器学习 routine mobility，LLM 提供疫情事件先验，两者结合后能在不使用 2022 标签训练的前提下修正 post-pandemic distribution shift。