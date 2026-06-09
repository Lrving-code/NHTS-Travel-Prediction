# 方法比较说明

## 比较对象

本项目按家庭颗粒度组织为两个行为预测输出：

- Trip generation：家庭每日出行次数预测，目标变量是 `CNTTDHH`。
- Mode composition：家庭层面的出行方式结构预测，由 trip-level `TRPTRANS` 聚合得到。
- Mode-specific trip count：用预测总出行次数乘以预测 mode share，得到各方式出行次数。

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
| Historical mean | 5.5116 | 6.1251 | 4.4995 | -1.1722 | -27.06% |
| Historical predictor | 4.3377 | 5.3169 | 3.6052 | -0.6367 | 0.00% |
| Historical trend shift | 4.1504 | 5.1463 | 3.3485 | -0.5334 | 4.32% |
| LLM-only pressure | 2.7175 | 3.9876 | -0.3732 | 0.0794 | 37.35% |
| Global event prior | 2.5223 | 3.7432 | -0.7477 | 0.1888 | 41.85% |
| Random prior control | 2.6466 | 3.8807 | -0.6368 | 0.1280 | 38.99% |
| LLM trip suppression | 2.4820 | 3.6601 | -0.5404 | 0.2244 | 42.78% |
| Gated LLM correction | 2.5023 | 3.6038 | -0.0230 | 0.2480 | 42.31% |

Trip-generation 结论：

- 主口径使用固定 no-label gated rule：`gated_trip_suppression_a1_d0p15`，weighted MAE `2.5023`，weighted bias `-0.0230`，weighted R2 `0.2480`。
- LLM-only pressure baseline 的 weighted MAE 是 `2.7175`，说明 LLM 事件先验能给出方向，但需要 historical household predictor 提供个体化 baseline。
- 最低 MAE 是 sensitivity row `llm_trip_suppression_a1p25`：weighted MAE 从 `4.3377` 降到 `2.4820`，降低 `42.78%`；它不作为严格 no-label 主方法的参数选择依据。
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

Mode-composition 结论：

- 纯 LLM-style correction 相比 2017 mean prior 有改善，但仍弱于 XGBoost。
- XGBoost + LLM transit prior 把 weighted TV 从 `0.2008` 降到 `0.1985`，总体提升较小。
- 但公共交通这一项改善明显：transit-share weighted MAE 从 `0.0325` 降到 `0.0269`。
- LLM-only corrected transit MAE 是 `0.0451`，说明 LLM 知道方向，但需要历史预测器提供 household-specific baseline。

## 最终口径

不要把这个项目讲成“LLM 替代传统模型”。更准确的说法是：历史预测器学习 routine mobility，LLM 提供疫情事件先验，两者结合后能在不使用 2022 标签训练的前提下修正 post-pandemic distribution shift。