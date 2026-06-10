# 方法比较与主方法优势简报

## 研究场景

本项目预测的是 2022 年 NHTS 疫情后家庭层面的出行行为。核心难点是：历史年份学到的 routine mobility 在 2022 出现系统性失效，普通模型会明显高估家庭出行次数。

我们的主方法是 **Hybrid gated LLM event adapter**：

```text
2022 prediction = historical household baseline * event correction
```

其中，历史模型负责学习 household-level routine baseline，LLM 负责生成 trip suppression、remote work、transit avoidance 等 post-pandemic event priors。LLM 不直接预测 `CNTTDHH`。

## Trip-count 主结果

| 方法 | 角色 | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 |
|---|---|---:|---:|---:|---:|
| Historical XGBoost | 普通历史表格模型 | 4.3377 | 5.3169 | +3.6052 | -0.6367 |
| CatBoost GPU | 更强非 LLM baseline | 4.2196 | 5.2095 | +3.4873 | -0.5712 |
| Historical trend shift | 简单趋势修正 | 4.1504 | 5.1463 | +3.3485 | -0.5334 |
| Pure LLM pressure | 只用 LLM 事件压力 | 2.7175 | 3.9876 | -0.3732 | 0.0794 |
| Global event prior | 全体家庭统一疫情折减 | 2.5531 | 3.6272 | +0.1229 | 0.2383 |
| LLM trip suppression | 最低 MAE sensitivity row | 2.4820 | 3.6601 | -0.5404 | 0.2244 |
| **Hybrid gated adapter** | **主方法** | **2.5023** | **3.6038** | **-0.0230** | **0.2480** |

## 家庭颗粒度预测准确率

| 方法 | Exact rounded | Within 1 trip | Within 2 trips | Within 3 trips |
|---|---:|---:|---:|---:|
| Historical XGBoost | 5.8% | 11.8% | 26.5% | 41.9% |
| Pure LLM pressure | 14.2% | 26.2% | 51.0% | 68.0% |
| LLM trip suppression | 15.6% | 30.2% | 56.1% | 72.3% |
| **Hybrid gated adapter** | **17.7%** | **29.8%** | **54.5%** | **71.2%** |

加权口径下，主方法 within 2 trips accuracy 是 `54.0%`，within 3 trips accuracy 是 `70.8%`。也就是说，超过一半家庭的预测误差在 2 次出行以内，约七成家庭在 3 次出行以内。

## 我们方法相对其他方法的优势

### 1. 相比传统表格模型

传统 XGBoost 和 CatBoost 能学习家庭差异，但不知道 2022 疫情后机制变化，所以严重高估出行。Historical XGBoost 的 weighted bias 是 `+3.6052`，CatBoost GPU 的 weighted bias 仍有 `+3.4873`。

主方法相对 Historical XGBoost：

- Weighted MAE 从 `4.3377` 降到 `2.5023`，降低 `42.31%`。
- 绝对 weighted bias 从 `3.6052` 降到 `0.0230`，降低 `99.36%`。
- Weighted R2 从 `-0.6367` 提高到 `0.2480`。

主方法相对最强非 LLM baseline CatBoost GPU：

- Weighted MAE 低 `40.70%`。
- 说明单纯增强表格模型容量仍不能解决 post-pandemic mechanism shift。

### 2. 相比 pure LLM prediction

Pure LLM pressure 能抓到疫情后出行下降方向，但缺少 household-level numerical baseline，容易校准不稳。

主方法相对 Pure LLM pressure：

- Weighted MAE 从 `2.7175` 降到 `2.5023`，进一步降低 `7.92%`。
- 绝对 weighted bias 从 `0.3732` 降到 `0.0230`，降低 `93.83%`。
- Weighted R2 从 `0.0794` 提高到 `0.2480`。

这说明 LLM 的最佳角色不是直接做 trip-count predictor，而是作为 event-prior generator。

### 3. 相比 global event prior

Global event prior 是所有家庭使用同一个疫情折减。它表现很强，说明 2022 的主要信号确实是 event-level travel suppression。

但主方法仍有两个优势：

- Weighted MAE 更低：`2.5531 -> 2.5023`。
- Weighted bias 更接近 0：`+0.1229 -> -0.0230`。

所以正确说法不是“LLM cohort ranking 完全碾压 global rule”，而是：global rule 是强低成本 baseline；hybrid gated adapter 在此基础上提供更好的校准和 cohort-level refinement。

### 4. 相比最低 MAE sensitivity row

`llm_trip_suppression_a1p25` 的 Weighted MAE 是 `2.4820`，略低于主方法 `2.5023`，但它的 weighted bias 是 `-0.5404`，说明整体压得偏低。

主方法选择 gated adapter，是因为它更适合 planning：

- MAE 接近最低。
- Bias 几乎为 0。
- Weighted R2 更高。
- 总体出行量估计更可信。

## 多输出行为结果

| 目标 | 指标 | 传统方法 | 事件适配方法 | 改善 |
|---|---|---:|---:|---:|
| Trip generation | Weighted MAE | 4.3377 | 2.5023 | 42.31% |
| Trip generation calibration | Abs weighted bias | 3.6052 | 0.0230 | 99.36% |
| Mode composition | Weighted total variation | 0.2008 | 0.1985 | 1.13% |
| Sustainable mode signal | Transit-share weighted MAE | 0.0325 | 0.0269 | 17.35% |
| Mode-specific trip volume | Mode-trip MAE | 4.8467 | 3.2802 | 32.32% |

## 一句话结论

我们的优势在于把传统模型、LLM 和可控 adapter 的优势组合起来：传统模型提供 household grounding，LLM 提供 post-pandemic event semantics，gated adapter 控制数值校准。因此在 2022 这种事件冲击场景下，主方法同时实现了更低误差、更小系统偏差、更好的可解释性和更强的可审计性。
