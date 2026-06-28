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
| Zero-shot LLM rule tree | LLM 直接生成规则树 | 2.7508 | 3.7834 | +0.5000 | 0.1712 |
| Zero-shot pseudo-label tree | LLM 规则标签蒸馏树 | 2.7767 | 3.8197 | +0.5000 | 0.1553 |
| LLM rule + 500-history calibration | LLM 规则 + 少量历史校准 | 2.7723 | 3.8226 | +0.4627 | 0.1538 |
| Global event prior | 全体家庭统一疫情折减 | 2.5531 | 3.6272 | +0.1229 | 0.2383 |
| **Hybrid gated adapter** | **主方法** | **2.5023** | **3.6038** | **-0.0230** | **0.2480** |

## 家庭颗粒度预测准确率

| 方法 | Exact rounded | Within 1 trip | Within 2 trips | Within 3 trips |
|---|---:|---:|---:|---:|
| Historical XGBoost | 5.8% | 11.8% | 26.5% | 41.9% |
| Pure LLM pressure | 14.2% | 26.2% | 51.0% | 68.0% |
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

### 3. 相比 zero-shot LLM rule tree

老师追问的“直接让 LLM 零样本构建 2022 决策树”已经作为补充 ablation 实现。这个 baseline 允许使用 household covariates 和 LLM event priors，但不允许使用 2022 `CNTTDHH` 标签训练 split threshold 或 leaf value。

结果是：

- Zero-shot LLM rule tree weighted MAE `2.7508`，weighted bias `+0.5000`。
- Zero-shot pseudo-label tree weighted MAE `2.7767`，weighted bias `+0.5000`。
- 主方法 weighted MAE `2.5023`，weighted bias `-0.0230`。

这说明 LLM 直接构建 rule tree 可以表达疫情机制方向，但它更像 qualitative belief tree，数值校准仍不如 historical household model + LLM event adapter。

### 4. 相比 LLM rule + small historical calibration

队友提出的“倒过来：先让 LLM 提取规则，再用少量历史数据校准”也已经作为中间 baseline 实现。我们用 2001/2009/2017 的少量历史样本校准 LLM-style routine leaves，然后再乘以 2022 LLM event pressure。

结果是：

- Rule + 100 historical samples weighted MAE `3.0012`。
- Rule + 500 historical samples weighted MAE `2.7723`，weighted bias `+0.4627`。
- Rule + 1000 historical samples weighted MAE `2.7905`。
- Full-history rule calibration weighted MAE `2.7870`。

这个结果说明这条路线是合理的 data-sparse/cold-start bridge，但在当前 2022 event-shift 任务里，历史 routine labels 会把 base demand 校得偏高；如果 event correction 不够强，就会重新产生高估。因此它适合作为方法谱系的一环，而不是替代主方法。

### 5. 相比 global event prior

Global event prior 是所有家庭使用同一个疫情折减。它表现很强，说明 2022 的主要信号确实是 event-level travel suppression。

但主方法仍有两个优势：

- Weighted MAE 更低：`2.5531 -> 2.5023`。
- Weighted bias 更接近 0：`+0.1229 -> -0.0230`。

所以正确说法不是“LLM cohort ranking 完全碾压 global rule”，而是：global rule 是强低成本 baseline；hybrid gated adapter 在此基础上提供更好的校准和 cohort-level refinement。

## 为什么不直接用 LLM 零样本构建 2022 决策树

这个方案已经作为补充 baseline 跑过，但不适合作为主方法。原因是决策树本质上需要标签来学习两个东西：第一，变量 split threshold；第二，叶节点的数值预测。如果没有 2022 `CNTTDHH` 标签，LLM 只能根据常识生成一个 belief tree，或者先生成 synthetic 2022 labels 再训练树。这样优化目标就会变成拟合 LLM 的假设，而不是拟合真实 NHTS 行为。

我们当前设计更稳：历史 XGBoost/CatBoost 用真实 NHTS 学 household baseline，LLM 只提供疫情事件机制先验，adapter 再把事件先验转成有限、可审计的修正。实验上 zero-shot rule tree 的 wMAE 是 `2.7508`、wBias 是 `+0.5000`，LLM rule + 500-history calibration 的 wMAE 是 `2.7723`、wBias 是 `+0.4627`，pure LLM pressure 的 wMAE 是 `2.7175`、wBias 是 `-0.3732`，而主方法 wMAE 是 `2.5023`、wBias 是 `-0.0230`。这说明 LLM 单独能给方向，少量历史校准能构成 cold-start bridge，但当前主任务仍需要更强的 historical household grounding + event adapter。

## 机制相关性负对照

我们还做了 irrelevant pseudo-event placebo：构造和真实 LLM pressure 边际分布相同、但语义上和疫情出行机制无关的 cohort score。结果显示：

- Best irrelevant ranked pseudo-event weighted MAE `2.6274`。
- Best irrelevant gated pseudo-event weighted MAE `2.5610`。
- 二者都弱于主方法 `2.5023`。

这说明改进不是任意 cohort-level shrinkage score 都能做到，而需要和 post-pandemic travel mechanism 对齐。

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
