# 中文汇报讲稿

## 核心主线

我们把 2022 NHTS household travel behavior prediction 定义成 event-driven temporal adaptation 问题。历史模型负责学习 routine mobility，LLM event priors 负责表达疫情带来的 remote work、transit avoidance、delivery substitution 等事件机制。

现在的项目不是 pure LLM 预测，也不是单纯 XGBoost。主方法是 hybrid gated adapter：先用历史 NHTS 学一个 household baseline，再用 LLM 生成的 event pressure 做事件修正。

可以用一句公式概括：

```text
2022 prediction = historical household baseline × event correction
```

LLM 输出的是结构化事件先验，不直接输出 `CNTTDHH`。

## 一分钟版本

传统历史预测器会明显高估 2022 年家庭出行次数。主口径使用固定 no-label gated rule，weighted MAE 从 `4.3377` 降到 `2.5023`，weighted bias 变成 `-0.0230`，几乎消除了系统性高估。LLM-only pressure baseline 的 weighted MAE 是 `2.7175`。这说明 LLM 不是直接预测 trip count，而是提供疫情事件语义先验，再去修正历史 routine-mobility predictor。

## 10 分钟主讲路径

正式模板 PPT 的前 23 页是主讲路径。END 之后的 B1-B6 是 backup Q&A，不主动讲，只有在讨论环节老师追问时跳转。

建议时间分配：

- 背景与研究问题：1.5 分钟。
- 数据、技术路线和 LLM 事件先验：2 分钟。
- 方法比较和主结果：2.5 分钟。
- 稳健性、异质性、多目标和 mode/purpose 扩展：2 分钟。
- 总结与边界：2 分钟。

如果时间不够，优先保留：研究问题、技术路线、方法比较、主结果、稳健性边界和总结；error distribution、subgroup、mode/purpose 可以快速带过。

## 各方法怎么比较

- 传统 XGBoost / CatBoost：优势是有 household grounding，问题是不知道 2022 疫情机制变化。普通 historical predictor 的 wMAE 是 `4.3377`，wBias 是 `3.6052`，明显高估。
- 更强表格 baseline 仍然不够。最强非 LLM baseline `catboost_gpu` 的 wMAE 是 `4.2196`，wBias 是 `3.4873`。
- Pure LLM pressure：优势是知道疫情后出行下降方向，问题是缺少家庭数值基线，wMAE `2.7175`，wBias `-0.3732`。
- Zero-shot LLM rule tree：已经作为补充 ablation 跑过，wMAE `2.6019`，wBias `-0.4072`，能表达机制方向但校准弱于 hybrid。
- LLM rule + 500 historical calibration：这是队友路线的 bridge baseline，wMAE `2.7723`，wBias `0.4627`，说明少量历史校准合理但会重新高估 2022。
- Global event prior：低成本且很强，wMAE `2.5223`，wBias `-0.7477`。这说明主信号确实是 event-level suppression，不能夸大成 cohort LLM ranking 独自贡献全部提升。
- 我们的 hybrid gated：wMAE `2.5023`，wBias `-0.0230`，wR2 `0.2480`。优势是同时保留 household baseline、event semantics 和 near-zero bias。

这就是主方法优势：传统模型有家庭基线但没有疫情机制，pure LLM 有事件方向但校准弱，global rule 太粗；hybrid gated 把三者的优点组合起来。

## 指标解释

- Weighted MAE/RMSE：加权后的出行次数误差。
- Weighted Bias：系统性高估或低估，越接近 0 越好。
- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。
- Weighted total variation：家庭 mode-share 向量整体误差。
- Transit-share weighted MAE：公共交通占比这一项的误差。

## Mode Extension 怎么讲

出行方式结构是第二个家庭层面的预测输出。XGBoost + LLM 把 transit-share weighted MAE 从 `0.0325` 降到 `0.0269`，说明 LLM 的 transit avoidance prior 对公共交通这一项有帮助；但总体 mode composition 改善不大，所以汇报时要把它讲成 mode-structure 维度，而不是夸大成主要增益。

进一步的 planning 输出是 mode-specific trip volume：用预测总出行次数乘以预测 mode share，得到各方式出行量。这个比单独报 mode share 更接近城市规划里的需求评估。

## 边界和不能过度声称的地方

- 不能说 LLM 直接预测 household trips。
- 不能说我们识别了 COVID 的 causal effect；应该说 causal guardrails 和 mechanism proxy。
- purpose composition 目前是探索性输出，不作为主贡献。
- global event prior 很强，所以 contribution 要讲成 event-level adaptation + auditable cohort refinement。
- 外部验证包括 ACS mechanism-level 证据和 PSRC household-level recovery-transfer microdata。PSRC 的 MAE 增益很小，但 bias 改善明显；BTS 仍作为 device trip count 与 NHTS `CNTTDHH` 不可直接对齐的 guardrail。

## 可能被问到的问题

- 为什么有两套指标？因为 trip generation 是 count regression，mode composition 是 share-vector prediction。
- 这是 pure LLM 吗？不是。纯 LLM-style correction 比 XGBoost + LLM 弱，说明 LLM 适合作为 event-prior adapter。
- 为什么不直接让 LLM 零样本构建 2022 决策树？我们已经把它作为 ablation 跑过，zero-shot rule tree wMAE `2.6019`，仍弱于主方法。原因是没有标签时 split threshold 和 leaf value 缺少数据校准，更像 LLM belief tree。
- 有没有用 2022 标签训练？主实验没有。2022 `CNTTDHH` 只在最终 evaluation 中使用。
- 结果够不够做大作业？够，因为我们有完整数据链路、强 baseline、LLM event prior、无标签修正、稳健性检验、mode 扩展和 10 分钟汇报材料。

## Backup 页怎么用

- B1：老师问“为什么不直接用 LLM 构树”。
- B2：老师问“有没有用 2022 标签或泄漏”。
- B3：老师问“global rule 已经很强，LLM 到底贡献什么”。
- B4：老师问“这和最新 LLM mobility/foundation model 工作是什么关系”。
- B5：老师问“mode/purpose 是不是已经做完了”。
- B6：老师问“Weighted MAE、bias、within-k accuracy 怎么解释”。
- B7：老师问“有没有 NHTS 之外的外部验证”。
