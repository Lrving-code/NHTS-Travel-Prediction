# 中文汇报讲稿

## 题目口径

PPT 新题目是“面向时序迁移的家庭出行预测：模拟选择器与大模型修正框架”，英文题目是 “LLM-Guided Temporal Adaptation for Household Mobility Prediction”。

这个题目避免把项目讲成疫情特例。2022 后疫情恢复期只是一个 target-year temporal shift case；真正的问题是：当目标年出现历史数据没有覆盖的情境变化时，如何让模型具备时序迁移能力。

两方工作可以统一成一条技术路线：一类模块做模拟预测选择器，在多个候选预测或规则之间选择更可信的输出；另一类模块做 LLM-corrected XGBoost，用大模型抽取的目标年情境先验去修正历史模型。汇报时不要讲成两套互相竞争的方案，而是讲成从“选择候选预测”到“修正基础预测器”的 temporal-adaptation framework。

## 核心主线

我们把 2022 NHTS household travel behavior prediction 定义成 target-year temporal adaptation 问题。历史模型负责学习 routine mobility，LLM context/event priors 负责表达目标年出现的 remote work、transit avoidance、delivery substitution 等情境机制。

现在的项目不是 pure LLM 预测，也不是单纯 XGBoost。主方法是 hybrid gated adapter：先用历史 NHTS 学一个 household baseline，再用 LLM 生成的 event pressure 做事件修正。

## 33 页完整版本怎么串两条方向

完整 PPT 现在按“一个问题、两个分支”组织。第 1-5 页先讲目标年时序迁移和 household behavior system；第 6 页提前给出方法谱系，把 data-only、pure LLM、LLM rule selector、rule + history、LLM-corrected XGBoost 放到同一条链上；第 7 页再展开双分支技术路线。

两条方向的关系可以这样讲：

```text
同一个问题：目标年情境变化下如何预测家庭出行行为
  ├─ selector branch：冷启动、数据稀疏、字段缺失时，用 LLM 规则和置信度选择快速模拟预测
  └─ correction branch：有历史 NHTS 标签但目标年变了时，用 LLM 情境先验修正 XGBoost
```

因此不要说“这是两套方案”。更准确的说法是：selector branch 解决没有足够历史标签时的预测选择问题，LLM-corrected XGBoost 解决有历史标签但发生时序迁移时的数值校准问题。正式汇报应先讲第 6 页方法谱系，再讲第 7 页技术路线，这样老师会更容易理解两部分为什么属于同一个框架。

可以用一句公式概括：

```text
2022 prediction = historical household baseline × event correction
```

LLM 输出的是结构化事件先验，不直接输出 `CNTTDHH`。

## 一分钟版本

传统历史预测器会明显高估 2022 年家庭出行次数。主口径使用固定 no-label gated rule，weighted MAE 从 `4.3377` 降到 `2.5023`，weighted bias 变成 `-0.0230`，几乎消除了系统性高估。LLM-only pressure baseline 的 weighted MAE 是 `2.7175`。这说明 LLM 不是直接预测 trip count，而是提供目标年情境语义先验，再去修正历史 routine-mobility predictor。

## 10 分钟主讲路径

正式模板 PPT 的前 24 页是主讲路径。END 之后的 B1-B8 是 backup Q&A，不主动讲，只有在讨论环节老师追问时跳转。

建议时间分配：

- 背景与研究问题：1.5 分钟。
- 数据、技术路线和 LLM 事件先验：2 分钟。
- 方法比较和主结果：2.5 分钟。
- 稳健性、异质性、多目标、mode/purpose 和方法谱系：2 分钟。
- 总结与边界：2 分钟。

如果时间不够，优先保留：研究问题、技术路线、方法比较、主结果、稳健性边界和总结；error distribution、subgroup、mode/purpose 可以快速带过。

## 方法谱系怎么讲

这里要把“LLM 知识蒸馏 + 低置信度回退”的模拟预测选择器路线融合进来，而不是讲成两套互相竞争的方案。推荐说法是：

```text
data-only fitting -> pure LLM -> LLM rule tree -> LLM rule + small historical calibration -> historical model + LLM event adapter
```

这条链条说明：纯 LLM 和 LLM rule tree 都能提供情境方向和可解释规则，但缺少真实 NHTS 数值校准；少量历史校准是 data-sparse/cold-start 场景的合理桥梁；当前 2022 NHTS 主任务有多年历史调查数据，因此最强方法是保留 historical household grounding，再用 LLM context prior 修正目标年情境变化。

## LLM 蒸馏与调用效率怎么讲

推荐口径是：纯 LLM 逐户调用有两个问题，一是成本和延迟随 household 数量线性增加，二是缺少 NHTS 数值校准。我们的做法不是让 LLM 逐户报 `CNTTDHH`，而是先把家庭聚合成 cohort，再用 batch size 15 生成结构化 event priors。主实验一共从 1,327 个 cohorts 压缩到 89 个 prompts，相比 7,893 个 household 逐户调用减少约 98.9% 请求。

蒸馏后的先验进入固定 adapter，所以正式预测阶段不再调用 LLM。低置信度回退可以作为未来线上部署扩展：如果某个 cohort 的事件先验不稳定，可以再触发 LLM 或改用 global pressure fallback。但主结果为了保持 no-label 和可复现，使用的是固定规则，不把在线 LLM fallback 混入 2022 evaluation。也不要直接使用参考 PPT 中的 75%/25% 调用比例，除非后续我们真的按这个阈值跑出可复现实验。

## 各方法怎么比较

- 传统 XGBoost / CatBoost：优势是有 household grounding，问题是不知道目标年情境机制变化。普通 historical predictor 的 wMAE 是 `4.3377`，wBias 是 `3.6052`，明显高估。
- 更强表格 baseline 仍然不够。最强非 LLM baseline `catboost_gpu` 的 wMAE 是 `4.2196`，wBias 是 `3.4873`。
- Pure LLM pressure：优势是知道目标年出行下降方向，问题是缺少家庭数值基线，wMAE `2.7175`，wBias `-0.3732`。
- Zero-shot LLM rule tree：已经作为补充 ablation 跑过，wMAE `2.6019`，wBias `-0.4072`，能表达机制方向但校准弱于 hybrid。
- LLM rule + 500 historical calibration：这是队友路线的 bridge baseline，wMAE `2.7723`，wBias `0.4627`，说明少量历史校准合理但会重新高估 2022。
- Global event prior：低成本且很强，wMAE `2.5223`，wBias `-0.7477`。这说明主信号确实是 event-level suppression，不能夸大成 cohort LLM ranking 独自贡献全部提升。
- 我们的 hybrid gated：wMAE `2.5023`，wBias `-0.0230`，wR2 `0.2480`。优势是同时保留 household baseline、event semantics 和 near-zero bias。
- 如果被追问 global prior 已经很强，补充 same-alpha decomposition：primary gated 比 `global_trip_suppression_a1` 低 `0.0508` wMAE，在 `77.8%` 的 subgroup cells 里更好；但它只在 `17.5%` 的加权家庭上使用 cohort-specific pressure，所以这是 selective refinement，不是说 cohort ranking 解释全部提升。

这就是主方法优势：传统模型有家庭基线但没有目标年机制，pure LLM 有情境方向但校准弱，global rule 太粗；hybrid gated 把三者的优点组合起来。

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
- 外部验证包括 ACS mechanism-level 证据和 PSRC household-level direct pre/post microdata。PSRC 2017+2019->2023 的 wMAE 从 `3.4271` 到 `3.2498`，wBias 从 `+1.0532` 到 `+0.2605`；BTS 仍作为 device trip count 与 household survey target 不可直接对齐的 negative guardrail。

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
- B8：老师问“规则蒸馏、低置信 LLM 回退和我们现在的方法是什么关系”。
- B9：老师继续追问“cohort-specific LLM prior 比 global prior 多贡献了多少”，用 same-alpha decomposition 图回答。
