# 0611 Refined 10 分钟讲稿

对应 PPT：

`outputs/final_project/NHTS_Temporal_Adaptation_0611_Refined_10min.pptx`

## 一句话主线

我们研究的不是让 LLM 直接预测出行，而是让 LLM 帮助传统出行模型适应目标年情境变化：在冷启动场景下，LLM 提供规则和预测选择器；在有历史 NHTS 数据的场景下，LLM 提供情境先验修正 XGBoost。

## 时间分配

| 页码 | 内容 | 时间 |
|---:|---|---:|
| 1 | 标题与核心问题 | 0:30 |
| 2 | 汇报路径 | 0:30 |
| 3 | 动机：为什么目标年迁移难 | 1:00 |
| 4 | 问题定义：household behavior system | 0:45 |
| 5 | 方法谱系：从 selector 到 corrector | 1:00 |
| 6 | 统一框架 | 1:00 |
| 7-8 | 数据、no-label setup、LLM 情境先验 | 1:15 |
| 9-10 | baseline 与主结果 | 1:45 |
| 11 | 统计验证与稳健性 | 0:50 |
| 12-13 | selector 分支和 LLM 角色 | 1:00 |
| 14-15 | 多输出与多目标规划 | 0:50 |
| 16-17 | 边界与总结 | 0:50 |
| 18 | backup map，不主动讲 | Q&A |

## 逐页讲法

### 1. Title

开场不要先讲疫情。先讲：

> 我们关注的是目标年时序迁移。也就是说，历史数据能学到常规出行规律，但目标年可能出现新的社会情境，导致直接迁移失效。

然后点出两个模块：

> 因此我们把方法分成两类能力：一是模拟预测选择器，处理冷启动和数据稀疏；二是 LLM-corrected XGBoost，处理有历史数据但目标年发生变化的场景。

### 2. Roadmap

强调这版不是两套方案拼接：

> 这份汇报按一个问题组织：目标年迁移下如何预测家庭出行行为。selector 和 correction 是同一框架里的两个模块。

### 3. Motivation

讲清楚三角关系：

- 传统模型：数值校准强，但无法理解目标年新机制。
- 纯 LLM：理解情境方向，但缺少 NHTS household-level 数值锚点。
- 融合：LLM 不直接报数，而是指导选择和修正。

关键句：

> XGBoost 的问题是 overpredict；LLM-only 的问题是 calibration weak；hybrid 的价值是同时有 household grounding 和 context adaptation。

### 4. Problem Definition

避免被问“到底预测什么”：

> 我们最终预测的是 household travel behavior system，不只是一项分类任务。主结果是 trip generation，方式和目的作为行为系统扩展。

### 5. Method Spectrum

这是全篇最重要的桥：

> 左边是 data-only，右边是 hybrid correction。中间的 LLM rule selector 不是被否定，而是作为 cold-start / sparse-data 分支存在。

如果老师问“你们两套方法是不是不一致”，回答：

> 它们对应不同数据条件。没有历史标签时，用 selector；有历史 NHTS 标签但目标年变了时，用 LLM-corrected XGBoost。

### 6. Unified Framework

讲两个分支：

> 同一个 household profile 进入两个模块。冷启动分支把 LLM 知识蒸馏成规则并做置信度选择；历史数据分支先训练 XGBoost baseline，再用 LLM context prior 做无标签修正。

### 7. Data and No-Label Setup

重点讲可信度：

> 训练只用历史标签，目标年标签只在最后评估使用。LLM 输入不含 CNTTDHH、WTHHFIN、HOUSEID，避免把答案泄露给模型。

### 8. LLM Context Prior

一句话：

> LLM 输出的是机制变量，例如 trip suppression、remote work、transit avoidance，不是直接输出家庭出行次数。

### 9. Baselines

讲比较公平性：

> 我们不是只和普通 XGBoost 比，也和 pure LLM、zero-shot rule tree、rule+history、global prior 等方法比较。

不要花太久解释每一行，只点出三类：

- data-only
- LLM-only / selector
- hybrid correction

### 10. Main Results

核心数字：

- ordinary XGBoost wMAE `4.338`，wBias `+3.605`
- LLM-only pressure wMAE `2.718`，wBias `-0.373`
- zero-shot rule tree wMAE `2.602`，wBias `-0.407`
- hybrid gated wMAE `2.502`，wBias `-0.023`

结论：

> LLM 有方向感，历史模型有数值锚点；hybrid gated 把两者结合后误差最低，而且 bias 几乎为 0。

### 11. Validation

讲审稿人视角：

> 我们做了 bootstrap CI、paired improvement、random-pressure control 和 global prior 对照。这里也主动承认 global prior 很强，所以 cohort-specific LLM ranking 是 selective refinement，而不是全部提升来源。

### 12. Selector Branch

这页是为了融合 0611 的前半部分：

> selector branch 的价值在冷启动和数据稀疏，而不是替代主结果。它解决的是效率和缺标签问题；主 NHTS temporal-shift 结果仍然看 LLM-corrected XGBoost。

不要说“同学做的方案”，说：

> selector branch。

### 13. What the LLM Adds

这页回答“为什么不用纯 LLM”：

> LLM-only 方向是对的，但 numerical calibration 不够。XGBoost 有 household grounding，但不能适应目标年新情境。LLM-guided temporal adaptation 的贡献就是把二者分工。

### 14. Behavior-System Outputs

简短讲：

> 除了家庭总出行次数，我们还做了 mode composition、purpose composition 和 mode-specific trip volume。这让结果可以用于规划解释，而不是只报一个 MAE。

### 15. Multi-Objective

讲规划意义：

> 如果目标是校准，选择 gated LLM；如果目标是低成本，global prior 也可以作为 operating point。这里体现的是多目标选择，而不是单一 leaderboard。

### 16. Limitations

主动降风险：

> 我们不声称识别 COVID 的 causal effect，也不声称 LLM 直接优于所有模型。我们的 claim 是 label-free temporal adaptation。

### 17. Takeaways

最后三句话：

1. 目标年时序迁移会让历史出行模型系统性偏差。
2. LLM 最适合作为情境先验和模拟选择器，不是直接数值预测器。
3. 在 NHTS 2022 stress test 中，LLM-corrected XGBoost 同时改善 wMAE、bias 和多目标规划指标。

### 18. Backup Map

不主动讲。Q&A 时用：

- 问 selector 细节：讲原 0611 第 9-14 页。
- 问 cold-start accuracy：讲原 0611 第 15-18 页。
- 问稳健性：讲误差、异质性、random pressure。
- 问外部验证：讲 ACS / PSRC / BTS。

## 最容易被问的问题

**Q: 你们到底是在做 selector 还是 corrected XGBoost？**

A: 是同一 temporal-adaptation framework 的两个分支。selector 面向 cold-start 和 sparse-data；corrected XGBoost 面向有历史 NHTS 标签但目标年发生情境变化的主实验。

**Q: 为什么不直接用 LLM 预测？**

A: 纯 LLM 有目标年情境方向，但缺少 NHTS household-level 数值校准。结果上 LLM-only wMAE 是 `2.718`，hybrid gated 是 `2.502`，而且 bias 更接近 0。

**Q: 为什么不只用 global prior？**

A: global prior 是强低成本 baseline，说明目标年整体 correction 很重要。我们的 gated method 在 calibration-first 和 balanced profile 下更好，cohort-specific LLM prior 是 selective refinement。

**Q: 这个是不是只解决疫情？**

A: 不是。2022 是 target-year temporal-shift stress test。框架本身是针对目标年情境变化：只要有新的外部机制，就可以把它转成 context prior 或 selector signal。
