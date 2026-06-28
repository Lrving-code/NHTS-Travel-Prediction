# 0611 融合版 PPT 审计与重构建议

审计对象：`refs/0611.pptx`

## 总体判断

这版 PPT 的主要问题不是“内容少”，而是“内容被两套逻辑顺序拼接”：

- 第 3-18 页主要讲 cold-start / LLM 知识蒸馏 / 模拟预测选择器。
- 第 19-30 页主要讲 NHTS 2022 temporal shift / LLM-corrected XGBoost。
- 第 31 页才试图把两条路线合起来，但放得太晚，导致听众前 20 分钟都不知道主方法到底是什么。

如果直接用于 10 分钟汇报，老师最可能追问：

1. 你们到底是在做出行方式分类，还是家庭出行次数预测？
2. 你们的核心方法是 LLM rule selector，还是 LLM 修正 XGBoost？
3. 为什么前半部分用 accuracy，后半部分用 weighted MAE/bias？
4. 为什么前半部分说“纯 LLM 和混合模型准确率相近”，后半部分又说 hybrid gated 才是主方法？
5. 2022 是一个实验场景，还是方法只能解决疫情？

## 推荐总题目

中文：

> 面向时序迁移的家庭出行预测：模拟选择器与大模型修正框架

英文：

> LLM-Guided Temporal Adaptation for Household Mobility Prediction

这个题目把 2022 疫情恢复期降为 target-year temporal shift case，而不是把项目限定为疫情研究。它也能同时容纳两条路线：

- 模拟预测选择器：适合 cold-start、数据稀疏、字段缺失场景。
- LLM-corrected XGBoost：适合已有历史调查数据、但目标年出现情境变化的场景。

## 核心主线

建议把主线改成一句话：

> 当目标年出行行为发生时序迁移时，单纯历史拟合缺少新情境机制，纯 LLM 又缺少数值校准。我们构建一个 LLM-guided temporal adaptation framework：在 cold-start 场景下用 LLM 蒸馏规则做预测选择器，在 NHTS 2022 主场景下用 LLM 情境先验修正历史 XGBoost，从而实现无目标年标签的家庭出行行为预测。

更短的口头版本：

> 不是让 LLM 直接预测出行，而是让 LLM 帮模型适应目标年的新情境。

## 当前页级问题

| 页码 | 当前内容 | 问题 | 建议 |
|---:|---|---|---|
| 1 | `XXXX` | 标题页未完成 | 换成新题目，副标题写 selector + LLM-corrected XGBoost |
| 2 | 目录 | 每章解释文字过长，占用注意力 | 改成 5 个章节：Motivation / Problem / Method / Results / Takeaways |
| 3-5 | 泛泛讲传统模型与 LLM | 太通用，和 NHTS/时序迁移联系弱 | 压缩成 1 页背景：为什么目标年迁移难 |
| 6 | 两类场景 | 这是关键桥梁，但信息排版混乱 | 提前放到第 3 页，作为全篇方法谱系 |
| 7 | 核心任务：出行方式预测 | 和后文主结果 trip count 冲突 | 改成 household behavior system：trips / mode / purpose |
| 8 | 评估方案 | 只讲准确率/效率/鲁棒性，未对应后文指标 | 改成两套指标：classification accuracy 和 weighted MAE/bias |
| 9-14 | 规则蒸馏与回退机制 | 内容可用，但太细 | 主讲压缩为 1 页，细节放 backup |
| 15-18 | cold-start 结果 | 可用，但会抢主线 | 主讲只保留 1 页方法谱系对比，其余放 backup |
| 19 | 多输出任务 | 好页，但应提前 | 放到 Problem/Data 后，明确不是只预测 CNTTDHH |
| 20-22 | 技术路线、LLM 先验、对比体系 | 核心页 | 保留，但术语从“疫情”改为“目标年情境变化” |
| 23-24 | 主结果和 CI | 核心页 | 保留，作为最重要结果 |
| 25 | 多目标选择 | 有价值 | 保留，但讲 30 秒即可 |
| 26-30 | 误差、异质性、稳健性、LLM 角色 | 有重复 | 压成 2 页：robustness + why hybrid |
| 31 | 总结谱系 | 应提前出现 | 移到 Method 前，作为全篇总框架 |

## 建议 10 分钟主讲结构

### 1. Title

题目：面向时序迁移的家庭出行预测：模拟选择器与大模型修正框架  
一句话：目标不是让 LLM 替代模型，而是让模型适应目标年情境变化。

### 2. Motivation: 目标年迁移为什么难

讲清楚两个失败：

- 传统历史模型：数值校准强，但不理解目标年新机制。
- 纯 LLM：有情境先验，但缺少 NHTS 数值校准。

### 3. Problem: household behavior system

不要只说“出行方式预测”。统一成：

- trip generation：家庭出行次数
- mode composition：方式占比
- purpose composition：目的占比
- mode-specific trips：各方式出行量

### 4. Method Spectrum: 从选择器到修正器

把第 31 页提前，改成方法演进链：

```text
data-only fitting
-> pure LLM
-> LLM rule / prediction selector
-> LLM rule + small historical calibration
-> historical model + LLM target-year correction
```

这一页必须讲清楚：selector 和 LLM-corrected XGBoost 不是互相替代，而是两个数据条件下的模块。

### 5. Unified Framework

建议画成两条分支：

```text
Input household profile
        |
        +-- Cold-start branch: LLM rule -> prediction selector -> fallback
        |
        +-- Historical-data branch: XGBoost baseline -> LLM context prior -> corrected prediction
```

### 6. Data and No-Label Setup

强调：

- 训练只用历史 NHTS 标签。
- 目标年出行标签只用于最终评估。
- LLM 输入不含 `CNTTDHH`、`WTHHFIN`、`HOUSEID`。

### 7. Baselines

保留第 22 页，但修正“2022 标签”列名为“目标年标签”。把 `●/×` 用作表格符号是可以的。

### 8. Main Result

保留第 23 页。

讲法：

- ordinary XGBoost wMAE 4.338，bias +3.605
- pure LLM pressure wMAE 2.718，bias -0.373
- zero-shot rule tree wMAE 2.602，bias -0.407
- hybrid gated wMAE 2.502，bias -0.023

结论：LLM 有方向感，历史模型有数值锚点，二者分工后最好。

### 9. Statistical and Robustness Checks

合并第 24、28 页：

- bootstrap CI 支持 improvement
- random-pressure control 支持 LLM pressure 不是随机排序
- global prior 很强，cohort prior 是 selective refinement

### 10. Behavior Outputs and Planning Use

简述 mode/purpose/mode-specific trips，不要展开太多。

### 11. Efficiency

只保留一句：

- cohort-level batch prompts 将 household-level LLM 调用压缩到 89 个 batch prompts。
- 主推理阶段使用 deterministic adapter，不逐户调用 LLM。

不要在主结果里使用“75%/25% 调用比例”，除非该比例已经用相同评估协议复现实验。

### 12. Conclusion

三句话：

1. 目标年时序迁移会让历史模型系统性偏差。
2. LLM 作为情境先验/选择器，而不是直接数值预测器。
3. hybrid correction 在 NHTS 2022 上同时改善 MAE、bias 和多目标规划指标。

## 建议主讲/备份分配

主讲保留约 16-18 页：

- 1 标题
- 2 背景
- 3 问题定义与输出
- 4 方法谱系
- 5 统一框架
- 6 数据与 no-label setup
- 7 LLM context prior
- 8 baseline 对比体系
- 9 主结果
- 10 statistical validation
- 11 robustness
- 12 LLM 角色：selector vs corrector
- 13 mode/purpose 扩展
- 14 efficiency
- 15 limitations
- 16 conclusion

Backup：

- 原第 9-14 页规则蒸馏细节
- 原第 15-18 页 cold-start accuracy 表
- 原第 26-30 页详细误差/异质性图
- 外部验证、PSRC、ACS、BTS

## 必须修改的术语

| 当前说法 | 建议改法 |
|---|---|
| 疫情影响异质性 | 目标年情境异质性 |
| 疫情事件先验 | 目标年情境先验 / event-context prior |
| 后疫情预测 | target-year temporal adaptation |
| 同学方案 | 模拟预测选择器路线 / LLM-rule selector branch |
| 我们方案 | LLM-corrected XGBoost / historical-data correction branch |
| 出行方式预测是核心任务 | household travel behavior prediction 是核心任务 |

## 最重要的改稿原则

不要按“谁做了哪部分”组织，而要按“同一个研究问题需要哪些模块”组织。

更好的叙事顺序是：

```text
问题：目标年时序迁移
困难：历史拟合和纯 LLM 各有缺陷
框架：LLM-guided temporal adaptation
模块：selector branch + correction branch
主实验：NHTS 2022 household behavior prediction
结论：LLM 最适合作为情境先验和选择器，而不是直接替代数值预测模型
```
