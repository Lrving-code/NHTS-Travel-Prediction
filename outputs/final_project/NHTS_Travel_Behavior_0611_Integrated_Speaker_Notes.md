# 0611 Integrated Speaker Notes

## 10 分钟主讲路径

1. 第 1-5 页：说明问题不是普通出行预测，而是目标年时序迁移；用 2025-2026 文献调研引出 LLM event-generalization。
2. 第 6-8 页：把两方工作统一成两条分支：selector branch 处理冷启动/字段缺失，correction branch 处理有历史数据但目标年情境变化。
3. 第 9-12 页：讲主方法。历史 XGBoost 学 routine mobility，LLM 生成 trip suppression / remote work / transit avoidance 等结构化 priors，再做 no-label correction。
4. 第 13-19 页：讲证据链。主结果、bootstrap CI、误差-偏差权衡、多目标选择、随机置换、异质性和分组稳健性。
5. 第 20-25 页：解释 LLM 的真实角色、mode/purpose 扩展、batch prompting/部署逻辑和总结。

## 关键口径

- 不要说 LLM 直接预测家庭出行次数；要说 LLM 生成 event prior / selector signal。
- 不要说两个方案拼接；要说它们是同一个 temporal-adaptation framework 的两个 operating regimes。
- 不要把总标题写成“知识蒸馏”或“模型蒸馏”；本项目更准确的说法是规则提取、事件先验生成和历史模型校准。
- 不要隐瞒 global prior 很强；应说主贡献是 event-level correction，cohort ranking 是 selective refinement。
- local/open-source LLM replication 目前是 protocol + environment audit，当前环境被 CPU-only PyTorch 阻塞，不能作为已完成结果。