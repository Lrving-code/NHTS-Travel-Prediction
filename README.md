# NHTS Travel Prediction

## 研究目标
利用历史NHTS数据（2001/2009/2017）训练模型，预测2022年家庭出行行为，比较不同迁移策略的效果。

## 方法对比
1. **Baseline**: XGBoost 2017训练 → 2022直接预测（无调整）
2. **策略A**: 样本重加权（domain adaptation）
3. **策略B**: LLM辅助残差修正
4. **策略C**: 多年份训练 + 时间特征（2001→2009→2017 → 预测2022）
5. **Upper bound**: XGBoost 2022训练 → 2022测试

## 数据来源
[NHTS](https://nhts.ornl.gov/) — National Household Travel Survey

| 年份 | 数据表 | 格式 |
|------|--------|------|
| 2001 | HOUSEHOLD, PERSON, VEHICLE, TRIP | CSV/SAS |
| 2009 | HOUSEHOLD, PERSON, VEHICLE, TRIP | CSV/SAS |
| 2017 | HOUSEHOLD, PERSON, VEHICLE, TRIP | CSV/SAS/SPSS |
| 2022 | HOUSEHOLD, PERSON, VEHICLE, TRIP | CSV/SAS/SPSS |

## 项目结构
```
data/
  raw/           — 各年份原始NHTS数据
    nhts_2001/
    nhts_2009/
    nhts_2017/
    nhts_2022/
  interim/       — 清洗对齐后的中间数据
  processed/     — 最终建模用数据集
src/             — Python脚本
notebooks/       — 探索性分析
outputs/         — 模型结果与评估
```

## 实验计划
- [ ] Phase 1: 数据下载与清洗，跨年份变量对齐
- [ ] Phase 2: Baseline实验（2017→2022直接迁移）
- [ ] Phase 3: 迁移策略A/B/C实现
- [ ] Phase 4: 评估对比与可视化
