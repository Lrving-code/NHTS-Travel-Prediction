"""Build a refined 10-minute deck from the 0611 merged presentation story."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pptx import Presentation

import build_template_presentation as base


LOGGER = logging.getLogger(__name__)
OUTPUT_PATH = base.FINAL_DIR / "NHTS_Temporal_Adaptation_0611_Refined_10min.pptx"
TOTAL_SLIDES = 18


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_presentation() -> Presentation:
    prs = Presentation(base.TEMPLATE_PATH) if base.TEMPLATE_PATH.exists() else Presentation()
    if base.TEMPLATE_PATH.exists():
        base.clear_template_slides(prs)
    prs.slide_width = base.Inches(13.333)
    prs.slide_height = base.Inches(7.5)
    base.TOTAL_SLIDES = TOTAL_SLIDES
    return prs


def metric(df: pd.DataFrame, method: str, column: str) -> float:
    return float(df.loc[df["method"] == method, column].iloc[0])


def add_title(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide, base.COLORS["navy"])
    base.add_logo(slide, logo, 11.82, 0.34, 0.92, True)
    base.text_box(slide, "面向时序迁移的家庭出行预测\n模拟选择器与大模型修正框架", 0.68, 0.78, 9.9, 0.95, 24, base.COLORS["white"], True)
    base.text_box(slide, "LLM-Guided Temporal Adaptation for Household Mobility Prediction", 0.72, 1.86, 10.2, 0.32, 11.5, base.RGBColor(203, 213, 225), True)
    base.text_box(slide, "prediction selector + LLM-corrected XGBoost；目标年标签只用于最终评估", 0.72, 2.23, 9.7, 0.28, 10, base.RGBColor(203, 213, 225))
    stages = [
        ("目标年迁移", "target-year shift"),
        ("模拟选择器", "cold-start branch"),
        ("LLM 修正模型", "correction branch"),
        ("行为系统预测", "trips / modes / purposes"),
    ]
    for idx, (title, note) in enumerate(stages):
        base.node(slide, 0.8 + idx * 3.02, 3.35, 2.25, 0.7, title, note, base.COLORS["teal"], base.RGBColor(30, 64, 91))
        if idx < len(stages) - 1:
            base.arrow(slide, 3.1 + idx * 3.02, 3.57, 0.48, 0.18, base.RGBColor(148, 163, 184))
    base.metric_card(slide, 0.82, 5.05, 2.55, 0.9, "Trip-count MAE", f"-{values['mae_reduction']:.1%}", "vs ordinary XGBoost", base.COLORS["blue"], base.COLORS["white"])
    base.metric_card(slide, 3.68, 5.05, 2.55, 0.9, "Weighted bias", f"{values['gated_bias']:+.3f}", "nearly unbiased", base.COLORS["green"], base.COLORS["white"])
    base.metric_card(slide, 6.54, 5.05, 2.55, 0.9, "Selector role", "fallback", "cold-start / sparse data", base.COLORS["orange"], base.COLORS["white"])
    base.metric_card(slide, 9.4, 5.05, 2.55, 0.9, "Behavior scope", "3+ outputs", "trips / modes / purposes", base.COLORS["purple"], base.COLORS["white"])
    base.text_box(slide, "Mobility Inequality and Sustainable Development", 0.72, 6.83, 4.9, 0.2, 8, base.RGBColor(203, 213, 225), True)
    base.text_box(slide, f"01 / {TOTAL_SLIDES}", 11.72, 6.83, 0.85, 0.2, 8, base.RGBColor(203, 213, 225), True, base.PP_ALIGN.RIGHT)


def add_agenda(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "00", "汇报路径 / Talk Roadmap", "按一个问题组织，而不是按两套方案拼接", 2, logo)
    items = [
        ("01", "Why", "目标年时序迁移为什么会让历史模型失效"),
        ("02", "What", "家庭出行行为系统：出行多少、怎么出行、为什么出行"),
        ("03", "How", "selector branch + LLM-corrected XGBoost"),
        ("04", "Evidence", "主结果、统计验证、稳健性与多目标选择"),
        ("05", "Takeaway", "LLM 是情境先验和选择器，不是直接替代数值模型"),
    ]
    for idx, (num, title, note) in enumerate(items):
        y = 1.35 + idx * 0.78
        base.metric_card(slide, 0.82, y, 1.25, 0.52, num, title, "", base.COLORS["teal"])
        base.text_box(slide, note, 2.3, y + 0.08, 8.9, 0.26, 12, base.COLORS["ink"], idx in {0, 2})
    base.text_box(slide, "讲法：同一个 temporal-adaptation 问题，需要两个模块；不是两份作业顺序拼接。", 1.0, 6.25, 10.9, 0.35, 12, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def add_motivation(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "01", "研究动机 / Motivation", "目标年情境变化会破坏历史出行规律", 3, logo)
    base.story_box(slide, "传统历史模型", "Learns household routine mobility but misses target-year mechanisms.", 0.78, 1.35, 3.6, 1.25, base.COLORS["blue"])
    base.story_box(slide, "纯 LLM 预测", "Knows context direction but lacks NHTS numerical calibration.", 4.82, 1.35, 3.6, 1.25, base.COLORS["purple"])
    base.story_box(slide, "融合思路", "Use LLM to guide adaptation: select when data are sparse; correct when history exists.", 8.86, 1.35, 3.6, 1.25, base.COLORS["teal"])
    base.metric_card(slide, 1.05, 3.25, 2.55, 0.95, "Ordinary XGBoost", f"{values['xgb_mae']:.3f}", f"wBias {values['xgb_bias']:+.3f}", base.COLORS["blue"])
    base.metric_card(slide, 3.95, 3.25, 2.55, 0.95, "LLM-only pressure", f"{values['llm_only_mae']:.3f}", f"wBias {values['llm_only_bias']:+.3f}", base.COLORS["purple"])
    base.metric_card(slide, 6.85, 3.25, 2.55, 0.95, "Hybrid gated", f"{values['gated_mae']:.3f}", f"wBias {values['gated_bias']:+.3f}", base.COLORS["green"])
    base.text_box(slide, "核心矛盾：历史模型有数值锚点但缺少新情境；LLM 有情境常识但校准弱。", 1.0, 5.0, 10.9, 0.38, 14, base.COLORS["ink"], True, base.PP_ALIGN.CENTER)


def add_problem_outputs(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "02", "问题定义 / Problem Definition", "核心不是单一 mode classification，而是 household behavior system", 4, logo)
    headers = ["Output", "Construction", "Planning meaning", "Main metric"]
    rows = [
        ["Trip generation", "CNTTDHH", "How much travel demand", "weighted MAE / bias"],
        ["Mode composition", "TRPTRANS -> share vector", "How demand is distributed", "TV / share MAE"],
        ["Purpose composition", "TRIPPURP -> purpose vector", "Why households travel", "TV / purpose MAE"],
        ["Mode-specific trips", "predicted trips × mode share", "Trips by mode", "derived MAE"],
    ]
    base.small_table(slide, headers, rows, 0.72, 1.35, [2.2, 3.0, 3.3, 2.3], row_h=0.45, font_size=8.5)
    base.text_box(slide, "主结果集中在 trip generation；mode/purpose 是行为系统扩展，不要把汇报讲成只做出行方式分类。", 1.0, 5.0, 11.0, 0.4, 12, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def add_method_spectrum(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "03", "方法谱系 / Method Spectrum", "把 0611 的两条路线提前合成一条演化链", 5, logo)
    cards = [
        ("Data-only", "历史拟合", values["xgb_mae"], base.COLORS["blue"]),
        ("Pure LLM", "情境方向", values["llm_only_mae"], base.COLORS["purple"]),
        ("LLM selector", "冷启动选择器", 2.602, base.COLORS["orange"]),
        ("Rule + history", "少量历史校准", values["rule_500_mae"], base.COLORS["teal"]),
        ("LLM-corrected XGB", "主方法", values["gated_mae"], base.COLORS["green"]),
    ]
    for idx, (title, note, mae, color) in enumerate(cards):
        x = 0.72 + idx * 2.45
        base.node(slide, x, 1.42, 2.02, 0.72, title, f"{note}\nwMAE {mae:.3f}", color)
        if idx < len(cards) - 1:
            base.arrow(slide, x + 2.08, 1.68, 0.34, 0.16, base.COLORS["gray"])
    base.small_table(
        slide,
        ["Branch", "When it matters", "Role in final story"],
        [
            ["Prediction selector", "冷启动、数据稀疏、字段缺失", "从 LLM 蒸馏可执行规则，低置信样本回退"],
            ["LLM-corrected XGBoost", "有历史 NHTS，但目标年出现情境变化", "历史模型做数值锚点，LLM 做目标年修正"],
        ],
        1.15,
        3.25,
        [2.7, 4.2, 4.2],
        row_h=0.5,
        font_size=8.5,
    )
    base.text_box(slide, "这一页应替代 0611 原第 31 页的位置：先给全局框架，再讲细节。", 1.0, 5.9, 11.0, 0.35, 12, base.COLORS["red"], True, base.PP_ALIGN.CENTER)


def add_unified_framework(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "03", "统一框架 / Unified Framework", "同一输入进入两个互补分支", 6, logo)
    base.node(slide, 0.82, 2.55, 2.25, 0.75, "Household profile", "income, workers, vehicles,\nurban context", base.COLORS["navy"])
    base.arrow(slide, 3.15, 2.75, 0.55, 0.18, base.COLORS["gray"])
    base.node(slide, 4.0, 1.35, 2.8, 0.9, "Selector branch", "LLM rule distillation\nconfidence gate / fallback", base.COLORS["orange"])
    base.node(slide, 4.0, 3.75, 2.8, 0.9, "Correction branch", "XGBoost baseline\nLLM context prior adapter", base.COLORS["green"])
    base.arrow(slide, 6.95, 1.65, 0.62, 0.18, base.COLORS["gray"])
    base.arrow(slide, 6.95, 4.05, 0.62, 0.18, base.COLORS["gray"])
    base.node(slide, 7.85, 1.35, 3.6, 0.9, "Cold-start output", "fast simulated prediction\nwhen labels are sparse", base.COLORS["orange"])
    base.node(slide, 7.85, 3.75, 3.6, 0.9, "Target-year output", "label-free corrected\nhousehold behavior prediction", base.COLORS["green"])
    base.text_box(slide, "讲法：selector 解决“没有足够历史标签怎么办”；corrector 解决“有历史标签但目标年变了怎么办”。", 1.0, 5.75, 11.1, 0.38, 12, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def add_no_label_setup(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "04", "数据与无标签设定 / Data and No-Label Setup", "目标年标签只用于最终评估", 7, logo)
    years = [("2001", "history"), ("2009", "history"), ("2017", "history + mode"), ("2022", "target-year eval")]
    for idx, (year, note) in enumerate(years):
        x = 0.95 + idx * 2.7
        color = base.COLORS["green"] if year != "2022" else base.COLORS["orange"]
        base.metric_card(slide, x, 1.45, 2.05, 0.8, year, note, "", color)
        if idx < len(years) - 1:
            base.arrow(slide, x + 2.12, 1.78, 0.36, 0.15, base.COLORS["gray"])
    base.small_table(
        slide,
        ["LLM branch allowed", "LLM branch excluded", "Reason"],
        [
            ["cohort profile", "CNTTDHH", "target label"],
            ["urban context / workers / vehicles", "WTHHFIN", "survey weight"],
            ["generic target-year context", "HOUSEID", "identifier"],
        ],
        1.05,
        3.05,
        [3.55, 3.2, 3.2],
        row_h=0.48,
        font_size=8.5,
    )
    base.text_box(slide, "No-label claim: 2022 outcomes do not train, calibrate, or prompt the LLM branch.", 1.1, 5.55, 10.8, 0.32, 12, base.COLORS["ink"], True, base.PP_ALIGN.CENTER)


def add_context_prior(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "04", "LLM 情境先验 / LLM Context Prior", "LLM 输出机制变量，不直接输出出行次数", 8, logo)
    rows = [
        ["trip_suppression", "总体出行折减压力", "correct total trips"],
        ["remote_work", "通勤被远程办公替代", "explain commute reduction"],
        ["transit_avoidance", "公共交通规避倾向", "correct transit share"],
        ["delivery_substitution", "购物/服务线上替代", "explain non-work decline"],
        ["recovery_sensitivity", "恢复到常态出行能力", "capture rebound heterogeneity"],
    ]
    base.small_table(slide, ["LLM output", "变量含义", "模型位置"], rows, 0.85, 1.35, [2.9, 4.2, 4.3], row_h=0.47, font_size=8.5)
    base.text_box(slide, "LLM role = context prior generator / selector signal, not standalone numerical predictor.", 1.05, 5.6, 11.0, 0.35, 12, base.COLORS["purple"], True, base.PP_ALIGN.CENTER)


def add_baselines(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "05", "对比体系 / Baselines", "公平比较：传统、纯 LLM、selector、global prior、hybrid correction", 9, logo)
    rows = [
        ["Ordinary XGBoost", "Yes", "No", "No", "historical routine only", f"{values['xgb_mae']:.2f}"],
        ["LLM-only pressure", "No", "Yes", "No", "context without calibration", f"{values['llm_only_mae']:.2f}"],
        ["Zero-shot rule tree", "No", "Yes", "No", "selector-style rule baseline", f"{values['zero_rule_mae']:.2f}"],
        ["Rule + 500 history", "Yes", "Yes", "No", "small calibration bridge", f"{values['rule_500_mae']:.2f}"],
        ["Global event rule", "Yes", "Global", "No", "low-cost target-year correction", f"{values['global_mae']:.2f}"],
        ["Hybrid gated", "Yes", "Yes", "No", "LLM-corrected XGBoost", f"{values['gated_mae']:.2f}"],
    ]
    base.small_table(slide, ["Method", "History", "LLM", "Target labels", "Role", "wMAE"], rows, 0.55, 1.18, [2.35, 1.25, 1.25, 1.55, 3.45, 1.0], row_h=0.43, font_size=8.1)
    base.text_box(slide, "把 selector branch 作为方法谱系中的 cold-start baseline；主 NHTS temporal-shift 结果看 hybrid correction。", 1.05, 5.85, 11.0, 0.35, 11.5, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def add_main_results(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "05", "主结果 / Main Results", "LLM 修正 XGBoost 同时降低误差和系统性偏差", 10, logo)
    rows = [
        ["Ordinary XGBoost", f"{values['xgb_mae']:.3f}", f"{values['xgb_bias']:+.3f}", f"{values['xgb_r2']:.3f}"],
        ["CatBoost GPU", "4.220", "+3.487", "-0.571"],
        ["LLM-only pressure", f"{values['llm_only_mae']:.3f}", f"{values['llm_only_bias']:+.3f}", f"{values['llm_only_r2']:.3f}"],
        ["Zero-shot rule tree", f"{values['zero_rule_mae']:.3f}", f"{values['zero_rule_bias']:+.3f}", "0.151"],
        ["Rule + 500 history", f"{values['rule_500_mae']:.3f}", "+0.463", "0.154"],
        ["Hybrid gated", f"{values['gated_mae']:.3f}", f"{values['gated_bias']:+.3f}", f"{values['gated_r2']:.3f}"],
    ]
    base.small_table(slide, ["Method", "wMAE", "wBias", "wR2"], rows, 0.78, 1.22, [3.8, 1.6, 1.6, 1.6], row_h=0.45, font_size=8.8)
    base.metric_card(slide, 9.1, 1.35, 2.65, 0.9, "MAE reduction", f"-{values['mae_reduction']:.1%}", "vs XGBoost", base.COLORS["green"])
    base.metric_card(slide, 9.1, 2.55, 2.65, 0.9, "Bias reduction", "-99.4%", "absolute wBias", base.COLORS["teal"])
    base.metric_card(slide, 9.1, 3.75, 2.65, 0.9, "Key claim", "hybrid", "direction + calibration", base.COLORS["purple"])
    base.text_box(slide, "结论：LLM 有情境方向，历史模型有数值锚点；二者分工后最好。", 1.0, 5.85, 11.0, 0.35, 12, base.COLORS["ink"], True, base.PP_ALIGN.CENTER)


def add_validation(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "06", "统计验证与稳健性 / Validation", "不是单个点估计，也不是随机 pressure 的偶然收益", 11, logo)
    base.small_table(
        slide,
        ["Check", "Evidence", "Interpretation"],
        [
            ["Bootstrap CI", "Hybrid wMAE 2.502 [2.429, 2.581]", "main gain is statistically stable"],
            ["Paired gain", "MAE gain 1.84, CI [1.74, 1.93]", "paired improvement remains positive"],
            ["Random pressure", "real LLM pressure beats 500 shuffles", "context prior is not arbitrary ranking"],
            ["Global prior", "global rule wMAE about 2.52", "event-level correction is strong baseline"],
        ],
        0.72,
        1.35,
        [2.35, 4.4, 4.2],
        row_h=0.55,
        font_size=8.5,
    )
    base.text_box(slide, "谨慎表述：global correction 是主体，cohort-specific LLM ranking 是 selective refinement。", 1.0, 5.8, 11.0, 0.35, 12, base.COLORS["red"], True, base.PP_ALIGN.CENTER)


def add_selector_branch(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "06", "Selector Branch 怎么放 / How to Position the Selector", "保留 0611 的贡献，但不要让它抢主线", 12, logo)
    base.story_box(slide, "适用场景", "Cold-start, sparse labels, missing fields, and fast simulation.", 0.78, 1.25, 3.55, 1.25, base.COLORS["orange"])
    base.story_box(slide, "方法机制", "LLM distills rules; confidence gate selects rule output or LLM fallback.", 4.88, 1.25, 3.55, 1.25, base.COLORS["purple"])
    base.story_box(slide, "汇报角色", "A bridge baseline and deployment module, not the final NHTS count winner.", 8.98, 1.25, 3.55, 1.25, base.COLORS["teal"])
    base.small_table(
        slide,
        ["0611 branch result", "Use in talk", "Caveat"],
        [
            ["25% LLM fallback", "efficiency motivation", "do not mix into main 2022 count metric"],
            ["accuracy close to pure LLM", "cold-start evidence", "different metric from wMAE"],
            ["rule interpretability", "explainability module", "needs same-protocol validation later"],
        ],
        1.05,
        3.35,
        [2.9, 3.6, 4.2],
        row_h=0.5,
        font_size=8.5,
    )


def add_llm_role(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "06", "LLM 的角色 / What the LLM Adds", "不是直接替代模型，而是提供时序适应信号", 13, logo)
    base.node(slide, 0.85, 1.35, 2.75, 0.9, "Historical model", f"household grounding\nwMAE {values['xgb_mae']:.2f}", base.COLORS["blue"])
    base.node(slide, 4.08, 1.35, 2.75, 0.9, "Pure LLM", f"context direction\nwMAE {values['llm_only_mae']:.2f}", base.COLORS["purple"])
    base.node(slide, 7.31, 1.35, 2.75, 0.9, "Selector", "rule / fallback\ncold-start module", base.COLORS["orange"])
    base.node(slide, 10.1, 1.35, 2.3, 0.9, "Hybrid", f"direction + calibration\nwMAE {values['gated_mae']:.2f}", base.COLORS["green"])
    base.small_table(
        slide,
        ["Failure mode", "Why hybrid fixes it"],
        [
            ["XGBoost overpredicts target-year trips", "LLM prior adds target-year correction"],
            ["LLM-only has weak numerical calibration", "historical baseline anchors household scale"],
            ["Selector branch is efficient but cold-start oriented", "main task uses full historical grounding"],
        ],
        1.05,
        3.35,
        [4.7, 6.0],
        row_h=0.55,
        font_size=8.5,
    )


def add_behavior_extensions(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "07", "行为系统扩展 / Behavior-System Outputs", "不仅是 trip count，也能服务规划解释", 14, logo)
    base.metric_card(slide, 0.92, 1.35, 2.55, 0.9, "Mode TV", f"{values['mode_tv_base']:.3f} -> {values['mode_tv_llm']:.3f}", "XGB + transit prior", base.COLORS["green"])
    base.metric_card(slide, 3.82, 1.35, 2.55, 0.9, "Transit MAE", f"{values['transit_base']:.3f} -> {values['transit_llm']:.3f}", "mode correction", base.COLORS["teal"])
    base.metric_card(slide, 6.72, 1.35, 2.55, 0.9, "Mode-trip MAE", f"{values['mode_trip_hybrid']:.2f}", "count × mode", base.COLORS["purple"])
    base.metric_card(slide, 9.62, 1.35, 2.55, 0.9, "Purpose", "exploratory", "future adapter", base.COLORS["orange"])
    base.text_box(slide, "讲法：方式和目的模块说明框架可以扩展到规划相关输出；最强证据仍是 trip-generation temporal adaptation。", 1.05, 3.25, 11.0, 0.45, 12, base.COLORS["ink"], True, base.PP_ALIGN.CENTER)
    base.small_table(
        slide,
        ["Planning question", "Output used"],
        [
            ["总体交通需求是否下降？", "trip generation"],
            ["公共交通恢复是否不足？", "mode share / transit share"],
            ["哪些出行目的发生变化？", "purpose composition"],
        ],
        2.0,
        4.25,
        [4.6, 4.6],
        row_h=0.5,
        font_size=8.8,
    )


def add_multi_objective(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "07", "多目标选择 / Multi-Objective Evaluation", "把模型结果转成规划决策", 15, logo)
    rows = [
        ["Calibration-first", "Gated LLM", "2.502", "0.023", "89"],
        ["Balanced / main", "Gated LLM", "2.502", "0.023", "89"],
        ["Low-cost", "Global prior", "2.553", "0.123", "1"],
    ]
    base.small_table(slide, ["Preference", "Selected method", "wMAE", "|bias|", "LLM req."], rows, 1.35, 1.6, [2.7, 2.8, 1.5, 1.5, 1.5], row_h=0.55, font_size=8.8)
    base.text_box(slide, "不是只有一个最优模型：如果预算很低，global prior 是可解释低成本点；如果追求校准，gated hybrid 是主点。", 1.1, 4.25, 10.8, 0.55, 13, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def add_limitations(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "08", "边界与改进 / Limitations", "主动讲清楚，避免被追问时被动", 16, logo)
    rows = [
        ["Not causal effect", "causal guardrails support mechanism plausibility, not COVID effect identification"],
        ["Not pure LLM superiority", "LLM-only remains less calibrated than hybrid"],
        ["Selector branch protocol", "needs same-metric validation before becoming main result"],
        ["Event context risk", "future work: frozen RAG corpus / prospective event feeds"],
    ]
    base.small_table(slide, ["Boundary", "Paper-safe wording"], rows, 0.85, 1.35, [3.0, 7.8], row_h=0.58, font_size=8.5)
    base.text_box(slide, "这页可以只讲 30 秒，但必须保留：它让项目显得更像研究，而不是宣传。", 1.05, 5.75, 11.0, 0.35, 12, base.COLORS["red"], True, base.PP_ALIGN.CENTER)


def add_final(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide, base.COLORS["navy"])
    base.add_frame(slide, "END", "总结 / Takeaways", "LLM-guided temporal adaptation, not direct LLM prediction", 17, logo, True)
    base.text_box(
        slide,
        "当目标年情境变化打破历史连续性时，LLM 最适合作为情境先验和模拟选择器，帮助传统 household model 做时序适应。",
        1.15,
        1.55,
        10.8,
        0.72,
        17,
        base.COLORS["white"],
        True,
        base.PP_ALIGN.CENTER,
    )
    base.metric_card(slide, 1.25, 3.05, 2.75, 0.95, "Prediction", f"wMAE {values['gated_mae']:.3f}", "hybrid gated", base.COLORS["green"], base.COLORS["white"])
    base.metric_card(slide, 4.45, 3.05, 2.75, 0.95, "Calibration", f"bias {values['gated_bias']:+.3f}", "near zero", base.COLORS["teal"], base.COLORS["white"])
    base.metric_card(slide, 7.65, 3.05, 2.75, 0.95, "Framework", "2 branches", "selector + corrector", base.COLORS["orange"], base.COLORS["white"])
    base.text_box(slide, "一句话回答老师：我们不是用 LLM 直接造预测，而是用 LLM 帮模型选择和修正，使它适应目标年的新情境。", 1.25, 5.05, 10.5, 0.45, 12.5, base.RGBColor(203, 213, 225), True, base.PP_ALIGN.CENTER)


def add_backup_map(prs: Presentation, logo: bytes | None) -> None:
    slide = base.blank_slide(prs)
    base.set_background(slide)
    base.add_frame(slide, "B1", "Backup Map", "0611 原稿中应转入 Q&A 的内容", 18, logo)
    rows = [
        ["Original 9-14", "LLM rule extraction iterations", "问 selector 细节时讲"],
        ["Original 15-18", "cold-start accuracy tables", "问数据稀疏实验时讲"],
        ["Original 26-30", "error / heterogeneity details", "问稳健性和误差分布时讲"],
        ["External checks", "ACS / PSRC / BTS", "问外部验证时讲"],
    ]
    base.small_table(slide, ["Source", "Content", "When to use"], rows, 0.95, 1.45, [2.4, 4.5, 4.0], row_h=0.58, font_size=8.5)
    base.text_box(slide, "主讲只保留故事线，细节全部作为 backup，保证 10 分钟内讲清楚。", 1.05, 5.6, 11.0, 0.35, 12, base.COLORS["blue"], True, base.PP_ALIGN.CENTER)


def load_values() -> dict[str, float]:
    trip, _acc, mode, mode_trips, _purpose, _ci, _paired_ci = base.load_results()
    xgb_mae = metric(trip, "historical_xgboost", "weighted_mae")
    gated_mae = metric(trip, "gated_trip_suppression_a1_d0p15", "weighted_mae")
    mode_trip_rows = mode_trips[mode_trips["method"] == "gated_count_x_llm_mode"] if not mode_trips.empty else pd.DataFrame()
    return {
        "xgb_mae": xgb_mae,
        "xgb_bias": metric(trip, "historical_xgboost", "weighted_bias"),
        "xgb_r2": metric(trip, "historical_xgboost", "weighted_r2"),
        "llm_only_mae": metric(trip, "llm_only_trip_suppression_a1p25", "weighted_mae"),
        "llm_only_bias": metric(trip, "llm_only_trip_suppression_a1p25", "weighted_bias"),
        "llm_only_r2": metric(trip, "llm_only_trip_suppression_a1p25", "weighted_r2"),
        "zero_rule_mae": metric(trip, "zero_shot_llm_rule_tree", "weighted_mae"),
        "zero_rule_bias": metric(trip, "zero_shot_llm_rule_tree", "weighted_bias"),
        "rule_500_mae": metric(trip, "llm_rule_small_hist_calibrated_n500", "weighted_mae"),
        "global_mae": metric(trip, "global_trip_suppression_a1p25", "weighted_mae"),
        "gated_mae": gated_mae,
        "gated_bias": metric(trip, "gated_trip_suppression_a1_d0p15", "weighted_bias"),
        "gated_r2": metric(trip, "gated_trip_suppression_a1_d0p15", "weighted_r2"),
        "mae_reduction": (xgb_mae - gated_mae) / xgb_mae,
        "mode_tv_base": metric(mode, "historical_xgboost", "weighted_total_variation"),
        "mode_tv_llm": metric(mode, "llm_transit_avoidance_a1", "weighted_total_variation"),
        "transit_base": metric(mode, "historical_xgboost", "transit_share_weighted_mae"),
        "transit_llm": metric(mode, "llm_transit_avoidance_a1", "transit_share_weighted_mae"),
        "mode_trip_hybrid": float(mode_trip_rows["weighted_total_mode_trip_mae"].iloc[0]) if not mode_trip_rows.empty else 3.2802,
    }


def create_deck() -> Path:
    prs = create_presentation()
    logo = base.extract_template_logo()
    values = load_values()
    add_title(prs, logo, values)
    add_agenda(prs, logo)
    add_motivation(prs, logo, values)
    add_problem_outputs(prs, logo)
    add_method_spectrum(prs, logo, values)
    add_unified_framework(prs, logo)
    add_no_label_setup(prs, logo)
    add_context_prior(prs, logo)
    add_baselines(prs, logo, values)
    add_main_results(prs, logo, values)
    add_validation(prs, logo)
    add_selector_branch(prs, logo)
    add_llm_role(prs, logo, values)
    add_behavior_extensions(prs, logo, values)
    add_multi_objective(prs, logo)
    add_limitations(prs, logo)
    add_final(prs, logo, values)
    add_backup_map(prs, logo)
    return base.save_presentation(prs, OUTPUT_PATH)


def main() -> None:
    configure_logging()
    path = create_deck()
    LOGGER.info("Wrote refined 0611 presentation: %s", path)


if __name__ == "__main__":
    main()
