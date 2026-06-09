"""Build a dense, template-inspired academic presentation deck."""

from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.slide import Slide
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "temp" / "ppt_template_reference.pptx"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "final_project" / "NHTS_Travel_Behavior_Template_Presentation.pptx"
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_project"
MODE_DIR = PROJECT_ROOT / "outputs" / "mode_composition_extension"
PAPER_FIGURE_DIR = FINAL_DIR / "figures" / "paper_style"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "adversarial_audit"
EMU_PER_INCH = 914400
TOTAL_SLIDES = 19

LOGGER = logging.getLogger(__name__)

COLORS = {
    "navy": RGBColor(17, 42, 66),
    "navy2": RGBColor(24, 54, 80),
    "teal": RGBColor(0, 121, 107),
    "green": RGBColor(25, 135, 84),
    "blue": RGBColor(37, 99, 235),
    "orange": RGBColor(234, 88, 12),
    "red": RGBColor(190, 18, 60),
    "purple": RGBColor(91, 33, 182),
    "gray": RGBColor(100, 116, 139),
    "line": RGBColor(203, 213, 225),
    "pale": RGBColor(241, 245, 249),
    "light": RGBColor(248, 250, 252),
    "ink": RGBColor(15, 23, 42),
    "white": RGBColor(255, 255, 255),
}

METHOD_LABELS = {
    "historical_mean_only": "Historical mean",
    "historical_xgboost": "Ordinary XGBoost",
    "historical_mean_trend_shift": "Trend / indicator",
    "llm_only_trip_suppression_a1p25": "LLM-only pressure",
    "global_trip_suppression_a1p25": "Global event rule",
    "random_trip_suppression_a1p25": "Random pressure",
    "gated_trip_suppression_a1_d0p15": "Hybrid gated",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def clear_template_slides(prs: Presentation) -> None:
    slide_id_list = prs.slides._sldIdLst  # noqa: SLF001 - python-pptx has no public clear API.
    for slide_id in list(slide_id_list):
        prs.part.drop_rel(slide_id.rId)
        slide_id_list.remove(slide_id)


def iter_nested_shapes(shapes):
    for shape in shapes:
        yield shape
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_nested_shapes(shape.shapes)


def extract_template_logo() -> bytes | None:
    if not TEMPLATE_PATH.exists():
        return None
    prs = Presentation(TEMPLATE_PATH)
    for slide in prs.slides:
        for shape in iter_nested_shapes(slide.shapes):
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            width = shape.width / EMU_PER_INCH
            height = shape.height / EMU_PER_INCH
            size = len(shape.image.blob)
            if 0.7 <= width <= 1.4 and 0.35 <= height <= 0.95 and size < 80_000:
                return shape.image.blob
    return None


def blank_slide(prs: Presentation) -> Slide:
    return prs.slides.add_slide(prs.slide_layouts[6])


def set_background(slide: Slide, color: RGBColor = COLORS["white"]) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def rect(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: RGBColor,
    line: RGBColor | None = None,
    radius: bool = False,
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    return shape


def text_box(
    slide: Slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: float = 10,
    color: RGBColor = COLORS["ink"],
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.01)
    frame.margin_bottom = Inches(0.01)
    frame.word_wrap = True
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Microsoft YaHei"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    for run in paragraph.runs:
        run.font.name = "Microsoft YaHei"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def add_logo(slide: Slide, logo: bytes | None, x: float, y: float, w: float, dark: bool = False) -> None:
    if logo is None:
        text_box(slide, "TSINGHUA", x - 0.05, y + 0.05, w + 0.15, 0.22, 7, COLORS["gray"], True, PP_ALIGN.CENTER)
        return
    if dark:
        rect(slide, x - 0.04, y - 0.03, w + 0.08, 0.62, COLORS["white"], None, True)
    slide.shapes.add_picture(BytesIO(logo), Inches(x), Inches(y), width=Inches(w))


def add_picture(slide: Slide, path: Path, x: float, y: float, w: float) -> None:
    if path.exists():
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    else:
        rect(slide, x, y, w, 2.6, COLORS["pale"], COLORS["line"], True)
        text_box(slide, f"Missing figure:\n{path.name}", x + 0.1, y + 1.0, w - 0.2, 0.5, 12, COLORS["red"], True, PP_ALIGN.CENTER)


def story_box(slide: Slide, zh: str, en: str, x: float, y: float, w: float, h: float, color: RGBColor) -> None:
    rect(slide, x, y, w, h, COLORS["pale"], COLORS["line"], True)
    rect(slide, x, y, 0.055, h, color)
    text_box(slide, zh, x + 0.18, y + 0.1, w - 0.32, 0.28, 12, COLORS["ink"], True)
    text_box(slide, en, x + 0.18, y + 0.43, w - 0.32, h - 0.48, 12, COLORS["gray"])


def add_frame(
    slide: Slide,
    section: str,
    title: str,
    subtitle: str,
    page: int,
    logo: bytes | None,
    dark: bool = False,
) -> None:
    title_color = COLORS["white"] if dark else COLORS["ink"]
    sub_color = RGBColor(203, 213, 225) if dark else COLORS["gray"]
    line_color = RGBColor(148, 163, 184) if dark else COLORS["line"]
    text_box(slide, "PRE FOR Big Data and Urban Planning", 0.42, 0.12, 4.6, 0.18, 7.5, sub_color, True)
    text_box(slide, section, 0.42, 0.38, 0.55, 0.32, 14, COLORS["teal"], True)
    rect(slide, 1.05, 0.54, 0.48, 0.025, COLORS["teal"])
    text_box(slide, title, 1.66, 0.27, 8.8, 0.36, 17.5, title_color, True)
    if subtitle:
        text_box(slide, subtitle, 1.68, 0.64, 9.3, 0.25, 8.5, sub_color)
    add_logo(slide, logo, 12.05, 0.22, 0.78, dark)
    rect(slide, 0.42, 0.96, 12.25, 0.012, line_color)
    rect(slide, 0.42, 7.03, 12.25, 0.012, line_color)
    text_box(slide, "Pandemic-Aware Household Travel Behavior Prediction", 0.42, 7.08, 5.7, 0.18, 7.5, sub_color)
    text_box(slide, f"{page:02d} / {TOTAL_SLIDES}", 11.65, 7.08, 1.0, 0.18, 7.5, sub_color, True, PP_ALIGN.RIGHT)


def metric_card(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    note: str,
    color: RGBColor,
    fill: RGBColor = COLORS["light"],
) -> None:
    rect(slide, x, y, w, h, fill, COLORS["line"], True)
    rect(slide, x, y, 0.055, h, color)
    text_box(slide, label, x + 0.14, y + 0.08, w - 0.24, 0.18, 7.8, COLORS["gray"], True)
    text_box(slide, value, x + 0.14, y + 0.27, w - 0.24, 0.32, 16, color, True)
    text_box(slide, note, x + 0.14, y + 0.61, w - 0.24, 0.2, 7.5, COLORS["gray"])


def bullet_list(slide: Slide, items: list[str], x: float, y: float, w: float, h: float, size: float = 10) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(0.02)
    frame.margin_right = Inches(0.02)
    frame.word_wrap = True
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"• {item}"
        paragraph.level = 0
        paragraph.font.name = "Microsoft YaHei"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = COLORS["ink"]
        paragraph.space_after = Pt(2)


def node(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    note: str,
    color: RGBColor,
    fill: RGBColor = COLORS["white"],
) -> None:
    rect(slide, x, y, w, h, fill, color, True)
    text_box(slide, title, x + 0.08, y + 0.07, w - 0.16, 0.2, 9.2, color, True, PP_ALIGN.CENTER)
    text_box(slide, note, x + 0.1, y + 0.31, w - 0.2, h - 0.37, 7.8, COLORS["gray"], False, PP_ALIGN.CENTER)


def arrow(slide: Slide, x: float, y: float, w: float, h: float, color: RGBColor = COLORS["gray"]) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def small_table(
    slide: Slide,
    headers: list[str],
    rows: list[list[str]],
    x: float,
    y: float,
    widths: list[float],
    row_h: float = 0.34,
    font_size: float = 7.7,
    header_color: RGBColor = COLORS["navy"],
) -> None:
    for col, header in enumerate(headers):
        xx = x + sum(widths[:col])
        rect(slide, xx, y, widths[col], row_h, header_color)
        text_box(slide, header, xx + 0.04, y + 0.04, widths[col] - 0.08, row_h - 0.08, font_size, COLORS["white"], True, PP_ALIGN.CENTER)
    for row_idx, row in enumerate(rows):
        yy = y + row_h * (row_idx + 1)
        fill = COLORS["light"] if row_idx % 2 == 0 else COLORS["pale"]
        for col, value in enumerate(row):
            xx = x + sum(widths[:col])
            rect(slide, xx, yy, widths[col], row_h, fill, COLORS["line"])
            text_box(slide, value, xx + 0.05, yy + 0.035, widths[col] - 0.1, row_h - 0.07, font_size, COLORS["ink"], col == 0)


def draw_bar_chart(
    slide: Slide,
    values: list[tuple[str, float, RGBColor]],
    x: float,
    y: float,
    w: float,
    h: float,
    max_value: float,
    unit: str = "",
) -> None:
    row_h = h / len(values)
    for index, (label, value, color) in enumerate(values):
        yy = y + index * row_h
        text_box(slide, label, x, yy - 0.015, 2.25, 0.24, 7.8, COLORS["ink"])
        rect(slide, x + 2.35, yy + 0.04, w - 3.0, 0.17, RGBColor(226, 232, 240))
        width = (w - 3.0) * min(value / max_value, 1.0)
        rect(slide, x + 2.35, yy + 0.04, width, 0.17, color)
        text_box(slide, f"{value:.2f}{unit}", x + w - 0.55, yy - 0.02, 0.55, 0.24, 8, color, True, PP_ALIGN.RIGHT)


def draw_bias_strip(slide: Slide, values: list[tuple[str, float, RGBColor]], x: float, y: float, w: float, h: float) -> None:
    center = x + w * 0.52
    rect(slide, center, y, 0.015, h, COLORS["gray"])
    text_box(slide, "0", center - 0.12, y + h + 0.02, 0.24, 0.15, 7, COLORS["gray"], True, PP_ALIGN.CENTER)
    scale = (w * 0.45) / 4.8
    for idx, (label, value, color) in enumerate(values):
        yy = y + 0.18 + idx * 0.48
        text_box(slide, label, x, yy - 0.03, 1.55, 0.22, 7.5, COLORS["ink"])
        bar_w = abs(value) * scale
        if value >= 0:
            bx = center
        else:
            bx = center - bar_w
        rect(slide, bx, yy, bar_w, 0.15, color)
        text_box(slide, f"{value:+.2f}", bx + (bar_w if value >= 0 else -0.5), yy - 0.04, 0.5, 0.24, 7.5, color, True)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trip = pd.read_csv(FINAL_DIR / "final_metrics_summary.csv")
    acc = pd.read_csv(FINAL_DIR / "household_accuracy_summary.csv")
    mode = pd.read_csv(MODE_DIR / "mode_composition_metrics.csv")
    return trip, acc, mode


def metric(df: pd.DataFrame, method: str, column: str) -> float:
    return float(df.loc[df["method"] == method, column].iloc[0])


def pct_delta(before: float, after: float) -> float:
    return (before - after) / before


def add_title_slide(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide, COLORS["navy"])
    add_logo(slide, logo, 11.82, 0.34, 0.92, True)
    text_box(slide, "基于 NHTS 与 LLM 事件先验的\n疫情感知家庭出行行为预测", 0.68, 0.78, 9.6, 0.95, 25, COLORS["white"], True)
    text_box(slide, "Pandemic-Aware Household Travel Behavior Prediction", 0.72, 1.86, 8.4, 0.32, 12.5, RGBColor(203, 213, 225), True)
    text_box(slide, "LLM event priors for label-free adaptation under post-pandemic shift", 0.72, 2.23, 8.8, 0.28, 10, RGBColor(203, 213, 225))
    rect(slide, 0.72, 3.0, 11.9, 0.01, RGBColor(148, 163, 184))
    nodes = [
        ("历史 NHTS", "routine mobility"),
        ("2022 疫情冲击", "event-driven shift"),
        ("LLM 事件先验", "semantic adapter"),
        ("家庭出行行为", "trips + modes"),
    ]
    for idx, (title, note) in enumerate(nodes):
        node(slide, 0.8 + idx * 3.02, 3.35, 2.25, 0.7, title, note, COLORS["teal"], RGBColor(30, 64, 91))
        if idx < len(nodes) - 1:
            arrow(slide, 3.1 + idx * 3.02, 3.57, 0.48, 0.18, RGBColor(148, 163, 184))
    metric_card(slide, 0.82, 5.05, 2.55, 0.9, "Trip-count MAE", f"-{values['mae_reduction']:.1%}", "vs ordinary XGBoost", COLORS["blue"], COLORS["white"])
    metric_card(slide, 3.68, 5.05, 2.55, 0.9, "Weighted bias", f"{values['gated_bias']:+.3f}", "nearly unbiased", COLORS["green"], COLORS["white"])
    metric_card(slide, 6.54, 5.05, 2.55, 0.9, "Within 2 trips", f"{values['within2']:.1%}", "household tolerance", COLORS["orange"], COLORS["white"])
    metric_card(slide, 9.4, 5.05, 2.55, 0.9, "Transit MAE", f"-{values['transit_gain']:.1%}", "mode signal", COLORS["purple"], COLORS["white"])
    text_box(slide, "PRE FOR Big Data and Urban Planning", 0.72, 6.83, 4.6, 0.2, 8, RGBColor(203, 213, 225), True)
    text_box(slide, f"01 / {TOTAL_SLIDES}", 11.72, 6.83, 0.85, 0.2, 8, RGBColor(203, 213, 225), True, PP_ALIGN.RIGHT)


def add_content_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "00", "汇报结构 / Content", "从研究问题到技术路线，再到结果解释", 2, logo)
    rect(slide, 0.55, 1.45, 3.1, 4.7, COLORS["navy"])
    text_box(slide, "CONTENT", 1.02, 2.45, 2.1, 0.42, 20, COLORS["white"], True, PP_ALIGN.CENTER)
    text_box(slide, "一条主线：\n2022 不是普通预测年，而是疫情后事件冲击下的 label-free adaptation。", 0.92, 3.1, 2.38, 0.82, 10, RGBColor(203, 213, 225), False, PP_ALIGN.CENTER)
    items = [
        ("01", "Research gap", "为什么传统 cross-year transfer 会失效"),
        ("02", "Data and target system", "家庭层面预测：出行次数 + 方式结构"),
        ("03", "Method", "historical routine predictor + LLM event prior"),
        ("04", "Evaluation design", "不使用 2022 标签校准的对比体系"),
        ("05", "Insights", "准确率提升来自哪里，LLM 的真实角色是什么"),
    ]
    for idx, (num, title, note) in enumerate(items):
        y = 1.32 + idx * 0.94
        rect(slide, 4.12, y, 0.5, 0.5, COLORS["teal"], None, True)
        text_box(slide, num, 4.17, y + 0.08, 0.4, 0.22, 11, COLORS["white"], True, PP_ALIGN.CENTER)
        text_box(slide, title, 4.82, y + 0.02, 3.1, 0.22, 12, COLORS["ink"], True)
        text_box(slide, note, 4.82, y + 0.28, 6.8, 0.2, 8.5, COLORS["gray"])
        rect(slide, 4.55, y + 0.58, 7.4, 0.008, COLORS["line"])


def add_problem_slide(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "01", "研究问题 / Research Gap", "2022 不是普通跨年预测 | 2022 is not an ordinary transfer year", 3, logo)
    headers = ["Traditional framing", "Event-driven framing"]
    rows = [
        ["2017 -> 2022 as smooth temporal transfer", "2022 as post-pandemic behavioral regime"],
        ["Covariates explain most household differences", "Household covariates miss remote work / transit avoidance"],
        ["Model failure means weak tabular learner", "Model failure reveals unobserved event mechanisms"],
        ["Tune on target labels if available", "Use no 2022 labels for training/calibration"],
    ]
    small_table(slide, headers, rows, 0.68, 1.28, [5.55, 5.85], 0.38, 8.3)
    metric_card(slide, 0.72, 4.75, 2.6, 0.86, "Ordinary XGBoost", f"{values['baseline']:.2f}", "weighted MAE", COLORS["red"])
    metric_card(slide, 3.58, 4.75, 2.6, 0.86, "Systematic bias", f"{values['baseline_bias']:+.2f}", "over-predicts trips", COLORS["red"])
    metric_card(slide, 6.44, 4.75, 2.6, 0.86, "Target labels", "0", "used for calibration", COLORS["teal"])
    metric_card(slide, 9.3, 4.75, 2.6, 0.86, "Research target", "adapt", "not just fit", COLORS["blue"])
    story_box(
        slide,
        "核心问题：历史 household travel model 在 2022 会系统性高估出行，因此我们研究 label-free event adaptation，而不是单纯调参。",
        "Core question: can structured LLM event priors repair post-pandemic shift without using 2022 target labels for calibration?",
        0.82,
        5.9,
        10.95,
        0.78,
        COLORS["teal"],
    )


def add_data_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "02", "数据与目标 / Data and Target System", "一个 household behavior system，三个 linked outputs", 4, logo)
    years = [("2001", "history"), ("2009", "history"), ("2017", "history + mode base"), ("2022", "post-pandemic eval")]
    for idx, (year, label) in enumerate(years):
        x = 0.75 + idx * 2.65
        rect(slide, x, 1.52, 1.0, 0.55, COLORS["teal"] if year != "2022" else COLORS["orange"], None, True)
        text_box(slide, year, x + 0.08, 1.62, 0.84, 0.22, 12, COLORS["white"], True, PP_ALIGN.CENTER)
        text_box(slide, label, x - 0.25, 2.13, 1.5, 0.22, 8, COLORS["gray"], False, PP_ALIGN.CENTER)
        if idx < len(years) - 1:
            arrow(slide, x + 1.15, 1.72, 0.72, 0.16, COLORS["gray"])
    headers = ["Output", "Construction", "Planning meaning", "Main metric"]
    rows = [
        ["Trip generation", "CNTTDHH at household level", "How much travel demand", "weighted MAE / bias"],
        ["Mode composition", "TRPTRANS -> household share vector", "How demand is distributed", "TV distance / share MAE"],
        ["Mode-specific trips", "predicted trips x predicted mode share", "Trips by mode for planning", "derived output"],
    ]
    small_table(slide, headers, rows, 0.72, 3.02, [2.35, 3.5, 3.05, 2.35], 0.42, 8.3, COLORS["navy2"])
    story_box(
        slide,
        "目标不是只预测 CNTTDHH，而是构造 household travel behavior：出行多少次、用什么方式、每种方式多少次。",
        "The prediction target is a household behavior system: trip generation, mode composition, and derived mode-specific trip counts.",
        0.82,
        5.62,
        10.95,
        0.78,
        COLORS["blue"],
    )


def add_route_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "03", "技术路线 / Technical Route", "Routine predictor + LLM event prior + fixed no-label adapter", 5, logo)
    lanes = [
        ("Historical routine lane", 1.35, COLORS["blue"], [("NHTS 2001/09/17", "pre-2022 labels"), ("Feature harmonization", "stable household covariates"), ("XGBoost CUDA", "f_hist(X)")]),
        ("LLM event lane", 3.05, COLORS["orange"], [("2022 household profiles", "cohort aggregation"), ("GPT-5.5 prompts", "structured event reasoning"), ("Event priors", "trip suppression / transit avoidance")]),
        ("No-label adapter lane", 4.75, COLORS["green"], [("Fixed correction", "no 2022 label tuning"), ("Predicted behavior", "trip count + mode shares"), ("Final evaluation", "2022 labels only here")]),
    ]
    for lane, y, color, steps in lanes:
        text_box(slide, lane, 0.58, y + 0.18, 1.55, 0.25, 9, color, True)
        rect(slide, 2.1, y + 0.33, 9.95, 0.01, COLORS["line"])
        for idx, (title, note) in enumerate(steps):
            x = 2.35 + idx * 3.15
            node(slide, x, y, 2.3, 0.66, title, note, color)
            if idx < 2:
                arrow(slide, x + 2.38, y + 0.24, 0.45, 0.14, COLORS["gray"])
    rect(slide, 0.78, 6.08, 11.35, 0.5, COLORS["pale"], COLORS["line"], True)
    text_box(slide, "Core rule", 0.98, 6.17, 1.1, 0.2, 9, COLORS["teal"], True)
    text_box(slide, "prediction = historical routine prediction x clip(1 - alpha x event_pressure, min_factor, 1)", 2.05, 6.14, 9.5, 0.26, 11.5, COLORS["ink"], True)


def add_llm_prior_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "04", "LLM 事件先验 / LLM Event Prior", "约束 LLM 解释 event response，而不是直接预测标签", 6, logo)
    headers = ["Prior field", "Interpretation", "Expected 2022 effect"]
    rows = [
        ["trip_suppression", "overall travel reduction pressure", "lower total trip count"],
        ["remote_work", "work-trip substitution probability", "fewer commute-related trips"],
        ["transit_avoidance", "public-transit perceived risk", "lower transit share"],
        ["delivery_substitution", "online replacement of shopping trips", "fewer discretionary trips"],
        ["recovery_sensitivity", "ability to return to routine mobility", "heterogeneous rebound"],
        ["confidence", "LLM self-reported certainty", "audit / filtering signal"],
    ]
    small_table(slide, headers, rows, 0.68, 1.18, [2.35, 4.45, 4.3], 0.36, 8.1, COLORS["navy2"])
    small_table(
        slide,
        ["Allowed in prompt", "Excluded to avoid leakage"],
        [
            ["cohort-level household profile", "CNTTDHH / trip count label"],
            ["urban context, vehicles, workers", "sample weights and household IDs"],
            ["generic 2022 pandemic context", "aggregate target-year outcomes"],
        ],
        0.82,
        4.85,
        [5.15, 5.15],
        0.34,
        8.2,
        COLORS["teal"],
    )
    text_box(slide, "Design implication", 0.95, 6.32, 1.55, 0.22, 9.5, COLORS["orange"], True)
    text_box(slide, "The LLM contributes event semantics; the historical model keeps household-level numerical grounding.", 2.45, 6.28, 8.8, 0.26, 9.5, COLORS["ink"], True)


def add_comparison_slide(prs: Presentation, logo: bytes | None, trip: pd.DataFrame) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "05", "对比设计 / Evaluation Design", "Every baseline answers a different design question", 7, logo)
    rows = [
        ["Historical mean", "No", "No", "No", "naive historical level", f"{metric(trip, 'historical_mean_only', 'weighted_mae'):.2f}"],
        ["Ordinary XGBoost", "Yes", "No", "No", "routine transfer baseline", f"{metric(trip, 'historical_xgboost', 'weighted_mae'):.2f}"],
        ["Trend / indicator", "Yes", "No", "No", "simple non-semantic shift", f"{metric(trip, 'historical_mean_trend_shift', 'weighted_mae'):.2f}"],
        ["LLM-only pressure", "No", "Yes", "No", "event direction without baseline", f"{metric(trip, 'llm_only_trip_suppression_a1p25', 'weighted_mae'):.2f}"],
        ["Global event rule", "Yes", "No", "No", "common pandemic downscaling", f"{metric(trip, 'global_trip_suppression_a1p25', 'weighted_mae'):.2f}"],
        ["Random pressure", "Yes", "No", "No", "placebo heterogeneity", f"{metric(trip, 'random_trip_suppression_a1p25', 'weighted_mae'):.2f}"],
        ["Hybrid gated", "Yes", "Yes", "No", "proposed semantic adapter", f"{metric(trip, 'gated_trip_suppression_a1_d0p15', 'weighted_mae'):.2f}"],
    ]
    small_table(
        slide,
        ["Method", "pre-2022 labels", "LLM prior", "2022 labels", "Role", "wMAE"],
        rows,
        0.52,
        1.2,
        [2.05, 1.25, 1.0, 1.05, 4.0, 0.9],
        0.34,
        7.4,
    )
    story_box(
        slide,
        "公平性原则：2022 标签只用于最后 evaluation，主方法使用固定 no-label 参数，避免把测试年标签隐含用于调参。",
        "Fairness rule: 2022 labels are held out until final evaluation; the primary method uses fixed parameters rather than target-year calibration.",
        0.82,
        5.92,
        10.95,
        0.72,
        COLORS["teal"],
    )


def add_results_slide(prs: Presentation, logo: bytes | None, trip: pd.DataFrame, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "06", "主结果 / Main Trip-Count Results", "Hybrid adapters reduce error while controlling bias", 8, logo)
    chart_values = [
        (METHOD_LABELS["historical_mean_only"], metric(trip, "historical_mean_only", "weighted_mae"), COLORS["gray"]),
        (METHOD_LABELS["historical_xgboost"], values["baseline"], COLORS["red"]),
        (METHOD_LABELS["historical_mean_trend_shift"], metric(trip, "historical_mean_trend_shift", "weighted_mae"), COLORS["orange"]),
        (METHOD_LABELS["llm_only_trip_suppression_a1p25"], values["llm_only"], COLORS["purple"]),
        (METHOD_LABELS["global_trip_suppression_a1p25"], metric(trip, "global_trip_suppression_a1p25", "weighted_mae"), COLORS["teal"]),
        (METHOD_LABELS["gated_trip_suppression_a1_d0p15"], values["gated"], COLORS["green"]),
    ]
    draw_bar_chart(slide, chart_values, 0.65, 1.28, 6.25, 3.28, 6.0)
    rows = [
        ["Ordinary XGBoost", f"{values['baseline']:.3f}", f"{values['baseline_bias']:+.3f}", f"{values['baseline_r2']:.3f}"],
        ["LLM-only pressure", f"{values['llm_only']:.3f}", f"{values['llm_only_bias']:+.3f}", f"{values['llm_only_r2']:.3f}"],
        ["Hybrid gated", f"{values['gated']:.3f}", f"{values['gated_bias']:+.3f}", f"{values['gated_r2']:.3f}"],
    ]
    small_table(slide, ["Method", "wMAE", "wBias", "wR2"], rows, 7.15, 1.28, [2.05, 0.9, 0.9, 0.8], 0.38, 7.8, COLORS["teal"])
    metric_card(slide, 7.16, 3.32, 2.05, 0.82, "MAE reduction", f"-{values['mae_reduction']:.1%}", "primary vs XGBoost", COLORS["green"])
    metric_card(slide, 9.45, 3.32, 2.05, 0.82, "Bias reduction", "-99.4%", "absolute weighted bias", COLORS["blue"])
    story_box(
        slide,
        "结果主线：普通 XGBoost 学到的是正常时期出行强度，所以 2022 明显高估；LLM prior 提供缺失的疫情方向。",
        "Main insight: the historical model transfers routine mobility into a post-pandemic year, while the LLM prior supplies the missing event direction.",
        0.82,
        5.35,
        10.95,
        0.82,
        COLORS["orange"],
    )


def add_tradeoff_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "06A", "准确率-偏差权衡 / Accuracy-Bias Tradeoff", "论文图1：主方法同时降低误差和系统性偏差 | Paper-style Figure 1", 9, logo)
    add_picture(slide, PAPER_FIGURE_DIR / "mae_bias_tradeoff.png", 0.72, 1.28, 6.75)
    story_box(
        slide,
        "这张图要讲的不是“哪个点最低”这么简单，而是说明我们的方法同时移动到低 MAE 和低 bias 的区域。",
        "The proposed adapter improves the accuracy-bias Pareto position instead of only trading one metric for another.",
        7.75,
        1.42,
        4.1,
        1.28,
        COLORS["green"],
    )
    small_table(
        slide,
        ["Reading guide", "Story"],
        [
            ["x-axis", "closer to zero means lower systematic shift"],
            ["y-axis", "lower means smaller household trip-count error"],
            ["point size", "weighted RMSE, reflecting large-error risk"],
            ["hybrid gated", "near the lower-left region"],
        ],
        7.75,
        3.12,
        [1.45, 3.1],
        0.38,
        8.3,
        COLORS["teal"],
    )


def add_error_distribution_figure_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "06B", "误差分布与容忍度 / Error Distribution and Tolerance", "论文图2：从平均误差走向 household-level 可解释准确率", 10, logo)
    add_picture(slide, PAPER_FIGURE_DIR / "error_distribution.png", 0.65, 1.2, 5.85)
    add_picture(slide, PAPER_FIGURE_DIR / "tolerance_curve.png", 6.85, 1.2, 5.65)
    story_box(
        slide,
        "左图说明历史模型的误差分布明显向正方向偏移；右图说明 hybrid 方法在不同容忍阈值下都有更高的家庭覆盖率。",
        "The distributional view explains why weighted MAE improves, while the tolerance curve translates the result into household-level prediction usability.",
        0.85,
        5.72,
        10.95,
        0.82,
        COLORS["orange"],
    )


def add_event_heterogeneity_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "07", "事件异质性 / Event Heterogeneity", "论文图3：LLM prior 是否真的在表达疫情机制", 11, logo)
    add_picture(slide, PAPER_FIGURE_DIR / "pressure_quintile_gain.png", 0.65, 1.2, 5.85)
    add_picture(slide, PAPER_FIGURE_DIR / "event_prior_heatmap.png", 7.0, 1.13, 4.95)
    story_box(
        slide,
        "这页的故事是：LLM event pressure 不是装饰变量。它在不同 pressure strata 中都能带来修正，并且内部 prior 之间有可解释的相关结构。",
        "The event-prior structure is auditable: pressure-stratified gains and prior correlations show semantic consistency rather than arbitrary numerical fitting.",
        0.85,
        5.88,
        10.95,
        0.72,
        COLORS["teal"],
    )


def add_subgroup_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "07B", "分组鲁棒性 / Subgroup Robustness", "论文图4：哪些家庭群体从事件修正中获益更多", 13, logo)
    add_picture(slide, PAPER_FIGURE_DIR / "subgroup_gain_top.png", 0.72, 1.25, 6.6)
    story_box(
        slide,
        "这张图用于回答老师可能会问的 fairness / heterogeneity 问题：提升不是只来自一个总体均值，而是在多个 household subgroup 中都能观察到。",
        "The subgroup view reframes the result as heterogeneous adaptation, which is more convincing than only reporting one aggregate test-set metric.",
        7.65,
        1.38,
        4.25,
        1.32,
        COLORS["purple"],
    )
    small_table(
        slide,
        ["How to present", "Caveat"],
        [
            ["Use it as diagnostic evidence", "not a causal subgroup claim"],
            ["Focus on broad gain pattern", "codes need explanation in appendix"],
            ["Connect to planning", "which household segments are harder"],
        ],
        7.65,
        3.15,
        [1.75, 2.65],
        0.42,
        8.5,
        COLORS["navy2"],
    )


def add_adversarial_audit_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "07A", "对抗性审计 / Adversarial Audit", "主动回应审稿人最可能质疑的地方", 12, logo)
    add_picture(slide, AUDIT_DIR / "permutation_null_mae.png", 0.72, 1.22, 6.2)
    story_box(
        slide,
        "审计结论：真实 LLM cohort pressure 明显优于 500 次随机置换，但 global event pressure 本身也很强，所以不能把贡献夸大成强个体化 LLM 预测。",
        "Audit verdict: cohort-specific LLM pressure beats permutation controls, but global event downscaling explains much of the gain; the defensible claim is label-free event adaptation.",
        7.45,
        1.35,
        4.45,
        1.42,
        COLORS["red"],
    )
    small_table(
        slide,
        ["Reviewer concern", "Answer"],
        [
            ["Is global rule enough?", "Strong baseline; LLM ranking adds incremental value"],
            ["Is it label leakage?", "LLM inputs exclude target, weights, and IDs"],
            ["Can LLM replace model?", "No; hybrid beats LLM-only"],
            ["Is mode solved?", "No; transit-specific extension only"],
        ],
        7.45,
        3.25,
        [2.0, 2.75],
        0.4,
        8.2,
        COLORS["navy2"],
    )


def add_error_insight_slide(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "08", "误差洞察 / Error Insight", "主收益来自消除 post-pandemic over-prediction", 14, logo)
    metric_card(slide, 0.75, 1.28, 2.45, 0.9, "Exact rounded", f"{values['exact']:.1%}", "household accuracy", COLORS["blue"])
    metric_card(slide, 3.48, 1.28, 2.45, 0.9, "Within 2 trips", f"{values['within2']:.1%}", "practical tolerance", COLORS["green"])
    metric_card(slide, 6.21, 1.28, 2.45, 0.9, "Within 3 trips", f"{values['within3']:.1%}", "broad tolerance", COLORS["orange"])
    metric_card(slide, 8.94, 1.28, 2.45, 0.9, "Best sensitivity", f"{values['best']:.3f}", "wMAE, not main claim", COLORS["purple"])
    text_box(slide, "Weighted bias comparison", 0.82, 2.75, 2.3, 0.24, 10.5, COLORS["teal"], True)
    draw_bias_strip(
        slide,
        [
            ("Historical mean", values["mean_bias"], COLORS["gray"]),
            ("Ordinary XGBoost", values["baseline_bias"], COLORS["red"]),
            ("Trend / indicator", values["trend_bias"], COLORS["orange"]),
            ("LLM-only pressure", values["llm_only_bias"], COLORS["purple"]),
            ("Hybrid gated", values["gated_bias"], COLORS["green"]),
        ],
        0.82,
        3.12,
        5.1,
        2.35,
    )
    headers = ["Observation", "Interpretation for paper story"]
    rows = [
        ["XGBoost has strong household signal but positive bias", "routine mobility from history is too high for 2022"],
        ["LLM-only improves direction but lacks calibration", "event semantics alone cannot replace household baseline"],
        ["Hybrid gated is nearly unbiased", "separating routine demand and event response is the key design"],
    ]
    small_table(slide, headers, rows, 6.55, 3.02, [2.45, 3.1], 0.42, 8.0, COLORS["navy2"])


def add_llm_role_slide(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "09", "LLM 的角色 / What the LLM Adds", "事件方向有用，但需要和 household baseline 结合", 15, logo)
    rect(slide, 1.15, 1.28, 10.35, 4.48, COLORS["light"], COLORS["line"])
    rect(slide, 6.3, 1.28, 0.012, 4.48, COLORS["line"])
    rect(slide, 1.15, 3.52, 10.35, 0.012, COLORS["line"])
    text_box(slide, "low event semantics", 1.45, 1.02, 2.4, 0.22, 8, COLORS["gray"], True)
    text_box(slide, "high event semantics", 8.0, 1.02, 2.4, 0.22, 8, COLORS["gray"], True)
    text_box(slide, "weak household grounding", 0.18, 2.0, 0.75, 0.46, 8, COLORS["gray"], True, PP_ALIGN.CENTER)
    text_box(slide, "strong household grounding", 0.18, 4.35, 0.75, 0.46, 8, COLORS["gray"], True, PP_ALIGN.CENTER)
    node(slide, 1.75, 1.82, 3.4, 0.8, "Historical mean", "neither household heterogeneity nor event semantics\nwMAE 5.51", COLORS["gray"])
    node(slide, 7.15, 1.82, 3.4, 0.8, "LLM-only pressure", f"event direction without household baseline\nwMAE {values['llm_only']:.2f}", COLORS["purple"])
    node(slide, 1.75, 4.1, 3.4, 0.8, "Ordinary XGBoost", f"household grounding without pandemic semantics\nwMAE {values['baseline']:.2f}", COLORS["red"])
    node(slide, 7.15, 4.1, 3.4, 0.8, "Hybrid gated", f"household baseline + event prior\nwMAE {values['gated']:.2f}", COLORS["green"])
    rect(slide, 1.1, 6.1, 10.45, 0.48, COLORS["pale"], COLORS["line"], True)
    text_box(slide, "Paper sentence", 1.3, 6.2, 1.35, 0.2, 9, COLORS["orange"], True)
    text_box(slide, "LLM is not a black-box travel model; it is a semantic event adapter for rare temporal shocks.", 2.58, 6.15, 8.45, 0.28, 9.2, COLORS["ink"], True)


def add_mode_slide(prs: Presentation, logo: bytes | None, mode: pd.DataFrame, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "10", "方式结构扩展 / Mode Composition Extension", "同一个事件先验也能解释 transit-specific shift", 16, logo)
    modes = ["private", "walk", "bike", "transit", "taxi", "other"]
    for idx, mode_name in enumerate(modes):
        x = 0.78 + idx * 1.2
        color = COLORS["orange"] if mode_name == "transit" else COLORS["blue"]
        rect(slide, x, 1.42, 0.82, 0.42, color, None, True)
        text_box(slide, mode_name, x + 0.04, 1.53, 0.74, 0.16, 7.2, COLORS["white"], True, PP_ALIGN.CENTER)
    text_box(slide, "household mode-share vector", 2.25, 2.03, 3.25, 0.22, 9, COLORS["gray"], True, PP_ALIGN.CENTER)
    rows = [
        ["Historical XGBoost", f"{values['mode_tv_base']:.4f}", f"{values['transit_base']:.4f}", "routine mode pattern"],
        ["XGBoost + LLM transit prior", f"{values['mode_tv_llm']:.4f}", f"{values['transit_llm']:.4f}", "event-aware transit correction"],
        ["Improvement", f"{pct_delta(values['mode_tv_base'], values['mode_tv_llm']):.1%}", f"{values['transit_gain']:.1%}", "clearer on transit"],
    ]
    small_table(slide, ["Method", "weighted TV", "transit MAE", "Interpretation"], rows, 0.78, 2.72, [2.25, 1.1, 1.1, 2.15], 0.42, 8.0, COLORS["teal"])
    add_picture(slide, PAPER_FIGURE_DIR / "mode_component_mae.png", 7.25, 1.34, 4.8)
    story_box(
        slide,
        "方式结构的总体提升不如出行次数大，但 transit component 明显改善，正好对应疫情期间公共交通规避这个机制。",
        "Mode composition is a weaker but useful extension: transit-share improvement supports the semantic validity of the COVID event prior.",
        0.82,
        5.62,
        10.95,
        0.82,
        COLORS["orange"],
    )


def add_scalability_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "11", "可扩展性与审计 / Scalability and Auditability", "Cohort prompting makes LLM use measurable and reviewable", 17, logo)
    steps = [
        ("7,893 households", "2022 rows"),
        ("1,327 cohorts", "aggregated profiles"),
        ("15 concurrency", "full generation setting"),
        ("validated JSON", "normalized priors"),
        ("fixed adapter", "reproducible output"),
    ]
    for idx, (title, note) in enumerate(steps):
        x = 0.72 + idx * 2.3
        node(slide, x, 1.55, 1.75, 0.75, title, note, COLORS["teal"] if idx < 2 else COLORS["orange"])
        if idx < len(steps) - 1:
            arrow(slide, x + 1.82, 1.82, 0.32, 0.14, COLORS["gray"])
    metric_card(slide, 1.0, 3.08, 2.4, 0.86, "Request reduction", "83.2%", "5.95x fewer prompts", COLORS["blue"])
    metric_card(slide, 3.85, 3.08, 2.4, 0.86, "Invalid records", "0", "after validation", COLORS["green"])
    metric_card(slide, 6.7, 3.08, 2.4, 0.86, "Re-runnable", "Yes", "fixed adapter rule", COLORS["purple"])
    headers = ["Risk", "Control"]
    rows = [
        ["LLM hallucination", "structured schema + numeric range validation"],
        ["target leakage", "no target labels / weights / household IDs in prompt"],
        ["cost explosion", "cohort-level prompting instead of per-household prompting"],
        ["unreproducible tuning", "main rule uses fixed no-label parameters"],
    ]
    small_table(slide, headers, rows, 1.0, 4.55, [3.2, 6.6], 0.35, 8.1, COLORS["navy2"])


def add_contribution_slide(prs: Presentation, logo: bytes | None) -> None:
    slide = blank_slide(prs)
    set_background(slide)
    add_frame(slide, "12", "学术故事 / Academic Story", "From a course project to a defensible paper narrative", 18, logo)
    small_table(
        slide,
        ["Contribution", "Why it matters"],
        [
            ["Event-driven temporal adaptation framing", "turns 2022 prediction into a distribution-shift research problem"],
            ["LLM as event-prior generator", "uses LLM generalization without asking it to hallucinate exact labels"],
            ["Label-free hybrid adapter", "separates routine travel demand from pandemic response"],
            ["Behavior-system output", "covers both trip generation and mode composition"],
        ],
        0.72,
        1.18,
        [3.3, 7.5],
        0.43,
        8.2,
        COLORS["teal"],
    )
    small_table(
        slide,
        ["Current limitation", "How to discuss it honestly"],
        [
            ["Cohort heterogeneity gain is smaller than global event downscaling", "the strongest signal is event-level trip suppression"],
            ["Mode-composition gain is concentrated in transit", "this is expected because transit avoidance is the clearest COVID channel"],
            ["No prospective external event feed yet", "future work can use news/mobility indices available before survey release"],
        ],
        0.72,
        4.38,
        [3.3, 7.5],
        0.43,
        8.1,
        COLORS["orange"],
    )


def add_final_slide(prs: Presentation, logo: bytes | None, values: dict[str, float]) -> None:
    slide = blank_slide(prs)
    set_background(slide, COLORS["navy"])
    add_frame(slide, "END", "总结 / Final Takeaway", "A clean story: routine mobility + event semantics", 19, logo, True)
    rect(slide, 0.82, 1.45, 11.15, 2.2, RGBColor(30, 64, 91), RGBColor(71, 85, 105), True)
    text_box(slide, "我们研究的不是“LLM 直接预测出行次数”，而是：\n当 2022 疫情后分布变化打破历史连续性时，能否用 LLM 的事件泛化能力，为传统 household travel model 提供无标签修正先验。", 1.12, 1.73, 10.5, 0.9, 15, COLORS["white"], True, PP_ALIGN.CENTER)
    metric_card(slide, 1.0, 4.25, 2.45, 0.9, "Accuracy", f"-{values['mae_reduction']:.1%}", "weighted MAE", COLORS["blue"], COLORS["white"])
    metric_card(slide, 3.85, 4.25, 2.45, 0.9, "Bias", f"{values['gated_bias']:+.3f}", "weighted bias", COLORS["green"], COLORS["white"])
    metric_card(slide, 6.7, 4.25, 2.45, 0.9, "Scope", "2 tasks", "trip count + modes", COLORS["orange"], COLORS["white"])
    metric_card(slide, 9.55, 4.25, 2.45, 0.9, "LLM role", "adapter", "not standalone predictor", COLORS["purple"], COLORS["white"])
    text_box(slide, "Suggested one-sentence claim", 1.02, 6.08, 2.25, 0.22, 9.5, RGBColor(203, 213, 225), True)
    text_box(slide, "LLM event priors enable label-free adaptation of household travel behavior prediction under rare post-pandemic temporal shift.", 3.05, 6.02, 8.6, 0.32, 10, COLORS["white"], True)


def create_deck() -> Path:
    logo = extract_template_logo()
    template = TEMPLATE_PATH if TEMPLATE_PATH.exists() else None
    prs = Presentation(template) if template else Presentation()
    if not template:
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
    clear_template_slides(prs)
    trip, acc, mode = load_results()

    baseline = metric(trip, "historical_xgboost", "weighted_mae")
    gated = metric(trip, "gated_trip_suppression_a1_d0p15", "weighted_mae")
    llm_only = metric(trip, "llm_only_trip_suppression_a1p25", "weighted_mae")
    transit_base = metric(mode, "historical_xgboost", "transit_share_weighted_mae")
    transit_llm = metric(mode, "llm_transit_avoidance_a1", "transit_share_weighted_mae")
    mode_tv_base = metric(mode, "historical_xgboost", "weighted_total_variation")
    mode_tv_llm = metric(mode, "llm_transit_avoidance_a1", "weighted_total_variation")
    main_method = "gated_trip_suppression_a1_d0p15"
    values = {
        "baseline": baseline,
        "baseline_bias": metric(trip, "historical_xgboost", "weighted_bias"),
        "baseline_r2": metric(trip, "historical_xgboost", "weighted_r2"),
        "gated": gated,
        "gated_bias": metric(trip, main_method, "weighted_bias"),
        "gated_r2": metric(trip, main_method, "weighted_r2"),
        "llm_only": llm_only,
        "llm_only_bias": metric(trip, "llm_only_trip_suppression_a1p25", "weighted_bias"),
        "llm_only_r2": metric(trip, "llm_only_trip_suppression_a1p25", "weighted_r2"),
        "mean_bias": metric(trip, "historical_mean_only", "weighted_bias"),
        "trend_bias": metric(trip, "historical_mean_trend_shift", "weighted_bias"),
        "best": metric(trip, "llm_trip_suppression_a1p25", "weighted_mae"),
        "mae_reduction": pct_delta(baseline, gated),
        "exact": metric(acc, main_method, "exact_rounded_accuracy"),
        "within2": metric(acc, main_method, "within_2_trips"),
        "within3": metric(acc, main_method, "within_3_trips"),
        "transit_base": transit_base,
        "transit_llm": transit_llm,
        "transit_gain": pct_delta(transit_base, transit_llm),
        "mode_tv_base": mode_tv_base,
        "mode_tv_llm": mode_tv_llm,
    }

    add_title_slide(prs, logo, values)
    add_content_slide(prs, logo)
    add_problem_slide(prs, logo, values)
    add_data_slide(prs, logo)
    add_route_slide(prs, logo)
    add_llm_prior_slide(prs, logo)
    add_comparison_slide(prs, logo, trip)
    add_results_slide(prs, logo, trip, values)
    add_tradeoff_slide(prs, logo)
    add_error_distribution_figure_slide(prs, logo)
    add_event_heterogeneity_slide(prs, logo)
    add_adversarial_audit_slide(prs, logo)
    add_subgroup_slide(prs, logo)
    add_error_insight_slide(prs, logo, values)
    add_llm_role_slide(prs, logo, values)
    add_mode_slide(prs, logo, mode, values)
    add_scalability_slide(prs, logo)
    add_contribution_slide(prs, logo)
    add_final_slide(prs, logo, values)

    return save_presentation(prs, OUTPUT_PATH)


def save_presentation(prs: Presentation, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(output_path)
        return output_path
    except PermissionError:
        for index in range(2, 20):
            fallback = output_path.with_name(f"{output_path.stem}_v{index}{output_path.suffix}")
            try:
                prs.save(fallback)
                LOGGER.warning("Primary PPT is locked; wrote fallback file: %s", fallback)
                return fallback
            except PermissionError:
                continue
        raise


def main() -> None:
    configure_logging()
    path = create_deck()
    LOGGER.info("Wrote template-based presentation: %s", path)


if __name__ == "__main__":
    main()
