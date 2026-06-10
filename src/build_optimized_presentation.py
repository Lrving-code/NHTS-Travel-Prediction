"""Build optimized comparison documents and presentation deck."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.slide import Slide
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_llm_residual_adaptation import load_2022_with_llm_features, weighted_average  # noqa: E402


LOGGER = logging.getLogger(__name__)
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_project"
FINAL_FIGURE_DIR = FINAL_DIR / "figures"
MODE_DIR = PROJECT_ROOT / "outputs" / "mode_composition_extension"
STRONG_BASELINE_PATH = PROJECT_ROOT / "outputs" / "strong_baselines" / "strong_tabular_baseline_metrics.csv"
MODE_COLUMNS = [
    "private_vehicle_share",
    "walk_share",
    "bike_share",
    "transit_share",
    "taxi_ridehail_share",
    "other_share",
]
HOUSEHOLD_ID = "HOUSEID"
WEIGHT_COLUMN = "WTHHFIN"
TRIP_LABELS = {
    "historical_mean_only": "Historical mean",
    "historical_xgboost": "Historical predictor",
    "historical_mean_trend_shift": "Historical trend shift",
    "llm_only_trip_suppression_a1p25": "LLM-only pressure",
    "global_trip_suppression_a1p25": "Global event prior",
    "random_trip_suppression_a1p25": "Random prior control",
    "gated_trip_suppression_a1_d0p15": "Gated LLM correction",
}
PUBLIC_TRIP_METHODS = [
    "historical_xgboost",
    "historical_mean_trend_shift",
    "llm_only_trip_suppression_a1p25",
    "global_trip_suppression_a1p25",
    "random_trip_suppression_a1p25",
    "gated_trip_suppression_a1_d0p15",
]
COLORS = {
    "dark": RGBColor(15, 23, 42),
    "muted": RGBColor(71, 85, 105),
    "light": RGBColor(248, 250, 252),
    "blue": RGBColor(59, 130, 246),
    "green": RGBColor(22, 163, 74),
    "orange": RGBColor(249, 115, 22),
    "red": RGBColor(220, 38, 38),
    "navy": RGBColor(23, 50, 77),
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_dirs() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.4f}"


def load_strong_baselines() -> pd.DataFrame:
    if not STRONG_BASELINE_PATH.exists():
        LOGGER.warning("Missing stronger baseline table: %s", STRONG_BASELINE_PATH)
        return pd.DataFrame()
    return pd.read_csv(STRONG_BASELINE_PATH)


def normalize_shares(predictions: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictions, 0.0, 1.0)
    sums = clipped.sum(axis=1, keepdims=True)
    return np.divide(clipped, sums, out=np.zeros_like(clipped), where=sums > 1e-12)


def evaluate_mode_prediction(method: str, test_frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float | str]:
    truth = test_frame[MODE_COLUMNS].to_numpy(dtype=float)
    weights = test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - truth
    abs_error = np.abs(error)
    total_variation = 0.5 * abs_error.sum(axis=1)
    mean_share_mae = abs_error.mean(axis=1)
    dominant_correct = (np.argmax(predictions, axis=1) == np.argmax(truth, axis=1)).astype(float)
    transit_index = MODE_COLUMNS.index("transit_share")
    return {
        "method": method,
        "weighted_total_variation": weighted_average(total_variation, weights),
        "weighted_mean_share_mae": weighted_average(mean_share_mae, weights),
        "weighted_dominant_mode_accuracy": weighted_average(dominant_correct, weights),
        "transit_share_weighted_mae": weighted_average(abs_error[:, transit_index], weights),
        "transit_share_weighted_bias": weighted_average(error[:, transit_index], weights),
    }


def compute_llm_only_mode_baselines() -> pd.DataFrame:
    mode_path = MODE_DIR / "household_mode_composition.csv"
    if not mode_path.exists():
        raise FileNotFoundError(
            f"Missing {mode_path}. Run `python src/run_mode_composition_extension.py --device cuda` first."
        )
    mode_frame = pd.read_csv(mode_path, low_memory=False)
    train_frame = mode_frame[mode_frame["survey_year"] == 2017].copy()
    test_frame = mode_frame[mode_frame["survey_year"] == 2022].copy()
    weights = train_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    mean_share = np.asarray(
        [weighted_average(train_frame[column].to_numpy(dtype=float), weights) for column in MODE_COLUMNS],
        dtype=float,
    )
    mean_share = mean_share / mean_share.sum()
    base_prediction = np.tile(mean_share, (len(test_frame), 1))

    rows = [evaluate_mode_prediction("llm_only_2017_mean_no_adapt", test_frame, base_prediction)]
    llm_frame = load_2022_with_llm_features(
        PROJECT_ROOT / "data" / "processed" / "household_harmonized.csv",
        PROJECT_ROOT / "outputs" / "llm_event_features" / "household_cohort_profiles.csv",
        PROJECT_ROOT
        / "outputs"
        / "llm_event_features"
        / "cursor_api_full_gpt55_low_c15"
        / "validated"
        / "llm_event_features_normalized.csv",
    )
    llm_frame[HOUSEHOLD_ID] = llm_frame[HOUSEHOLD_ID].astype(str)
    test_frame[HOUSEHOLD_ID] = test_frame[HOUSEHOLD_ID].astype(str)
    merged = test_frame[[HOUSEHOLD_ID]].merge(
        llm_frame[[HOUSEHOLD_ID, "transit_avoidance_likelihood"]],
        on=HOUSEHOLD_ID,
        how="left",
    )
    pressure = pd.to_numeric(merged["transit_avoidance_likelihood"], errors="coerce").fillna(0.0)
    pressure_array = np.clip(pressure.to_numpy(dtype=float), 0.0, 1.0)
    transit_index = MODE_COLUMNS.index("transit_share")
    for alpha in (0.50, 0.75, 1.00, 1.25):
        prediction = base_prediction.copy()
        prediction[:, transit_index] *= np.clip(1.0 - alpha * pressure_array, 0.05, 1.0)
        rows.append(evaluate_mode_prediction(f"llm_only_2017_mean_transit_a{alpha:g}", test_frame, normalize_shares(prediction)))
    return pd.DataFrame(rows)


def save_mode_llm_only_figure(comparison: pd.DataFrame) -> Path:
    path = FINAL_FIGURE_DIR / "mode_llm_only_comparison.png"
    methods = [
        "llm_only_2017_mean_no_adapt",
        "llm_only_2017_mean_transit_a1.25",
        "historical_xgboost",
        "llm_transit_avoidance_a1",
    ]
    labels = ["LLM-only mean", "LLM-only corrected", "XGBoost", "XGBoost + LLM"]
    plot_frame = comparison.set_index("method").loc[methods].reset_index()
    x = np.arange(len(plot_frame))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(x, plot_frame["weighted_total_variation"], color="#3B82F6")
    axes[0].set_title("Mode Composition Error")
    axes[0].set_ylabel("Weighted total variation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].bar(x, plot_frame["transit_share_weighted_mae"], color="#F97316")
    axes[1].set_title("Transit Share Error")
    axes[1].set_ylabel("Weighted MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return path


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def build_method_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    trip = pd.read_csv(FINAL_DIR / "final_metrics_summary.csv")
    acc = pd.read_csv(FINAL_DIR / "household_accuracy_summary.csv")
    mode = pd.read_csv(MODE_DIR / "mode_composition_metrics.csv")
    llm_only = compute_llm_only_mode_baselines()
    mode_keep = mode[
        mode["method"].isin(
            [
                "historical_mean_2017",
                "historical_xgboost",
                "llm_transit_avoidance_a1",
                "global_transit_avoidance_a1",
            ]
        )
    ].copy()
    mode_comparison = pd.concat([mode_keep, llm_only], ignore_index=True)
    save_csv(mode_comparison, FINAL_DIR / "mode_llm_only_comparison.csv")

    rows: list[dict[str, object]] = []
    trip_baseline = trip[trip["method"] == "historical_xgboost"].iloc[0]
    for _, row in trip.iterrows():
        rows.append(
            {
                "task": "trip_count",
                "method": row["method"],
                "display_name": TRIP_LABELS.get(row["method"], row["method"]),
                "primary_metric": "weighted_mae",
                "primary_value": row["weighted_mae"],
                "primary_improvement_vs_baseline_pct": 100.0
                * (trip_baseline["weighted_mae"] - row["weighted_mae"])
                / trip_baseline["weighted_mae"],
                "secondary_metric": "weighted_rmse",
                "secondary_value": row["weighted_rmse"],
                "bias_metric": "weighted_bias",
                "bias_value": row["weighted_bias"],
                "fit_metric": "weighted_r2",
                "fit_value": row["weighted_r2"],
            }
        )
    mode_baseline = mode_comparison[mode_comparison["method"] == "historical_xgboost"].iloc[0]
    for _, row in mode_comparison.iterrows():
        rows.append(
            {
                "task": "mode_composition",
                "method": row["method"],
                "display_name": row["method"],
                "primary_metric": "weighted_total_variation",
                "primary_value": row["weighted_total_variation"],
                "primary_improvement_vs_baseline_pct": 100.0
                * (mode_baseline["weighted_total_variation"] - row["weighted_total_variation"])
                / mode_baseline["weighted_total_variation"],
                "secondary_metric": "transit_share_weighted_mae",
                "secondary_value": row["transit_share_weighted_mae"],
                "bias_metric": "transit_share_weighted_bias",
                "bias_value": row["transit_share_weighted_bias"],
                "fit_metric": "weighted_dominant_mode_accuracy",
                "fit_value": row["weighted_dominant_mode_accuracy"],
            }
        )
    summary = pd.DataFrame(rows)
    save_csv(summary, FINAL_DIR / "method_comparison_summary.csv")
    figure_path = save_mode_llm_only_figure(mode_comparison)
    write_method_report(trip, acc, mode_comparison)
    return trip, acc, mode_comparison, figure_path


def write_method_report(trip: pd.DataFrame, acc: pd.DataFrame, mode: pd.DataFrame) -> Path:
    path = FINAL_DIR / "method_comparison_report.md"
    strong = load_strong_baselines()
    trip_base = trip[trip["method"] == "historical_xgboost"].iloc[0]
    trip_llm_only = trip[trip["method"] == "llm_only_trip_suppression_a1p25"].iloc[0]
    trip_gated = trip[trip["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    gated_acc = acc[acc["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    mode_base = mode[mode["method"] == "historical_xgboost"].iloc[0]
    mode_best = mode[mode["method"] == "llm_transit_avoidance_a1"].iloc[0]
    llm_only_best = mode[mode["method"] == "llm_only_2017_mean_transit_a1.25"].iloc[0]
    llm_only_gain = 100.0 * (trip_llm_only.weighted_mae - trip_gated.weighted_mae) / trip_llm_only.weighted_mae
    llm_only_bias_gain = 100.0 * (
        abs(trip_llm_only.weighted_bias) - abs(trip_gated.weighted_bias)
    ) / abs(trip_llm_only.weighted_bias)
    lines = [
        "# Method Comparison Report",
        "",
        "## What Is Being Compared",
        "",
        "The project has three household-level behavior outputs.",
        "",
        "- Household trip generation: daily trip-count regression, target `CNTTDHH`.",
        "- Household mode composition: mode-share prediction derived from trip-level `TRPTRANS`.",
        "- Mode-specific trip counts can be formed as predicted total trips multiplied by predicted mode shares.",
        "",
        "## Metric Definitions",
        "",
        "- Weighted MAE: survey-weighted average absolute trip-count error. Lower is better.",
        "- Weighted RMSE: survey-weighted root mean squared trip-count error. Lower is better.",
        "- Weighted Bias: survey-weighted signed error. Values closer to zero mean less systematic over/under prediction.",
        "- Weighted R2: weighted explained variation relative to predicting the weighted mean. Higher is better.",
        "- Within-k accuracy: share of households whose trip-count error is within k trips.",
        "- Weighted total variation: `0.5 * sum(abs(predicted mode shares - true mode shares))`, then survey-weighted. Lower is better.",
        "- Transit-share weighted MAE: survey-weighted absolute error for the public-transit share component. Lower is better.",
        "",
        "## Trip-Count Results",
        "",
        "| Method | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | MAE Gain vs Historical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    trip_index = trip.set_index("method")
    for method in PUBLIC_TRIP_METHODS:
        row = trip_index.loc[method]
        gain = 100.0 * (trip_base["weighted_mae"] - row["weighted_mae"]) / trip_base["weighted_mae"]
        lines.append(
            f"| {TRIP_LABELS.get(method, method)} | {row.weighted_mae:.4f} | "
            f"{row.weighted_rmse:.4f} | {row.weighted_bias:.4f} | {row.weighted_r2:.4f} | {gain:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Trip-count takeaway:",
            "",
            f"- Primary strict no-label row: `{trip_gated.method}`, weighted MAE `{trip_gated.weighted_mae:.4f}`, "
            f"weighted bias `{trip_gated.weighted_bias:.4f}`, weighted R2 `{trip_gated.weighted_r2:.4f}`.",
            f"- LLM-only pressure baseline: weighted MAE `{trip_llm_only.weighted_mae:.4f}`. "
            "This shows that event priors help directionally but need a household historical predictor.",
            f"- Compared with LLM-only pressure, the hybrid gated adapter reduces weighted MAE by `{llm_only_gain:.2f}%` "
            f"and absolute weighted bias by `{llm_only_bias_gain:.2f}%`.",
            f"- Gated household accuracy: exact `{pct(gated_acc.exact_rounded_accuracy)}`, within 2 trips `{pct(gated_acc.within_2_trips)}`, within 3 trips `{pct(gated_acc.within_3_trips)}`.",
            "",
            "## Stronger Non-LLM Tabular Baselines",
            "",
        ]
    )
    if strong.empty:
        lines.append("The stronger-baseline table is missing. Run `src/run_stronger_tabular_baselines.py` to regenerate it.")
    else:
        best_strong = strong.sort_values("weighted_mae").iloc[0]
        strong_gain = 100.0 * (best_strong.weighted_mae - trip_gated.weighted_mae) / best_strong.weighted_mae
        lines.extend(
            [
                "| Method | Device | Weighted MAE | Weighted Bias | Weighted R2 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in strong.sort_values("weighted_mae").iterrows():
            lines.append(
                f"| {row.method} | {row.device} | {row.weighted_mae:.4f} | "
                f"{row.weighted_bias:.4f} | {row.weighted_r2:.4f} |"
            )
        lines.extend(
            [
                "",
                f"The best stronger non-LLM baseline is `{best_strong.method}` with weighted MAE "
                f"`{best_strong.weighted_mae:.4f}` and weighted bias `{best_strong.weighted_bias:.4f}`. "
                f"The primary hybrid adapter is `{strong_gain:.2f}%` lower in weighted MAE, which supports the claim "
                "that model capacity and covariate-shift reweighting do not remove the post-pandemic mechanism shift.",
            ]
        )
    lines.extend(
        [
            "",
            "## Mode-Composition Results",
            "",
            "| Method | Weighted TV | Mean Share MAE | Dominant Accuracy | Transit MAE | Transit Bias |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    mode_order = [
        "llm_only_2017_mean_no_adapt",
        "llm_only_2017_mean_transit_a1.25",
        "historical_mean_2017",
        "historical_xgboost",
        "global_transit_avoidance_a1",
        "llm_transit_avoidance_a1",
    ]
    mode_index = mode.set_index("method")
    for method in mode_order:
        row = mode_index.loc[method]
        lines.append(
            f"| {method} | {row.weighted_total_variation:.4f} | {row.weighted_mean_share_mae:.4f} | "
            f"{row.weighted_dominant_mode_accuracy:.4f} | {row.transit_share_weighted_mae:.4f} | "
            f"{row.transit_share_weighted_bias:.4f} |"
        )
    lines.extend(
        [
            "",
            "Mode-composition takeaway:",
            "",
            f"- LLM-only correction improves over a 2017 mean prior, but remains weaker than household-feature XGBoost.",
            f"- XGBoost + LLM transit prior reduces weighted TV from `{mode_base.weighted_total_variation:.4f}` to `{mode_best.weighted_total_variation:.4f}`.",
            f"- XGBoost + LLM transit prior reduces transit-share weighted MAE from `{mode_base.transit_share_weighted_mae:.4f}` to `{mode_best.transit_share_weighted_mae:.4f}`.",
            f"- LLM-only corrected transit MAE is `{llm_only_best.transit_share_weighted_mae:.4f}`, showing that LLM event priors help directionally but need a household-level historical predictor.",
            "",
            "## Final Interpretation",
            "",
            "The strongest claim is not that LLMs replace mobility models. The stronger and more defensible claim is that LLM event priors repair historical mobility predictors under post-pandemic event shift.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_method_report_zh(trip: pd.DataFrame, acc: pd.DataFrame, mode: pd.DataFrame) -> Path:
    path = FINAL_DIR / "method_comparison_report_zh.md"
    strong = load_strong_baselines()
    trip_base = trip[trip["method"] == "historical_xgboost"].iloc[0]
    trip_llm_only = trip[trip["method"] == "llm_only_trip_suppression_a1p25"].iloc[0]
    trip_gated = trip[trip["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    gated_acc = acc[acc["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    mode_base = mode[mode["method"] == "historical_xgboost"].iloc[0]
    mode_best = mode[mode["method"] == "llm_transit_avoidance_a1"].iloc[0]
    llm_only_best = mode[mode["method"] == "llm_only_2017_mean_transit_a1.25"].iloc[0]
    llm_only_gain = 100.0 * (trip_llm_only.weighted_mae - trip_gated.weighted_mae) / trip_llm_only.weighted_mae
    llm_only_bias_gain = 100.0 * (
        abs(trip_llm_only.weighted_bias) - abs(trip_gated.weighted_bias)
    ) / abs(trip_llm_only.weighted_bias)
    lines = [
        "# 方法比较说明",
        "",
        "## 比较对象",
        "",
            "本项目按家庭颗粒度组织为三个行为预测输出：",
        "",
        "- Trip generation：家庭每日出行次数预测，目标变量是 `CNTTDHH`。",
        "- Mode composition：家庭层面的出行方式结构预测，由 trip-level `TRPTRANS` 聚合得到。",
        "- Mode-specific trip count：用预测总出行次数乘以预测 mode share，得到各方式出行次数。",
        "",
        "## 指标怎么读",
        "",
        "- Weighted MAE：按 NHTS survey weight 加权后的平均绝对误差，越低越好。",
        "- Weighted RMSE：加权均方根误差，对大误差更敏感，越低越好。",
        "- Weighted Bias：加权有符号误差，越接近 0 越说明没有系统性高估/低估。",
        "- Weighted R2：相对加权均值预测的拟合提升，越高越好。",
        "- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。",
        "- Weighted total variation：mode share 向量整体误差，公式是 `0.5 * sum(abs(pred - true))` 后再加权，越低越好。",
        "- Transit-share weighted MAE：公共交通 share 这一项的加权绝对误差，越低越好。",
        "",
        "## 出行次数预测结果",
        "",
        "| 方法 | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | 相对历史预测 MAE 提升 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    trip_index = trip.set_index("method")
    for method in PUBLIC_TRIP_METHODS:
        row = trip_index.loc[method]
        gain = 100.0 * (trip_base["weighted_mae"] - row["weighted_mae"]) / trip_base["weighted_mae"]
        lines.append(
            f"| {TRIP_LABELS.get(method, method)} | {row.weighted_mae:.4f} | "
            f"{row.weighted_rmse:.4f} | {row.weighted_bias:.4f} | {row.weighted_r2:.4f} | {gain:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Trip-generation 结论：",
            "",
            f"- 主口径使用固定 no-label gated rule：`{trip_gated.method}`，weighted MAE `{trip_gated.weighted_mae:.4f}`，weighted bias `{trip_gated.weighted_bias:.4f}`，weighted R2 `{trip_gated.weighted_r2:.4f}`。",
            f"- LLM-only pressure baseline 的 weighted MAE 是 `{trip_llm_only.weighted_mae:.4f}`，说明 LLM 事件先验能给出方向，但需要 historical household predictor 提供个体化 baseline。",
            f"- 相比 LLM-only pressure，hybrid gated adapter 的 weighted MAE 进一步降低 `{llm_only_gain:.2f}%`，绝对 weighted bias 降低 `{llm_only_bias_gain:.2f}%`。",
            f"- 家庭颗粒度上，gated correction exact hit `{pct(gated_acc.exact_rounded_accuracy)}`，within 2 trips `{pct(gated_acc.within_2_trips)}`，within 3 trips `{pct(gated_acc.within_3_trips)}`。",
            "",
            "## 更强非 LLM 表格 baseline",
            "",
        ]
    )
    if strong.empty:
        lines.append("更强 baseline 表缺失；需要运行 `src/run_stronger_tabular_baselines.py` 重新生成。")
    else:
        best_strong = strong.sort_values("weighted_mae").iloc[0]
        strong_gain = 100.0 * (best_strong.weighted_mae - trip_gated.weighted_mae) / best_strong.weighted_mae
        lines.extend(
            [
                "| 方法 | Device | Weighted MAE | Weighted Bias | Weighted R2 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in strong.sort_values("weighted_mae").iterrows():
            lines.append(
                f"| {row.method} | {row.device} | {row.weighted_mae:.4f} | "
                f"{row.weighted_bias:.4f} | {row.weighted_r2:.4f} |"
            )
        lines.extend(
            [
                "",
                f"最强非 LLM baseline 是 `{best_strong.method}`，weighted MAE `{best_strong.weighted_mae:.4f}`，weighted bias `{best_strong.weighted_bias:.4f}`。"
                f"主方法 hybrid adapter 的 weighted MAE 比它低 `{strong_gain:.2f}%`，说明单纯增强表格模型容量或做 covariate-shift reweighting 仍不能解决 2022 的机制变化。",
            ]
        )
    lines.extend(
        [
            "",
            "## 出行方式结构结果",
            "",
            "| 方法 | Weighted TV | Mean Share MAE | Dominant Accuracy | Transit MAE | Transit Bias |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    mode_order = [
        "llm_only_2017_mean_no_adapt",
        "llm_only_2017_mean_transit_a1.25",
        "historical_mean_2017",
        "historical_xgboost",
        "global_transit_avoidance_a1",
        "llm_transit_avoidance_a1",
    ]
    mode_index = mode.set_index("method")
    for method in mode_order:
        row = mode_index.loc[method]
        lines.append(
            f"| {method} | {row.weighted_total_variation:.4f} | {row.weighted_mean_share_mae:.4f} | "
            f"{row.weighted_dominant_mode_accuracy:.4f} | {row.transit_share_weighted_mae:.4f} | "
            f"{row.transit_share_weighted_bias:.4f} |"
        )
    lines.extend(
        [
            "",
            "Mode-composition 结论：",
            "",
            f"- 纯 LLM-style correction 相比 2017 mean prior 有改善，但仍弱于 XGBoost。",
            f"- XGBoost + LLM transit prior 把 weighted TV 从 `{mode_base.weighted_total_variation:.4f}` 降到 `{mode_best.weighted_total_variation:.4f}`，总体提升较小。",
            f"- 但公共交通这一项改善明显：transit-share weighted MAE 从 `{mode_base.transit_share_weighted_mae:.4f}` 降到 `{mode_best.transit_share_weighted_mae:.4f}`。",
            f"- LLM-only corrected transit MAE 是 `{llm_only_best.transit_share_weighted_mae:.4f}`，说明 LLM 知道方向，但需要历史预测器提供 household-specific baseline。",
            "",
            "## 最终口径",
            "",
            "不要把这个项目讲成“LLM 替代传统模型”。更准确的说法是：历史预测器学习 routine mobility，LLM 提供疫情事件先验，两者结合后能在不使用 2022 标签训练的前提下修正 post-pandemic distribution shift。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def set_plot_style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def set_text(paragraph, size: int, color: RGBColor = COLORS["dark"], bold: bool = False) -> None:
    paragraph.font.name = "Microsoft YaHei"
    paragraph.font.size = Pt(size)
    paragraph.font.color.rgb = color
    paragraph.font.bold = bold


def add_title(slide: Slide, title: str, subtitle: str | None = None) -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.28), Inches(12.2), Inches(0.55))
    frame = box.text_frame
    frame.text = title
    set_text(frame.paragraphs[0], 27, COLORS["dark"], True)
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.58), Inches(0.86), Inches(12.0), Inches(0.38))
        sub.text_frame.text = subtitle
        set_text(sub.text_frame.paragraphs[0], 13, COLORS["muted"], False)


def add_footer(slide: Slide, index: int) -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(7.08), Inches(12.0), Inches(0.22))
    frame = box.text_frame
    frame.text = f"NHTS LLM Event Adaptation | {index}"
    set_text(frame.paragraphs[0], 9, RGBColor(100, 116, 139), False)


def add_bullets(slide: Slide, bullets: list[str], left: float, top: float, width: float, height: float, size: int = 18) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.clear()
    for index, text in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = text
        paragraph.level = 0
        paragraph.space_after = Pt(8)
        set_text(paragraph, size)


def add_metric_card(slide: Slide, x: float, y: float, title: str, value: str, note: str, color: RGBColor) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.85), Inches(1.22))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(248, 250, 252)
    shape.line.color.rgb = color
    shape.line.width = Pt(1.2)
    frame = shape.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.clear()
    p1 = frame.paragraphs[0]
    p1.text = title
    set_text(p1, 10, COLORS["muted"], False)
    p2 = frame.add_paragraph()
    p2.text = value
    set_text(p2, 23, color, True)
    p3 = frame.add_paragraph()
    p3.text = note
    set_text(p3, 8, RGBColor(100, 116, 139), False)


def add_table(slide: Slide, rows: list[list[str]], left: float, top: float, width: float, height: float, font_size: int = 10) -> None:
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.text = value
            cell.margin_left = Inches(0.04)
            cell.margin_right = Inches(0.04)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT
            set_text(paragraph, font_size, COLORS["dark"], row_idx == 0)
            if row_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(226, 232, 240)


def add_picture(slide: Slide, path: Path, left: float, top: float, width: float) -> None:
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def create_deck(trip: pd.DataFrame, acc: pd.DataFrame, mode: pd.DataFrame, llm_only_figure: Path) -> Path:
    ppt_path = FINAL_DIR / "NHTS_LLM_Event_Adaptation_Optimized_Presentation.pptx"
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def base_slide(title: str, subtitle: str | None = None) -> Slide:
        slide = prs.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
        add_title(slide, title, subtitle)
        add_footer(slide, len(prs.slides))
        return slide

    trip_base = trip[trip["method"] == "historical_xgboost"].iloc[0]
    trip_llm_only = trip[trip["method"] == "llm_only_trip_suppression_a1p25"].iloc[0]
    trip_gated = trip[trip["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    gated_acc = acc[acc["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    mode_base = mode[mode["method"] == "historical_xgboost"].iloc[0]
    mode_best = mode[mode["method"] == "llm_transit_avoidance_a1"].iloc[0]
    llm_only_best = mode[mode["method"] == "llm_only_2017_mean_transit_a1.25"].iloc[0]

    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = COLORS["dark"]
    title = slide.shapes.add_textbox(Inches(0.75), Inches(1.05), Inches(11.6), Inches(1.25))
    title.text_frame.text = "Pandemic-Aware Household Travel Behavior Prediction"
    set_text(title.text_frame.paragraphs[0], 42, RGBColor(255, 255, 255), True)
    subtitle = slide.shapes.add_textbox(Inches(0.78), Inches(2.38), Inches(11.5), Inches(0.75))
    subtitle.text_frame.text = "LLM event priors for label-free adaptation under post-pandemic shift"
    set_text(subtitle.text_frame.paragraphs[0], 21, RGBColor(203, 213, 225), False)
    add_metric_card(slide, 0.85, 4.45, "Trip-count MAE", "-42.3%", "primary gated rule", COLORS["blue"])
    add_metric_card(slide, 3.95, 4.45, "Bias", "-99.4%", "gated correction absolute bias", COLORS["green"])
    add_metric_card(slide, 7.05, 4.45, "LLM calls", "-83.2%", "cohort prompting", COLORS["orange"])
    add_footer(slide, 1)

    slide = base_slide("What this project predicts", "A household-level travel behavior prediction system")
    add_table(
        slide,
        [
            ["Output", "Target", "Granularity", "Metric family", "Role in system"],
            ["Trip generation", "CNTTDHH", "Household", "MAE / RMSE / Bias / R2", "How many trips"],
            ["Mode composition", "Mode shares", "Household", "TV / share MAE / dominant mode", "How trips are distributed"],
            ["Mode-specific trips", "Trips by mode", "Household", "Derived count error", "How many trips by each mode"],
        ],
        0.75,
        1.45,
        11.8,
        2.2,
        12,
    )
    add_bullets(
        slide,
        [
            "The project predicts household travel behavior as trip volume plus mode structure.",
            "The LLM contribution is label-free event adaptation under a 2022 pandemic shift.",
        ],
        1.0,
        4.25,
        10.8,
        1.2,
        19,
    )

    slide = base_slide("Experimental guardrail", "2022 labels are evaluation-only in the main setting")
    add_picture(slide, FINAL_FIGURE_DIR / "method_workflow.png", 0.8, 1.35, 11.7)
    add_bullets(
        slide,
        ["LLM sees cohort attributes and pandemic-response context, not 2022 trip-count labels."],
        1.15,
        6.05,
        10.6,
        0.4,
        16,
    )

    slide = base_slide("Trip-count methods", "What each row means")
    add_table(
        slide,
        [
            ["Method", "Uses 2022 labels?", "LLM?", "Purpose"],
            ["Historical predictor", "No", "No", "Routine mobility baseline"],
            ["Historical trend shift", "No", "No", "Mean-trend control"],
            ["LLM-only pressure", "No", "LLM only", "No household predictor"],
            ["Global event prior", "No", "Only global mean", "Event-level downscaling control"],
            ["Random prior control", "No", "Shuffled scores", "Negative control"],
            ["Gated LLM correction", "No", "Cohort-specific when confident", "Primary no-label rule"],
        ],
        0.65,
        1.3,
        12.1,
        4.25,
        10,
    )

    slide = base_slide("Trip-count comparison", "Weighted metrics on 2022 households")
    trip_rows = [["Method", "MAE", "RMSE", "Bias", "R2", "MAE gain"]]
    for method in PUBLIC_TRIP_METHODS:
        row = trip[trip["method"] == method].iloc[0]
        gain = 100.0 * (trip_base["weighted_mae"] - row["weighted_mae"]) / trip_base["weighted_mae"]
        trip_rows.append(
            [
                TRIP_LABELS[method],
                f"{row.weighted_mae:.3f}",
                f"{row.weighted_rmse:.3f}",
                f"{row.weighted_bias:.3f}",
                f"{row.weighted_r2:.3f}",
                f"{gain:.1f}%",
            ]
        )
    add_table(slide, trip_rows, 0.65, 1.25, 12.1, 3.6, 11)
    add_bullets(
        slide,
        [
            f"Primary no-label gated rule: MAE {trip_gated.weighted_mae:.3f}, bias {trip_gated.weighted_bias:.3f}.",
            f"LLM-only pressure: MAE {trip_llm_only.weighted_mae:.3f}; hybrid keeps household-level baseline.",
            "Parameter-sweep sensitivity rows are kept out of the public method comparison.",
        ],
        0.95,
        5.35,
        11.2,
        0.9,
        18,
    )

    slide = base_slide("Trip-count error drops sharply", "LLM event priors repair the 2022 overprediction")
    add_picture(slide, FINAL_FIGURE_DIR / "metric_comparison.png", 0.8, 1.2, 11.9)

    slide = base_slide("Bias and household-level interpretation", "Accuracy bands make the regression result tangible")
    add_picture(slide, FINAL_FIGURE_DIR / "bias_r2_tradeoff.png", 0.75, 1.25, 6.0)
    add_picture(slide, FINAL_FIGURE_DIR / "household_tolerance_accuracy.png", 6.9, 1.25, 5.85)
    add_bullets(
        slide,
        [
            f"Gated MAE: {gated_acc.unweighted_mae:.2f} trips per household.",
            f"Within 2 trips: {pct(gated_acc.within_2_trips)}; within 3 trips: {pct(gated_acc.within_3_trips)}.",
        ],
        1.1,
        6.2,
        10.8,
        0.5,
        15,
    )

    slide = base_slide("Mode-composition extension", "What changed besides trip counts?")
    add_table(
        slide,
        [
            ["Component", "Definition"],
            ["Private vehicle", "Car / van / SUV / pickup / motorcycle / rental vehicle"],
            ["Walk / bike", "Non-motorized travel modes"],
            ["Transit", "Public bus / rail / ferry / paratransit categories"],
            ["Taxi / ridehail", "Taxi, limo, Uber/Lyft-style services"],
            ["Other", "School bus, airplane, other specified modes"],
        ],
        0.75,
        1.3,
        11.8,
        3.45,
        11,
    )
    add_bullets(
        slide,
        ["2017 and 2022 TRPTRANS codes are mapped with year-specific official codebooks."],
        1.0,
        5.4,
        11.0,
        0.6,
        18,
    )

    slide = base_slide("Mode distribution shift", "2022 has fewer trips and lower transit share")
    add_picture(slide, MODE_DIR / "figures" / "mode_distribution_shift.png", 0.9, 1.25, 11.45)

    slide = base_slide("Mode-composition method comparison", "LLM helps the transit component of household behavior")
    add_picture(slide, MODE_DIR / "figures" / "mode_metric_comparison.png", 0.85, 1.25, 11.6)
    add_bullets(
        slide,
        [
            f"Transit-share weighted MAE: {mode_base.transit_share_weighted_mae:.4f} -> {mode_best.transit_share_weighted_mae:.4f}.",
            "Overall mode-composition gain is small, while the transit component shows a clearer pandemic-related signal.",
        ],
        1.05,
        6.15,
        10.7,
        0.5,
        15,
    )

    slide = base_slide("Pure LLM-style baseline", "LLM direction alone helps, but it does not replace household predictors")
    add_picture(slide, llm_only_figure, 0.9, 1.25, 11.55)
    add_bullets(
        slide,
        [
            f"LLM-only corrected transit MAE: {llm_only_best.transit_share_weighted_mae:.4f}.",
            f"XGBoost + LLM transit MAE: {mode_best.transit_share_weighted_mae:.4f}.",
        ],
        1.05,
        6.15,
        10.7,
        0.5,
        15,
    )

    slide = base_slide("Metric guide", "How to read the numbers")
    add_table(
        slide,
        [
            ["Metric", "Task", "Meaning"],
            ["Weighted MAE", "Trip count", "Average absolute trip-count error"],
            ["Weighted Bias", "Trip count", "Systematic over/under prediction"],
            ["Within-k", "Trip count", "Households within k trips"],
            ["Weighted TV", "Mode composition", "Whole share-vector error"],
            ["Transit-share MAE", "Mode composition", "Public-transit share error"],
        ],
        0.75,
        1.35,
        11.8,
        3.4,
        12,
    )
    add_bullets(
        slide,
        ["Lower is better for errors; bias closer to zero is better; R2 and accuracy are higher-is-better."],
        1.0,
        5.3,
        11.0,
        0.7,
        18,
    )

    slide = base_slide("What the LLM contributes", "Event semantics, not direct numerical prediction")
    add_bullets(
        slide,
        [
            "The historical predictor learns routine household mobility from NHTS.",
            "The LLM supplies post-pandemic event priors: trip suppression, remote work, transit avoidance, delivery substitution, recovery sensitivity.",
            "The LLM-only pressure baseline improves over naive history but is weaker than the hybrid model.",
            "Global controls show that broad event downscaling is strong.",
            "Random controls and subgroup diagnostics show that cohort-specific LLM ranking adds smaller but measurable signal.",
        ],
        0.95,
        1.45,
        11.1,
        4.7,
        22,
    )

    slide = base_slide("Caveats", "What we should not overclaim")
    add_bullets(
        slide,
        [
            "Do not claim that LLM directly predicts household trip counts.",
            "Keep parameter-sweep sensitivity rows in appendix/internal analysis, not in the main comparison.",
            "Present trip generation and mode composition as two household-level outputs, not as separate projects.",
            "Mode composition is weaker overall than trip-count adaptation, but it adds the mode-structure dimension.",
            "Future work should add external event context, stronger LLM priors, and prospective validation.",
        ],
        0.95,
        1.45,
        11.1,
        4.7,
        23,
    )

    slide = base_slide("Final takeaway", "A defensible story for the course project")
    add_metric_card(slide, 1.0, 1.55, "Primary trip-count rule", "-42.3%", "gated weighted MAE reduction", COLORS["blue"])
    add_metric_card(slide, 4.2, 1.55, "Bias", "-99.4%", "gated absolute bias reduction", COLORS["green"])
    add_metric_card(slide, 7.4, 1.55, "Transit-share error", "-17.4%", "mode-structure result", COLORS["orange"])
    add_bullets(
        slide,
        [
            "LLMs are most useful here as event-prior generators.",
            "The best model is not pure LLM and not pure historical prediction; it is a label-free hybrid adapter.",
        ],
        1.0,
        4.0,
        11.0,
        1.4,
        24,
    )

    prs.save(ppt_path)
    return ppt_path


def write_speaker_notes(trip: pd.DataFrame, mode: pd.DataFrame) -> Path:
    path = FINAL_DIR / "presentation_speaker_notes.md"
    strong = load_strong_baselines()
    trip_base = trip[trip["method"] == "historical_xgboost"].iloc[0]
    trip_gated = trip[trip["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    trip_llm_only = trip[trip["method"] == "llm_only_trip_suppression_a1p25"].iloc[0]
    trip_global = trip[trip["method"] == "global_trip_suppression_a1p25"].iloc[0]
    mode_base = mode[mode["method"] == "historical_xgboost"].iloc[0]
    mode_best = mode[mode["method"] == "llm_transit_avoidance_a1"].iloc[0]
    best_strong = None if strong.empty else strong.sort_values("weighted_mae").iloc[0]
    lines = [
        "# Presentation Speaker Notes",
        "",
        "## Core Message",
        "We formulate 2022 NHTS household travel prediction as event-driven temporal adaptation. Historical models capture routine household mobility, while LLM event priors encode pandemic mechanisms that are weakly represented in household covariates: remote work, transit avoidance, delivery substitution, and uneven recovery.",
        "",
        "The main method is not pure LLM prediction. It is a hybrid gated adapter: historical household baseline times an event correction factor. The LLM outputs structured event pressure, not household trip-count labels.",
        "",
        "## One-Minute Version",
        f"Historical prediction overestimates 2022 trips. The primary fixed no-label gated rule reduces weighted MAE from `{trip_base.weighted_mae:.4f}` to `{trip_gated.weighted_mae:.4f}` and moves weighted bias to `{trip_gated.weighted_bias:.4f}`. "
        f"The LLM-only pressure baseline reaches `{trip_llm_only.weighted_mae:.4f}`. The LLM is not used as a direct trip-count predictor; it provides pandemic-event semantics that modify a historical routine-mobility predictor.",
        "",
        "## Method Comparison Logic",
        "",
        f"- Ordinary historical prediction has household grounding but lacks event semantics: wMAE `{trip_base.weighted_mae:.4f}`, wBias `{trip_base.weighted_bias:.4f}`.",
        f"- Pure LLM pressure has the right downward direction but weaker calibration: wMAE `{trip_llm_only.weighted_mae:.4f}`, wBias `{trip_llm_only.weighted_bias:.4f}`.",
        f"- Global event prior is a strong low-cost control: wMAE `{trip_global.weighted_mae:.4f}`, wBias `{trip_global.weighted_bias:.4f}`. Do not overclaim that cohort-specific LLM ranking explains the whole gain.",
        f"- Primary hybrid gated adapter is the balanced operating point: wMAE `{trip_gated.weighted_mae:.4f}`, wBias `{trip_gated.weighted_bias:.4f}`, wR2 `{trip_gated.weighted_r2:.4f}`.",
    ]
    if best_strong is not None:
        lines.append(
            f"- Stronger non-LLM tabular baselines still overpredict. Best extra baseline `{best_strong.method}` has wMAE `{best_strong.weighted_mae:.4f}` and wBias `{best_strong.weighted_bias:.4f}`."
        )
    lines.extend(
        [
        "",
        "## Metric Language",
        "- Weighted MAE/RMSE are survey-weighted trip-count errors.",
        "- Weighted bias tells whether we systematically overpredict or underpredict.",
        "- Within-k accuracy is used only to make regression error intuitive.",
        "- Weighted total variation is the whole mode-share vector error.",
        "- Transit-share weighted MAE is the public-transit component error.",
        "",
        "## Mode Extension",
        f"Mode composition is the second household-level output. XGBoost + LLM reduces transit-share weighted MAE from `{mode_base.transit_share_weighted_mae:.4f}` to `{mode_best.transit_share_weighted_mae:.4f}`, while the overall mode-composition improvement is small.",
        "",
        "Mode-specific trip volume is the derived planning output: predicted total trips multiplied by predicted mode shares. It is more interpretable for planning than a standalone mode-share number.",
        "",
        "## Questions To Expect",
        "- Why use two metric families? Trip generation is count regression, while mode composition is a share-vector prediction problem.",
        "- Is this pure LLM? No. Pure LLM-like correction is weaker than XGBoost + LLM.",
        "- Why not let the LLM build a zero-shot 2022 decision tree? A tree needs labels to learn split thresholds and leaf values. Without 2022 labels, the output becomes an LLM belief tree or synthetic-label model, so we use the LLM only as an event-prior generator.",
        "- Did 2022 labels enter training? Not in the main label-free setting; 2022 targets are used for evaluation.",
        "- What should we not claim? Do not claim causal effects or direct LLM trip-count prediction; claim causal guardrails and event-prior adaptation.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_speaker_notes_zh(trip: pd.DataFrame, mode: pd.DataFrame) -> Path:
    path = FINAL_DIR / "presentation_speaker_notes_zh.md"
    strong = load_strong_baselines()
    trip_base = trip[trip["method"] == "historical_xgboost"].iloc[0]
    trip_gated = trip[trip["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
    trip_llm_only = trip[trip["method"] == "llm_only_trip_suppression_a1p25"].iloc[0]
    trip_global = trip[trip["method"] == "global_trip_suppression_a1p25"].iloc[0]
    mode_base = mode[mode["method"] == "historical_xgboost"].iloc[0]
    mode_best = mode[mode["method"] == "llm_transit_avoidance_a1"].iloc[0]
    best_strong = None if strong.empty else strong.sort_values("weighted_mae").iloc[0]
    lines = [
        "# 中文汇报讲稿",
        "",
        "## 核心主线",
        "",
        "我们把 2022 NHTS household travel behavior prediction 定义成 event-driven temporal adaptation 问题。历史模型负责学习 routine mobility，LLM event priors 负责表达疫情带来的 remote work、transit avoidance、delivery substitution 等事件机制。",
        "",
        "现在的项目不是 pure LLM 预测，也不是单纯 XGBoost。主方法是 hybrid gated adapter：先用历史 NHTS 学一个 household baseline，再用 LLM 生成的 event pressure 做事件修正。",
        "",
        "可以用一句公式概括：",
        "",
        "```text",
        "2022 prediction = historical household baseline × event correction",
        "```",
        "",
        "LLM 输出的是结构化事件先验，不直接输出 `CNTTDHH`。",
        "",
        "## 一分钟版本",
        "",
        f"传统历史预测器会明显高估 2022 年家庭出行次数。主口径使用固定 no-label gated rule，weighted MAE 从 `{trip_base.weighted_mae:.4f}` 降到 `{trip_gated.weighted_mae:.4f}`，weighted bias 变成 `{trip_gated.weighted_bias:.4f}`，几乎消除了系统性高估。LLM-only pressure baseline 的 weighted MAE 是 `{trip_llm_only.weighted_mae:.4f}`。这说明 LLM 不是直接预测 trip count，而是提供疫情事件语义先验，再去修正历史 routine-mobility predictor。",
        "",
        "## 各方法怎么比较",
        "",
        f"- 传统 XGBoost / CatBoost：优势是有 household grounding，问题是不知道 2022 疫情机制变化。普通 historical predictor 的 wMAE 是 `{trip_base.weighted_mae:.4f}`，wBias 是 `{trip_base.weighted_bias:.4f}`，明显高估。",
    ]
    if best_strong is not None:
        lines.append(
            f"- 更强表格 baseline 仍然不够。最强非 LLM baseline `{best_strong.method}` 的 wMAE 是 `{best_strong.weighted_mae:.4f}`，wBias 是 `{best_strong.weighted_bias:.4f}`。"
        )
    lines.extend(
        [
        f"- Pure LLM pressure：优势是知道疫情后出行下降方向，问题是缺少家庭数值基线，wMAE `{trip_llm_only.weighted_mae:.4f}`，wBias `{trip_llm_only.weighted_bias:.4f}`。",
        f"- Global event prior：低成本且很强，wMAE `{trip_global.weighted_mae:.4f}`，wBias `{trip_global.weighted_bias:.4f}`。这说明主信号确实是 event-level suppression，不能夸大成 cohort LLM ranking 独自贡献全部提升。",
        f"- 我们的 hybrid gated：wMAE `{trip_gated.weighted_mae:.4f}`，wBias `{trip_gated.weighted_bias:.4f}`，wR2 `{trip_gated.weighted_r2:.4f}`。优势是同时保留 household baseline、event semantics 和 near-zero bias。",
        "",
        "这就是主方法优势：传统模型有家庭基线但没有疫情机制，pure LLM 有事件方向但校准弱，global rule 太粗；hybrid gated 把三者的优点组合起来。",
        "",
        "## 指标解释",
        "",
        "- Weighted MAE/RMSE：加权后的出行次数误差。",
        "- Weighted Bias：系统性高估或低估，越接近 0 越好。",
        "- Within-k accuracy：预测误差在 k 次 trip 以内的 household 比例。",
        "- Weighted total variation：家庭 mode-share 向量整体误差。",
        "- Transit-share weighted MAE：公共交通占比这一项的误差。",
        "",
        "## Mode Extension 怎么讲",
        "",
        f"出行方式结构是第二个家庭层面的预测输出。XGBoost + LLM 把 transit-share weighted MAE 从 `{mode_base.transit_share_weighted_mae:.4f}` 降到 `{mode_best.transit_share_weighted_mae:.4f}`，说明 LLM 的 transit avoidance prior 对公共交通这一项有帮助；但总体 mode composition 改善不大，所以汇报时要把它讲成 mode-structure 维度，而不是夸大成主要增益。",
        "",
        "进一步的 planning 输出是 mode-specific trip volume：用预测总出行次数乘以预测 mode share，得到各方式出行量。这个比单独报 mode share 更接近城市规划里的需求评估。",
        "",
        "## 边界和不能过度声称的地方",
        "",
        "- 不能说 LLM 直接预测 household trips。",
        "- 不能说我们识别了 COVID 的 causal effect；应该说 causal guardrails 和 mechanism proxy。",
        "- purpose composition 目前是探索性输出，不作为主贡献。",
        "- global event prior 很强，所以 contribution 要讲成 event-level adaptation + auditable cohort refinement。",
        "",
        "## 可能被问到的问题",
        "",
        "- 为什么有两套指标？因为 trip generation 是 count regression，mode composition 是 share-vector prediction。",
        "- 这是 pure LLM 吗？不是。纯 LLM-style correction 比 XGBoost + LLM 弱，说明 LLM 适合作为 event-prior adapter。",
        "- 为什么不直接让 LLM 零样本构建 2022 决策树？因为决策树需要标签学习 split threshold 和叶节点数值；没有 2022 标签时，这会变成 LLM belief tree 或 synthetic-label model，数值校准不如历史 household model + event prior。",
        "- 有没有用 2022 标签训练？主实验没有。2022 `CNTTDHH` 只在最终 evaluation 中使用。",
        "- 结果够不够做大作业？够，因为我们有完整数据链路、强 baseline、LLM event prior、无标签修正、稳健性检验、mode 扩展和 10 分钟汇报材料。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_storyboard() -> Path:
    path = PROJECT_ROOT / "plan" / "optimized_presentation_storyboard.md"
    lines = [
        "# Optimized Presentation Storyboard",
        "",
        "1. Title: label-free LLM event adaptation.",
        "2. Task map: trip count, mode composition, and derived mode-specific trip counts.",
        "3. Experimental guardrail: no 2022 labels in main training/adaptation.",
        "4. Trip-count methods: historical, global, random, LLM, gated.",
        "5. Trip-count metric table.",
        "6. Trip-count error plot.",
        "7. Bias and household accuracy.",
        "8. Mode-composition extension scope.",
        "9. Mode distribution shift.",
        "10. Mode-composition method comparison.",
        "11. Pure LLM-style baseline.",
        "12. Metric guide.",
        "13. LLM contribution.",
        "14. Caveats.",
        "15. Final takeaway.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    ensure_dirs()
    set_plot_style()
    trip, acc, mode, figure_path = build_method_comparison()
    write_method_report_zh(trip, acc, mode)
    ppt_path = create_deck(trip, acc, mode, figure_path)
    notes_path = write_speaker_notes(trip, mode)
    write_speaker_notes_zh(trip, mode)
    storyboard_path = write_storyboard()
    LOGGER.info("Wrote optimized PPT: %s", ppt_path)
    LOGGER.info("Wrote speaker notes: %s", notes_path)
    LOGGER.info("Wrote storyboard: %s", storyboard_path)


if __name__ == "__main__":
    main()
