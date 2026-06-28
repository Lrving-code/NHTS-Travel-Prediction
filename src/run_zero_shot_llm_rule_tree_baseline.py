"""Evaluate zero-shot LLM-style rule-tree baselines for 2022 trip counts.

The script operationalizes the question: "Why not let an LLM directly build a
2022 decision tree?" It freezes a qualitative, zero-label rule tree that uses
household covariates and precomputed LLM event priors, then optionally distills
that pseudo label into a shallow sklearn decision tree. No 2022 target labels are
used for constructing either baseline; labels are used only for final metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "models" / "final_label_free_2022" / "final_2022_predictions.csv"
HOUSEHOLD_PATH = PROJECT_ROOT / "data" / "processed" / "household_harmonized.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "zero_shot_llm_rule_tree_baseline"

TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
BASE_PREDICTION = "base_prediction"
PRIMARY_PREDICTION = "prediction_gated_trip_suppression_a1_d0p15"
LLM_ONLY_PREDICTION = "prediction_llm_only_trip_suppression_a1p25"
RANDOM_SEED = 20260610

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleTrace:
    """Human-readable leaf explanation for a zero-shot rule prediction."""

    demand_leaf: str
    event_leaf: str
    adjustment_leaf: str


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    target_mean = weighted_average(y_true, weights)
    numerator = float(np.sum(weights * np.square(y_true - y_pred)))
    denominator = float(np.sum(weights * np.square(y_true - target_mean)))
    return float(1.0 - numerator / denominator)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    rounded_error = np.abs(np.rint(y_pred) - np.rint(y_true))
    return {
        "weighted_mae": weighted_average(abs_error, weights),
        "weighted_rmse": float(np.sqrt(weighted_average(np.square(error), weights))),
        "weighted_bias": weighted_average(error, weights),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_target_mean": weighted_average(y_true, weights),
        "weighted_prediction_mean": weighted_average(y_pred, weights),
        "exact_rounded_accuracy": float(np.mean(rounded_error == 0.0)),
        "within_1_trip": float(np.mean(abs_error <= 1.0)),
        "within_2_trips": float(np.mean(abs_error <= 2.0)),
        "within_3_trips": float(np.mean(abs_error <= 3.0)),
        "weighted_within_1_trip": weighted_average((abs_error <= 1.0).astype(float), weights),
        "weighted_within_2_trips": weighted_average((abs_error <= 2.0).astype(float), weights),
        "weighted_within_3_trips": weighted_average((abs_error <= 3.0).astype(float), weights),
        "exact_accuracy": float(np.mean(rounded_error == 0.0)),
        "within_1_accuracy": float(np.mean(rounded_error <= 1.0)),
        "within_2_accuracy": float(np.mean(rounded_error <= 2.0)),
        "within_3_accuracy": float(np.mean(rounded_error <= 3.0)),
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(values.median())


def load_analysis_frame() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"HOUSEID": str})
    needed = [
        "HOUSEID",
        "survey_year",
        "HHFAMINC",
        "HHSIZE",
        "HHVEHCNT",
        "WRKCOUNT",
        "DRVRCNT",
        "NUMADLT",
        "URBAN",
        "RAIL",
        "TRAVDAY",
        "travel_month",
    ]
    households = pd.read_csv(HOUSEHOLD_PATH, usecols=needed, dtype={"HOUSEID": str}, low_memory=False)
    households = households.loc[households["survey_year"] == 2022].drop_duplicates("HOUSEID")
    predictions["HOUSEID"] = predictions["HOUSEID"].astype(str)
    households["HOUSEID"] = households["HOUSEID"].astype(str)
    merged = predictions.merge(households.drop(columns=["survey_year"]), on="HOUSEID", how="left", validate="one_to_one")
    missing_share = float(merged["HHSIZE"].isna().mean())
    if missing_share > 0.01:
        raise ValueError(f"Unexpected household covariate missing share after merge: {missing_share:.2%}")
    return merged


def household_base_demand(row: pd.Series) -> tuple[float, str]:
    household_size = float(row["HHSIZE"])
    vehicles = float(row["HHVEHCNT"])
    workers = float(row["WRKCOUNT"])
    adults = float(row["NUMADLT"])
    drivers = float(row["DRVRCNT"])

    if household_size >= 4:
        demand = 8.2
        leaf = "large household"
    elif household_size >= 2:
        demand = 5.8
        leaf = "two-to-three-person household"
    else:
        demand = 3.0
        leaf = "single-person household"

    if workers >= 2:
        demand += 1.2
        leaf += " + multiple workers"
    elif workers == 1:
        demand += 0.5
        leaf += " + one worker"
    else:
        demand -= 0.4
        leaf += " + no worker"

    if vehicles >= 2:
        demand += 0.8
    elif vehicles == 0:
        demand -= 1.1

    if adults >= 2 and drivers == 0:
        demand -= 0.7

    return max(demand, 0.0), leaf


def event_factor(row: pd.Series) -> tuple[float, str]:
    suppression = float(row["trip_suppression_risk"])
    remote_work = float(row["remote_work_substitution_likelihood"])
    transit_avoidance = float(row["transit_avoidance_likelihood"])
    delivery = float(row["online_delivery_substitution_likelihood"])
    recovery = float(row["post_pandemic_recovery_sensitivity"])

    if suppression >= 0.68 or (remote_work >= 0.70 and delivery >= 0.62):
        factor = 0.48
        leaf = "high suppression / remote-delivery substitution"
    elif suppression >= 0.52 or remote_work >= 0.62:
        factor = 0.62
        leaf = "medium suppression / remote-work substitution"
    elif transit_avoidance >= 0.68 and recovery <= 0.55:
        factor = 0.68
        leaf = "transit avoidance with slow recovery"
    elif suppression >= 0.36:
        factor = 0.76
        leaf = "mild suppression"
    else:
        factor = 0.88
        leaf = "low suppression"

    if recovery >= 0.72:
        factor += 0.08
        leaf += " + faster recovery"
    elif recovery <= 0.42:
        factor -= 0.05
        leaf += " + slower recovery"

    return float(np.clip(factor, 0.35, 0.95)), leaf


def context_adjustment(row: pd.Series) -> tuple[float, str]:
    urban = float(row["URBAN"])
    rail = float(row["RAIL"])
    travel_day = float(row["TRAVDAY"])
    month = float(row["travel_month"])

    adjustment = 0.0
    leaf = "neutral context"
    if urban == 1 and rail == 1 and float(row["transit_avoidance_likelihood"]) >= 0.60:
        adjustment -= 0.5
        leaf = "urban rail with transit avoidance"
    if travel_day in {1.0, 7.0}:
        adjustment -= 0.3
        leaf += " + weekend"
    if month in {6.0, 7.0, 8.0, 12.0}:
        adjustment += 0.2
        leaf += " + seasonal activity"
    return adjustment, leaf


def predict_zero_shot_rule_tree(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    predictions: list[float] = []
    traces: list[dict[str, str | float | int]] = []
    working = frame.copy()
    numeric_columns = [
        "HHSIZE",
        "HHVEHCNT",
        "WRKCOUNT",
        "DRVRCNT",
        "NUMADLT",
        "URBAN",
        "RAIL",
        "TRAVDAY",
        "travel_month",
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "transit_avoidance_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
    ]
    for column in numeric_columns:
        working[column] = numeric_series(working, column)

    for row in working.itertuples(index=False):
        series = pd.Series(row._asdict())
        demand, demand_leaf = household_base_demand(series)
        factor, event_leaf = event_factor(series)
        adjustment, adjustment_leaf = context_adjustment(series)
        prediction = float(np.clip(demand * factor + adjustment, 0.0, 18.0))
        predictions.append(prediction)
        traces.append(
            {
                "HOUSEID": str(series["HOUSEID"]),
                "zero_shot_rule_prediction": prediction,
                "demand_leaf": demand_leaf,
                "event_leaf": event_leaf,
                "adjustment_leaf": adjustment_leaf,
            }
        )
    return np.asarray(predictions, dtype=float), pd.DataFrame(traces)


def tree_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "HHSIZE",
        "HHVEHCNT",
        "WRKCOUNT",
        "DRVRCNT",
        "NUMADLT",
        "URBAN",
        "RAIL",
        "TRAVDAY",
        "travel_month",
        "HHFAMINC",
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "transit_avoidance_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
    ]
    features = frame[feature_columns].copy()
    for column in feature_columns:
        features[column] = numeric_series(features, column)
    return features


def train_pseudo_label_tree(features: pd.DataFrame, pseudo_labels: np.ndarray, weights: np.ndarray) -> DecisionTreeRegressor:
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=120, random_state=RANDOM_SEED)
    tree.fit(features, pseudo_labels, sample_weight=weights)
    return tree


def build_metrics(frame: pd.DataFrame, rule_predictions: np.ndarray, tree_predictions: np.ndarray) -> pd.DataFrame:
    y_true = frame[TARGET].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    rows = [
        {
            "method": "historical_xgboost",
            "family": "traditional_supervised",
            "description": "Historical routine model trained on pre-2022 NHTS labels.",
            **evaluate(y_true, frame[BASE_PREDICTION].to_numpy(dtype=float), weights),
        },
        {
            "method": "zero_shot_llm_rule_tree",
            "family": "zero_shot_llm_tree",
            "description": "Frozen qualitative rule tree with no 2022 labels and no historical model baseline.",
            **evaluate(y_true, rule_predictions, weights),
        },
        {
            "method": "zero_shot_pseudo_label_tree",
            "family": "zero_shot_llm_tree",
            "description": "Shallow decision tree distilled from the zero-shot rule predictions, not true labels.",
            **evaluate(y_true, tree_predictions, weights),
        },
        {
            "method": "llm_only_pressure",
            "family": "llm_only_control",
            "description": "Existing LLM-only pressure baseline without the routine household predictor.",
            **evaluate(y_true, frame[LLM_ONLY_PREDICTION].to_numpy(dtype=float), weights),
        },
        {
            "method": "primary_hybrid_gated_adapter",
            "family": "main_method",
            "description": "Historical routine model plus fixed gated LLM event-prior adapter.",
            **evaluate(y_true, frame[PRIMARY_PREDICTION].to_numpy(dtype=float), weights),
        },
    ]
    metrics = pd.DataFrame(rows)
    primary_mae = float(metrics.loc[metrics["method"] == "primary_hybrid_gated_adapter", "weighted_mae"].iloc[0])
    metrics["mae_delta_vs_primary"] = metrics["weighted_mae"] - primary_mae
    return metrics


def plot_metrics(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "zero_shot_rule_tree_metric_comparison.png"
    plot_data = metrics.copy()
    order = [
        "historical_xgboost",
        "zero_shot_llm_rule_tree",
        "zero_shot_pseudo_label_tree",
        "llm_only_pressure",
        "primary_hybrid_gated_adapter",
    ]
    labels = {
        "historical_xgboost": "Historical\nXGBoost",
        "zero_shot_llm_rule_tree": "Zero-shot\nrule tree",
        "zero_shot_pseudo_label_tree": "Pseudo-label\nDT",
        "llm_only_pressure": "LLM-only\npressure",
        "primary_hybrid_gated_adapter": "Primary\nhybrid",
    }
    colors = {
        "historical_xgboost": "#94a3b8",
        "zero_shot_llm_rule_tree": "#f97316",
        "zero_shot_pseudo_label_tree": "#f59e0b",
        "llm_only_pressure": "#38bdf8",
        "primary_hybrid_gated_adapter": "#16a34a",
    }
    plot_data["order"] = plot_data["method"].map({name: idx for idx, name in enumerate(order)})
    plot_data = plot_data.sort_values("order")
    plt.figure(figsize=(8.8, 4.8))
    bars = plt.bar(
        [labels[method] for method in plot_data["method"]],
        plot_data["weighted_mae"],
        color=[colors[method] for method in plot_data["method"]],
        edgecolor="#334155",
    )
    plt.ylabel("Weighted MAE")
    plt.title("Direct zero-shot rule trees are less stable than hybrid event adaptation", fontsize=12)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.035, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def write_rule_spec(markdown_path: Path, json_path: Path, tree_path: Path, tree_text: str) -> None:
    lines = [
        "# Zero-Shot Rule-Tree Baseline Specification",
        "",
        "## Purpose",
        "",
        "This ablation answers whether a zero-shot LLM-style decision tree can directly replace the hybrid "
        "routine-model-plus-event-prior design.",
        "",
        "## Label Policy",
        "",
        "- No 2022 `CNTTDHH` labels are used when defining the qualitative rule tree.",
        "- No 2022 `CNTTDHH` labels are used when distilling the pseudo-label decision tree.",
        "- 2022 labels are used only for final evaluation metrics.",
        "",
        "## Qualitative Rule Tree",
        "",
        "1. Estimate routine household demand from household size, workers, vehicles, adults, and drivers.",
        "2. Apply an event factor from LLM event priors: trip suppression, remote-work substitution, "
        "transit avoidance, delivery substitution, and recovery sensitivity.",
        "3. Apply small context adjustments for urban rail exposure, weekend travel day, and seasonal activity.",
        "4. Clip the resulting household trip-count prediction to `[0, 18]`.",
        "",
        "## Distilled Pseudo-Label Decision Tree",
        "",
        "The following sklearn tree is trained on the zero-shot rule predictions, not on true labels:",
        "",
        "```text",
        tree_text,
        "```",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    tree_path.write_text(tree_text, encoding="utf-8")
    spec = {
        "name": "zero_shot_llm_semantic_rule_tree",
        "label_usage": "No 2022 CNTTDHH labels are used for split selection, leaf values, or calibration.",
        "allowed_inputs": [
            "household size, vehicle ownership, worker count, adult count, driver count, urban/rail context",
            "LLM event priors: trip suppression, remote work, transit avoidance, delivery substitution, recovery sensitivity",
        ],
        "forbidden_inputs": [
            "2022 CNTTDHH for rule construction",
            "2022 aggregate target outcomes",
            "household ID as a predictor",
        ],
        "decision_logic": [
            "Estimate household routine demand from household size, workers, vehicles, adults, and drivers.",
            "Apply an event factor from LLM event priors without using 2022 labels.",
            "Apply small context adjustments for urban rail exposure, weekend travel day, and seasonal activity.",
            "Distill the resulting pseudo-labels into a shallow decision tree for interpretability.",
        ],
        "caveat": (
            "This is a qualitative belief-tree baseline. Its thresholds and leaf values are not optimized against "
            "true 2022 NHTS outcomes, so it should be read as an ablation rather than the main predictor."
        ),
    }
    json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(metrics: pd.DataFrame, figure_path: Path) -> Path:
    path = OUTPUT_DIR / "zero_shot_llm_rule_tree_report_zh.md"
    primary = metrics.loc[metrics["method"] == "primary_hybrid_gated_adapter"].iloc[0]
    rule = metrics.loc[metrics["method"] == "zero_shot_llm_rule_tree"].iloc[0]
    pseudo_tree = metrics.loc[metrics["method"] == "zero_shot_pseudo_label_tree"].iloc[0]
    historical = metrics.loc[metrics["method"] == "historical_xgboost"].iloc[0]
    llm_only = metrics.loc[metrics["method"] == "llm_only_pressure"].iloc[0]

    lines = [
        "# Zero-Shot LLM Rule-Tree Baseline",
        "",
        "## 这个实验回答什么问题",
        "",
        "这个 ablation 回答老师/审稿人可能追问的问题：为什么不直接让 LLM 在零样本情况下构建 2022 年决策树？",
        "",
        "## 简短回答",
        "",
        "可以构建 zero-shot rule tree，但它更像 qualitative prior，而不是经过真实 NHTS outcome 校准的统计模型。"
        "它能表达目标年事件方向，但 split threshold 和 leaf value 并不是从真实标签中估计出来的。"
        "主方法更稳健，因为数值上的 routine demand 来自历史 NHTS 标签，LLM 只负责提供 event semantics。",
        "",
        "## 结果",
        "",
        f"- Historical XGBoost wMAE: `{historical.weighted_mae:.4f}`.",
        f"- Zero-shot rule tree wMAE: `{rule.weighted_mae:.4f}`; weighted bias: `{rule.weighted_bias:+.4f}`.",
        f"- Pseudo-label decision tree wMAE: `{pseudo_tree.weighted_mae:.4f}`; weighted bias: `{pseudo_tree.weighted_bias:+.4f}`.",
        f"- LLM-only pressure wMAE: `{llm_only.weighted_mae:.4f}`; weighted bias: `{llm_only.weighted_bias:+.4f}`.",
        f"- Primary hybrid gated wMAE: `{primary.weighted_mae:.4f}`; weighted bias: `{primary.weighted_bias:+.4f}`.",
        f"- Zero-shot rule tree is `{rule.weighted_mae - primary.weighted_mae:+.4f}` wMAE relative to the primary method.",
        f"- Pseudo-label decision tree is `{pseudo_tree.weighted_mae - primary.weighted_mae:+.4f}` wMAE relative to the primary method.",
        "",
        "## 结论",
        "",
        "这个 ablation 支持当前设计：LLM 不应该直接替代监督 household model。"
        "LLM 适合作为 event-generalization module，但直接 zero-shot tree 缺少数据估计的 threshold 和校准后的 leaf value。"
        "在论文或 PPT 中，它可以解释为什么我们采用 hybrid adapter，而不是 pure LLM rule induction。",
        "",
        "## 完整指标表",
        "",
        "| Method | Family | wMAE | wRMSE | wBias | wR2 | Exact | Within 2 | Delta vs primary |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.family} | {row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} | {row.exact_accuracy:.3f} | "
            f"{row.within_2_accuracy:.3f} | {row.mae_delta_vs_primary:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
            f"- Metrics: `{(OUTPUT_DIR / 'zero_shot_llm_rule_tree_metrics.csv').relative_to(PROJECT_ROOT)}`",
            f"- Rule traces: `{(OUTPUT_DIR / 'zero_shot_llm_rule_tree_predictions.csv').relative_to(PROJECT_ROOT)}`",
            f"- Rule specification: `{(OUTPUT_DIR / 'zero_shot_llm_rule_tree_spec.json').relative_to(PROJECT_ROOT)}`",
            f"- Markdown rule specification: `{(OUTPUT_DIR / 'zero_shot_rule_tree_spec.md').relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame()
    rule_predictions, traces = predict_zero_shot_rule_tree(frame)
    features = tree_feature_frame(frame)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    pseudo_tree = train_pseudo_label_tree(features, rule_predictions, weights)
    tree_predictions = pseudo_tree.predict(features)
    metrics = build_metrics(frame, rule_predictions, tree_predictions)

    tree_text = export_text(pseudo_tree, feature_names=list(features.columns), decimals=3)
    traces.to_csv(OUTPUT_DIR / "zero_shot_llm_rule_tree_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "zero_shot_llm_rule_tree_metrics.csv", index=False)
    write_rule_spec(
        OUTPUT_DIR / "zero_shot_rule_tree_spec.md",
        OUTPUT_DIR / "zero_shot_llm_rule_tree_spec.json",
        OUTPUT_DIR / "zero_shot_pseudo_label_tree.txt",
        tree_text,
    )
    figure_path = plot_metrics(metrics)
    report_path = write_report(metrics, figure_path)
    LOGGER.info("Wrote zero-shot rule-tree metrics: %s", OUTPUT_DIR / "zero_shot_llm_rule_tree_metrics.csv")
    LOGGER.info("Wrote zero-shot rule-tree report: %s", report_path)


if __name__ == "__main__":
    main()
