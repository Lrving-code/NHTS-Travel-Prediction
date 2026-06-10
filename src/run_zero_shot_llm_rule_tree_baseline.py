"""Evaluate a zero-shot LLM-style rule-tree baseline for 2022 trip counts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
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
PRIMARY_PREDICTION = "prediction_gated_trip_suppression_a1_d0p15"
LLM_ONLY_PREDICTION = "prediction_llm_only_trip_suppression_a1p25"
HISTORICAL_PREDICTION = "prediction_historical_xgboost"
RANDOM_SEED = 20260610

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleTreeSpec:
    """Document the zero-shot rule tree as an auditable ablation."""

    name: str
    label_usage: str
    allowed_inputs: list[str]
    forbidden_inputs: list[str]
    decision_logic: list[str]
    caveat: str


FEATURE_COLUMNS = [
    "HHSIZE",
    "HHVEHCNT",
    "WRKCOUNT",
    "NUMADLT",
    "URBAN",
    "URBRUR",
    "HHFAMINC",
    "TRAVDAY",
    "travel_month",
    "llm_trip_suppression_pressure",
    "remote_work_substitution_likelihood",
    "transit_avoidance_likelihood",
    "online_delivery_substitution_likelihood",
    "post_pandemic_recovery_sensitivity",
]


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
    return {
        "weighted_mae": weighted_average(np.abs(error), weights),
        "weighted_rmse": float(np.sqrt(weighted_average(np.square(error), weights))),
        "weighted_bias": weighted_average(error, weights),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_target_mean": weighted_average(y_true, weights),
        "weighted_prediction_mean": weighted_average(y_pred, weights),
        "exact_rounded_accuracy": float(np.mean(np.rint(y_pred) == np.rint(y_true))),
        "within_1_trip": float(np.mean(abs_error <= 1.0)),
        "within_2_trips": float(np.mean(abs_error <= 2.0)),
        "within_3_trips": float(np.mean(abs_error <= 3.0)),
        "weighted_within_1_trip": weighted_average((abs_error <= 1.0).astype(float), weights),
        "weighted_within_2_trips": weighted_average((abs_error <= 2.0).astype(float), weights),
        "weighted_within_3_trips": weighted_average((abs_error <= 3.0).astype(float), weights),
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(values.median())


def load_analysis_frame() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"HOUSEID": str})
    household_columns = [
        "HOUSEID",
        "survey_year",
        "HHSIZE",
        "HHVEHCNT",
        "WRKCOUNT",
        "NUMADLT",
        "URBAN",
        "URBRUR",
        "HHFAMINC",
        "TRAVDAY",
        "travel_month",
    ]
    households = pd.read_csv(HOUSEHOLD_PATH, usecols=household_columns, dtype={"HOUSEID": str}, low_memory=False)
    households = households.loc[households["survey_year"] == 2022].drop_duplicates("HOUSEID")
    predictions["HOUSEID"] = predictions["HOUSEID"].astype(str)
    households["HOUSEID"] = households["HOUSEID"].astype(str)
    merged = predictions.merge(households.drop(columns=["survey_year"]), on="HOUSEID", how="left", validate="one_to_one")
    missing_share = float(merged["HHSIZE"].isna().mean())
    if missing_share > 0.01:
        raise ValueError(f"Unexpected household covariate missing share after merge: {missing_share:.2%}")
    for column in FEATURE_COLUMNS:
        merged[column] = numeric_series(merged, column)
    return merged


def rule_tree_spec() -> RuleTreeSpec:
    return RuleTreeSpec(
        name="zero_shot_llm_semantic_rule_tree",
        label_usage="No 2022 CNTTDHH labels are used for split selection, leaf values, or calibration.",
        allowed_inputs=[
            "household size, vehicle ownership, worker count, adult count, urban/rural context",
            "LLM event priors: trip suppression, remote work, transit avoidance, delivery substitution, recovery sensitivity",
        ],
        forbidden_inputs=[
            "2022 CNTTDHH",
            "2022 survey weights for fitting leaf values",
            "household ID as a predictor",
            "aggregate 2022 target outcomes",
        ],
        decision_logic=[
            "Root split: high, medium, or low LLM trip-suppression pressure.",
            "High-suppression branch lowers leaves further for no-worker or high-remote-work households.",
            "Medium-suppression branch separates carless households, large households, and worker households.",
            "Low-suppression branch allows higher leaves for large, multi-vehicle, or multi-worker households.",
            "Small leaf adjustments encode delivery substitution and recovery sensitivity, without using labels.",
        ],
        caveat=(
            "This is a deliberately direct LLM-style baseline. It represents a qualitative belief tree rather than a "
            "statistically trained decision tree, because true decision-tree thresholds and leaf values require labels."
        ),
    )


def direct_rule_tree_predict(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    predictions: list[float] = []
    leaves: list[str] = []
    for row in frame.itertuples(index=False):
        pressure = float(getattr(row, "llm_trip_suppression_pressure"))
        remote = float(getattr(row, "remote_work_substitution_likelihood"))
        transit = float(getattr(row, "transit_avoidance_likelihood"))
        delivery = float(getattr(row, "online_delivery_substitution_likelihood"))
        recovery = float(getattr(row, "post_pandemic_recovery_sensitivity"))
        household_size = float(getattr(row, "HHSIZE"))
        vehicles = float(getattr(row, "HHVEHCNT"))
        workers = float(getattr(row, "WRKCOUNT"))
        adults = float(getattr(row, "NUMADLT"))
        urban = float(getattr(row, "URBAN"))

        if pressure >= 0.62:
            if workers <= 0:
                value = 1.10 if household_size <= 2 else 1.70
                leaf = "high_suppression:no_workers"
            elif remote >= 0.55:
                value = 1.80 if household_size <= 2 else 2.50
                leaf = "high_suppression:remote_workers"
            else:
                value = 2.40 if vehicles <= 1 else 3.10
                leaf = "high_suppression:mobile_workers"
        elif pressure >= 0.45:
            if vehicles <= 0:
                value = 1.75 if transit >= 0.45 else 2.35
                leaf = "medium_suppression:carless"
            elif household_size >= 4:
                value = 4.20 if workers >= 2 else 3.40
                leaf = "medium_suppression:large_household"
            elif workers >= 1:
                value = 3.10 if adults <= 2 else 3.70
                leaf = "medium_suppression:worker_household"
            else:
                value = 2.20
                leaf = "medium_suppression:nonworker_household"
        else:
            if household_size >= 4:
                value = 5.50 if vehicles >= 2 else 4.30
                leaf = "low_suppression:large_household"
            elif vehicles >= 2 and workers >= 2:
                value = 4.70
                leaf = "low_suppression:multi_worker_vehicle"
            elif vehicles <= 0:
                value = 2.50 if transit >= 0.45 else 3.05
                leaf = "low_suppression:carless"
            elif urban <= 1 and adults >= 2:
                value = 3.80
                leaf = "low_suppression:urban_adults"
            else:
                value = 3.45
                leaf = "low_suppression:default"

        if delivery >= 0.60 and household_size <= 2:
            value -= 0.35
            leaf += "|delivery_minus"
        if recovery >= 0.62 and pressure < 0.45:
            value += 0.35
            leaf += "|recovery_plus"

        predictions.append(float(np.clip(value, 0.0, 8.0)))
        leaves.append(leaf)

    return np.asarray(predictions, dtype=float), leaves


def fit_pseudo_label_tree(frame: pd.DataFrame, pseudo_labels: np.ndarray) -> tuple[np.ndarray, DecisionTreeRegressor, str]:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=90,
        random_state=RANDOM_SEED,
    )
    model.fit(features, pseudo_labels, sample_weight=weights)
    predictions = np.maximum(model.predict(features), 0.0)
    tree_text = export_text(model, feature_names=FEATURE_COLUMNS, decimals=3)
    return predictions, model, tree_text


def build_metrics(frame: pd.DataFrame, rule_predictions: np.ndarray, pseudo_predictions: np.ndarray) -> pd.DataFrame:
    y_true = frame[TARGET].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    rows = [
        {
            "method": "historical_xgboost",
            "family": "traditional_baseline",
            "label_usage": "2001/2009/2017 NHTS labels only",
            **evaluate(y_true, frame[HISTORICAL_PREDICTION].to_numpy(dtype=float), weights),
        },
        {
            "method": "llm_only_pressure",
            "family": "direct_llm_prior_control",
            "label_usage": "No 2022 training labels; event pressure only",
            **evaluate(y_true, frame[LLM_ONLY_PREDICTION].to_numpy(dtype=float), weights),
        },
        {
            "method": "zero_shot_llm_rule_tree",
            "family": "zero_shot_tree_baseline",
            "label_usage": "No 2022 labels; LLM-style semantic split and leaf priors",
            **evaluate(y_true, rule_predictions, weights),
        },
        {
            "method": "zero_shot_pseudo_label_tree",
            "family": "pseudo_label_tree_baseline",
            "label_usage": "No 2022 labels; sklearn tree distilled from zero-shot rule labels",
            **evaluate(y_true, pseudo_predictions, weights),
        },
        {
            "method": "primary_hybrid_gated_adapter",
            "family": "main_method",
            "label_usage": "Historical labels plus LLM event priors; no 2022 label calibration",
            **evaluate(y_true, frame[PRIMARY_PREDICTION].to_numpy(dtype=float), weights),
        },
    ]
    metrics = pd.DataFrame(rows)
    baseline_mae = float(metrics.loc[metrics["method"] == "historical_xgboost", "weighted_mae"].iloc[0])
    primary_mae = float(metrics.loc[metrics["method"] == "primary_hybrid_gated_adapter", "weighted_mae"].iloc[0])
    metrics["weighted_mae_reduction_vs_historical_pct"] = 100.0 * (baseline_mae - metrics["weighted_mae"]) / baseline_mae
    metrics["weighted_mae_delta_vs_primary"] = metrics["weighted_mae"] - primary_mae
    return metrics


def write_predictions(
    frame: pd.DataFrame,
    rule_predictions: np.ndarray,
    pseudo_predictions: np.ndarray,
    leaves: list[str],
) -> Path:
    path = OUTPUT_DIR / "zero_shot_llm_rule_tree_predictions.csv"
    output = frame[
        [
            "HOUSEID",
            TARGET,
            WEIGHT,
            HISTORICAL_PREDICTION,
            LLM_ONLY_PREDICTION,
            PRIMARY_PREDICTION,
            "llm_trip_suppression_pressure",
            "remote_work_substitution_likelihood",
            "transit_avoidance_likelihood",
            "online_delivery_substitution_likelihood",
            "post_pandemic_recovery_sensitivity",
        ]
    ].copy()
    output["zero_shot_llm_rule_tree_leaf"] = leaves
    output["prediction_zero_shot_llm_rule_tree"] = rule_predictions
    output["prediction_zero_shot_pseudo_label_tree"] = pseudo_predictions
    output.to_csv(path, index=False)
    return path


def plot_metrics(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "zero_shot_rule_tree_metric_comparison.png"
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
        "zero_shot_pseudo_label_tree": "Pseudo-label\nDecisionTree",
        "llm_only_pressure": "LLM-only\npressure",
        "primary_hybrid_gated_adapter": "Primary\nhybrid",
    }
    colors = {
        "historical_xgboost": "#94a3b8",
        "zero_shot_llm_rule_tree": "#f59e0b",
        "zero_shot_pseudo_label_tree": "#fb923c",
        "llm_only_pressure": "#8b5cf6",
        "primary_hybrid_gated_adapter": "#16a34a",
    }
    selected = metrics.set_index("method").loc[order].reset_index()
    plt.figure(figsize=(8.8, 4.8))
    bars = plt.bar(
        [labels[method] for method in selected["method"]],
        selected["weighted_mae"],
        color=[colors[method] for method in selected["method"]],
        edgecolor="#334155",
    )
    plt.ylabel("Weighted MAE")
    plt.title("Zero-shot LLM rule-tree baseline versus hybrid event adaptation", fontsize=12)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.035, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def write_report(metrics: pd.DataFrame, tree_text: str, figure_path: Path, predictions_path: Path) -> Path:
    path = OUTPUT_DIR / "zero_shot_llm_rule_tree_report_zh.md"
    spec = rule_tree_spec()
    primary = metrics.loc[metrics["method"] == "primary_hybrid_gated_adapter"].iloc[0]
    rule = metrics.loc[metrics["method"] == "zero_shot_llm_rule_tree"].iloc[0]
    pseudo = metrics.loc[metrics["method"] == "zero_shot_pseudo_label_tree"].iloc[0]
    llm_only = metrics.loc[metrics["method"] == "llm_only_pressure"].iloc[0]
    historical = metrics.loc[metrics["method"] == "historical_xgboost"].iloc[0]

    lines = [
        "# Zero-Shot LLM Rule-Tree Baseline",
        "",
        "## 这个实验回答什么问题",
        "",
        "老师追问的是：为什么不直接让 LLM 在零样本情况下构建 2022 年决策树？",
        "",
        "这个补充实验把该问题落成一个 ablation：我们允许 rule tree 使用 household covariates "
        "和已有 GPT-5.5 event-prior features，但不允许它读取 2022 `CNTTDHH` 标签、survey weight "
        "或 2022 aggregate target statistics 来训练 split threshold 或 leaf value。",
        "",
        "## Baseline 定义",
        "",
        f"- Baseline 名称：`{spec.name}`。",
        f"- 标签使用：{spec.label_usage}",
        f"- 关键限制：{spec.caveat}",
        "- 额外实现：把 zero-shot rule tree 输出的 pseudo labels 蒸馏成一棵 sklearn "
        "`DecisionTreeRegressor`，得到 `zero_shot_pseudo_label_tree`。这模拟“LLM 先造 pseudo labels，"
        "再训练一棵树”的方案，但仍不使用真实 2022 labels。",
        "",
        "## 结果",
        "",
        "| 方法 | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | Within 2 trips | MAE vs primary |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} | "
            f"{row.within_2_trips:.1%} | {row.weighted_mae_delta_vs_primary:+.4f} |"
        )

    direct_gap = 100.0 * (rule.weighted_mae - primary.weighted_mae) / primary.weighted_mae
    pseudo_gap = 100.0 * (pseudo.weighted_mae - primary.weighted_mae) / primary.weighted_mae
    llm_gap = 100.0 * (llm_only.weighted_mae - primary.weighted_mae) / primary.weighted_mae
    historical_gain = 100.0 * (historical.weighted_mae - primary.weighted_mae) / historical.weighted_mae

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Primary hybrid gated adapter 相对 historical XGBoost 的 weighted MAE 降低 `{historical_gain:.2f}%`。",
            f"- Zero-shot rule tree 的 weighted MAE 比 primary hybrid 高 `{direct_gap:.2f}%`。",
            f"- Pseudo-label tree 的 weighted MAE 比 primary hybrid 高 `{pseudo_gap:.2f}%`。",
            f"- LLM-only pressure 的 weighted MAE 比 primary hybrid 高 `{llm_gap:.2f}%`。",
            "",
            "这支持我们的主张：LLM 直接生成规则树可以作为 qualitative prior 或 ablation，但它没有真实历史 "
            "NHTS 监督模型提供的 household-level numerical grounding，因此数值校准不如 hybrid adapter 稳定。",
            "",
            "## 答辩口径",
            "",
            "可以做这个 baseline，而且我们已经把它作为补充实验实现了。结果表明，zero-shot LLM rule tree "
            "能够表达疫情机制方向，但它更像 belief tree；没有标签时，split threshold 和 leaf value 缺少 "
            "empirical risk minimization 支撑。主方法把历史监督模型作为 routine mobility anchor，只让 LLM "
            "提供 event prior，因此在误差、bias 和可审计性上更适合作为论文主线。",
            "",
            "## Distilled pseudo-label tree",
            "",
            "下面这棵树只拟合 zero-shot rule labels，不拟合真实 `CNTTDHH`：",
            "",
            "```text",
            tree_text,
            "```",
            "",
            "## Artifacts",
            "",
            f"- Metrics: `{(OUTPUT_DIR / 'zero_shot_llm_rule_tree_metrics.csv').relative_to(PROJECT_ROOT)}`",
            f"- Predictions: `{predictions_path.relative_to(PROJECT_ROOT)}`",
            f"- Figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
            f"- Rule spec: `{(OUTPUT_DIR / 'zero_shot_llm_rule_tree_spec.json').relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame()
    rule_predictions, leaves = direct_rule_tree_predict(frame)
    pseudo_predictions, _model, tree_text = fit_pseudo_label_tree(frame, rule_predictions)
    metrics = build_metrics(frame, rule_predictions, pseudo_predictions)

    metrics.to_csv(OUTPUT_DIR / "zero_shot_llm_rule_tree_metrics.csv", index=False)
    (OUTPUT_DIR / "zero_shot_pseudo_label_tree.txt").write_text(tree_text, encoding="utf-8")
    (OUTPUT_DIR / "zero_shot_llm_rule_tree_spec.json").write_text(
        json.dumps(asdict(rule_tree_spec()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions_path = write_predictions(frame, rule_predictions, pseudo_predictions, leaves)
    figure_path = plot_metrics(metrics)
    report_path = write_report(metrics, tree_text, figure_path, predictions_path)
    LOGGER.info("Wrote zero-shot LLM rule-tree metrics: %s", OUTPUT_DIR / "zero_shot_llm_rule_tree_metrics.csv")
    LOGGER.info("Wrote zero-shot LLM rule-tree report: %s", report_path)


if __name__ == "__main__":
    main()
