"""Evaluate LLM-rule structures calibrated by small historical samples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOUSEHOLD_PATH = PROJECT_ROOT / "data" / "processed" / "household_harmonized.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "models" / "final_label_free_2022" / "final_2022_predictions.csv"
ZERO_SHOT_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "zero_shot_llm_rule_tree_baseline"
    / "zero_shot_llm_rule_tree_metrics.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_rule_small_data_calibration"

TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
RANDOM_SEED = 20260610
SAMPLE_SIZES = (50, 100, 250, 500, 1000)
REPEATS = 20
EVENT_ALPHA = 1.0
MIN_EVENT_FACTOR = 0.05
LEAF_SHRINKAGE_ROWS = 20.0

HOUSEHOLD_COLUMNS = [
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
    TARGET,
    WEIGHT,
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeafCalibrator:
    """Historical leaf-value calibrator for LLM-style routine rules."""

    sample_size: int
    seed: int
    global_mean: float
    leaf_values: dict[str, float]


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
        "weighted_mae": weighted_average(abs_error, weights),
        "weighted_rmse": float(np.sqrt(weighted_average(np.square(error), weights))),
        "weighted_bias": weighted_average(error, weights),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_prediction_mean": weighted_average(y_pred, weights),
        "exact_rounded_accuracy": float(np.mean(np.rint(y_pred) == np.rint(y_true))),
        "within_2_trips": float(np.mean(abs_error <= 2.0)),
        "within_3_trips": float(np.mean(abs_error <= 3.0)),
        "weighted_within_2_trips": weighted_average((abs_error <= 2.0).astype(float), weights),
        "weighted_within_3_trips": weighted_average((abs_error <= 3.0).astype(float), weights),
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(values.median())


def assign_routine_leaf(frame: pd.DataFrame) -> pd.Series:
    hsize = numeric_series(frame, "HHSIZE")
    vehicles = numeric_series(frame, "HHVEHCNT")
    workers = numeric_series(frame, "WRKCOUNT")
    urban = numeric_series(frame, "URBAN")

    conditions = [
        (vehicles <= 0) & (workers >= 1),
        vehicles <= 0,
        (hsize >= 4) & (vehicles >= 2),
        hsize >= 4,
        (workers >= 2) & (vehicles >= 2),
        workers >= 1,
        (workers <= 0) & (urban == 1),
    ]
    choices = [
        "carless_worker",
        "carless_nonworker",
        "large_multi_vehicle",
        "large_low_vehicle",
        "multi_worker_multi_vehicle",
        "worker_vehicle_household",
        "urban_nonworker",
    ]
    leaves = np.select(conditions, choices, default="nonworker_other")
    return pd.Series(leaves, index=frame.index, name="llm_routine_leaf")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    households = pd.read_csv(HOUSEHOLD_PATH, usecols=HOUSEHOLD_COLUMNS, dtype={"HOUSEID": str}, low_memory=False)
    history = households.loc[households["survey_year"].isin([2001, 2009, 2017])].copy()
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"HOUSEID": str})
    covariates = households.loc[households["survey_year"] == 2022].drop_duplicates("HOUSEID")
    merged = predictions.merge(
        covariates.drop(columns=["survey_year", TARGET, WEIGHT]),
        on="HOUSEID",
        how="left",
        validate="one_to_one",
    )
    for frame in (history, merged):
        frame["llm_routine_leaf"] = assign_routine_leaf(frame)
    return history, merged


def fit_leaf_calibrator(sample: pd.DataFrame, sample_size: int, seed: int) -> LeafCalibrator:
    target = sample[TARGET].to_numpy(dtype=float)
    weights = sample[WEIGHT].to_numpy(dtype=float)
    global_mean = weighted_average(target, weights)
    leaf_values: dict[str, float] = {}
    for leaf, group in sample.groupby("llm_routine_leaf"):
        group_target = group[TARGET].to_numpy(dtype=float)
        group_weights = group[WEIGHT].to_numpy(dtype=float)
        leaf_mean = weighted_average(group_target, group_weights)
        shrink_weight = len(group) / (len(group) + LEAF_SHRINKAGE_ROWS)
        leaf_values[str(leaf)] = float(shrink_weight * leaf_mean + (1.0 - shrink_weight) * global_mean)
    return LeafCalibrator(sample_size=sample_size, seed=seed, global_mean=global_mean, leaf_values=leaf_values)


def predict_routine_base(frame: pd.DataFrame, calibrator: LeafCalibrator) -> np.ndarray:
    base = frame["llm_routine_leaf"].map(calibrator.leaf_values).fillna(calibrator.global_mean)
    return np.maximum(base.to_numpy(dtype=float), 0.0)


def event_factor(frame: pd.DataFrame) -> np.ndarray:
    pressure = frame["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    return np.clip(1.0 - EVENT_ALPHA * pressure, MIN_EVENT_FACTOR, 1.0)


def calibrated_rule_prediction(frame: pd.DataFrame, calibrator: LeafCalibrator) -> np.ndarray:
    routine_base = predict_routine_base(frame, calibrator)
    return np.maximum(routine_base * event_factor(frame), 0.0)


def sample_history(history: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size >= len(history):
        return history.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(history.index.to_numpy(), size=sample_size, replace=False)
    return history.loc[indices].copy()


def run_calibration_grid(history: pd.DataFrame, target_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = target_frame[TARGET].to_numpy(dtype=float)
    weights = target_frame[WEIGHT].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []
    leaf_rows: list[dict[str, float | int | str]] = []

    for sample_size in SAMPLE_SIZES:
        for repeat in range(REPEATS):
            seed = RANDOM_SEED + sample_size * 100 + repeat
            sample = sample_history(history, sample_size, seed)
            calibrator = fit_leaf_calibrator(sample, sample_size, seed)
            prediction = calibrated_rule_prediction(target_frame, calibrator)
            rows.append(
                {
                    "method": f"llm_rule_small_hist_calibrated_n{sample_size}",
                    "sample_size": sample_size,
                    "seed": seed,
                    "calibration_rows": len(sample),
                    "event_alpha": EVENT_ALPHA,
                    **evaluate(y_true, prediction, weights),
                }
            )
            for leaf, value in calibrator.leaf_values.items():
                leaf_rows.append(
                    {
                        "method": f"llm_rule_small_hist_calibrated_n{sample_size}",
                        "sample_size": sample_size,
                        "seed": seed,
                        "leaf": leaf,
                        "calibrated_leaf_value": value,
                    }
                )

    full_calibrator = fit_leaf_calibrator(history, len(history), RANDOM_SEED)
    full_prediction = calibrated_rule_prediction(target_frame, full_calibrator)
    rows.append(
        {
            "method": "llm_rule_full_history_calibrated",
            "sample_size": len(history),
            "seed": RANDOM_SEED,
            "calibration_rows": len(history),
            "event_alpha": EVENT_ALPHA,
            **evaluate(y_true, full_prediction, weights),
        }
    )
    for leaf, value in full_calibrator.leaf_values.items():
        leaf_rows.append(
            {
                "method": "llm_rule_full_history_calibrated",
                "sample_size": len(history),
                "seed": RANDOM_SEED,
                "leaf": leaf,
                "calibrated_leaf_value": value,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(leaf_rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs.groupby(["method", "sample_size"], as_index=False)
        .agg(
            runs=("seed", "count"),
            weighted_mae_mean=("weighted_mae", "mean"),
            weighted_mae_std=("weighted_mae", "std"),
            weighted_rmse_mean=("weighted_rmse", "mean"),
            weighted_bias_mean=("weighted_bias", "mean"),
            weighted_r2_mean=("weighted_r2", "mean"),
            weighted_within_2_trips_mean=("weighted_within_2_trips", "mean"),
            weighted_within_3_trips_mean=("weighted_within_3_trips", "mean"),
        )
        .sort_values("sample_size")
    )
    summary["weighted_mae_std"] = summary["weighted_mae_std"].fillna(0.0)
    return summary


def load_static_method_metrics() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"HOUSEID": str})
    y_true = predictions[TARGET].to_numpy(dtype=float)
    weights = predictions[WEIGHT].to_numpy(dtype=float)
    rows = [
        {
            "method": "historical_xgboost",
            "family": "data_only",
            **evaluate(y_true, predictions["prediction_historical_xgboost"].to_numpy(dtype=float), weights),
        },
        {
            "method": "llm_only_pressure",
            "family": "llm_only",
            **evaluate(y_true, predictions["prediction_llm_only_trip_suppression_a1p25"].to_numpy(dtype=float), weights),
        },
        {
            "method": "primary_hybrid_gated_adapter",
            "family": "hybrid_event_adapter",
            **evaluate(y_true, predictions["prediction_gated_trip_suppression_a1_d0p15"].to_numpy(dtype=float), weights),
        },
    ]
    if ZERO_SHOT_METRICS_PATH.exists():
        zero = pd.read_csv(ZERO_SHOT_METRICS_PATH)
        for method in ["zero_shot_llm_rule_tree", "zero_shot_pseudo_label_tree"]:
            row = zero.loc[zero["method"] == method].iloc[0]
            rows.append(
                {
                    "method": method,
                    "family": "llm_rule_baseline",
                    "weighted_mae": float(row["weighted_mae"]),
                    "weighted_rmse": float(row["weighted_rmse"]),
                    "weighted_bias": float(row["weighted_bias"]),
                    "weighted_r2": float(row["weighted_r2"]),
                    "weighted_prediction_mean": float(row["weighted_prediction_mean"]),
                    "exact_rounded_accuracy": float(row["exact_rounded_accuracy"]),
                    "within_2_trips": float(row["within_2_trips"]),
                    "within_3_trips": float(row["within_3_trips"]),
                    "weighted_within_2_trips": float(row["weighted_within_2_trips"]),
                    "weighted_within_3_trips": float(row["weighted_within_3_trips"]),
                }
            )
    return pd.DataFrame(rows)


def build_method_spectrum(static_metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    chosen = summary.loc[summary["sample_size"].isin([100, 500, 1000])].copy()
    calibrated_rows = []
    for row in chosen.itertuples(index=False):
        calibrated_rows.append(
            {
                "method": row.method,
                "family": "llm_rule_small_data_calibration",
                "weighted_mae": row.weighted_mae_mean,
                "weighted_rmse": row.weighted_rmse_mean,
                "weighted_bias": row.weighted_bias_mean,
                "weighted_r2": row.weighted_r2_mean,
                "weighted_within_2_trips": row.weighted_within_2_trips_mean,
                "weighted_within_3_trips": row.weighted_within_3_trips_mean,
            }
        )
    full = summary.loc[summary["method"] == "llm_rule_full_history_calibrated"].iloc[0]
    calibrated_rows.append(
        {
            "method": "llm_rule_full_history_calibrated",
            "family": "llm_rule_full_history_calibration",
            "weighted_mae": full.weighted_mae_mean,
            "weighted_rmse": full.weighted_rmse_mean,
            "weighted_bias": full.weighted_bias_mean,
            "weighted_r2": full.weighted_r2_mean,
            "weighted_within_2_trips": full.weighted_within_2_trips_mean,
            "weighted_within_3_trips": full.weighted_within_3_trips_mean,
        }
    )
    spectrum = pd.concat([static_metrics, pd.DataFrame(calibrated_rows)], ignore_index=True, sort=False)
    primary_mae = float(spectrum.loc[spectrum["method"] == "primary_hybrid_gated_adapter", "weighted_mae"].iloc[0])
    historical_mae = float(spectrum.loc[spectrum["method"] == "historical_xgboost", "weighted_mae"].iloc[0])
    spectrum["weighted_mae_delta_vs_primary"] = spectrum["weighted_mae"] - primary_mae
    spectrum["weighted_mae_gain_vs_historical_pct"] = 100.0 * (historical_mae - spectrum["weighted_mae"]) / historical_mae
    return spectrum


def plot_spectrum(spectrum: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "llm_rule_small_data_calibration_spectrum.png"
    order = [
        "historical_xgboost",
        "llm_only_pressure",
        "zero_shot_llm_rule_tree",
        "llm_rule_small_hist_calibrated_n100",
        "llm_rule_small_hist_calibrated_n500",
        "llm_rule_small_hist_calibrated_n1000",
        "primary_hybrid_gated_adapter",
    ]
    labels = {
        "historical_xgboost": "Data-only\nXGBoost",
        "llm_only_pressure": "LLM-only\npressure",
        "zero_shot_llm_rule_tree": "Zero-shot\nrule tree",
        "llm_rule_small_hist_calibrated_n100": "Rule +\n100 history",
        "llm_rule_small_hist_calibrated_n500": "Rule +\n500 history",
        "llm_rule_small_hist_calibrated_n1000": "Rule +\n1000 history",
        "primary_hybrid_gated_adapter": "Hybrid\nevent adapter",
    }
    selected = spectrum.set_index("method").loc[order].reset_index()
    colors = ["#94a3b8", "#8b5cf6", "#f59e0b", "#fb923c", "#f97316", "#ea580c", "#16a34a"]

    plt.figure(figsize=(10.2, 4.8))
    bars = plt.bar([labels[m] for m in selected["method"]], selected["weighted_mae"], color=colors, edgecolor="#334155")
    plt.ylabel("Weighted MAE")
    plt.title("Method spectrum: LLM rules, small historical calibration, and hybrid adaptation", fontsize=12)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.035, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, spectrum: pd.DataFrame, figure_path: Path) -> Path:
    path = OUTPUT_DIR / "llm_rule_small_data_calibration_report_zh.md"
    primary = spectrum.loc[spectrum["method"] == "primary_hybrid_gated_adapter"].iloc[0]
    zero = spectrum.loc[spectrum["method"] == "zero_shot_llm_rule_tree"].iloc[0]
    n500 = spectrum.loc[spectrum["method"] == "llm_rule_small_hist_calibrated_n500"].iloc[0]
    n1000 = spectrum.loc[spectrum["method"] == "llm_rule_small_hist_calibrated_n1000"].iloc[0]
    full = spectrum.loc[spectrum["method"] == "llm_rule_full_history_calibrated"].iloc[0]
    best_small = summary.loc[summary["sample_size"].isin(SAMPLE_SIZES)].sort_values("weighted_mae_mean").iloc[0]

    lines = [
        "# LLM Rule + Small Historical Calibration",
        "",
        "## 这个实验回答什么问题",
        "",
        "这个实验回应队友提出的融合路线：先让 LLM 提供可解释的 rule structure，再用少量历史 NHTS 样本校准规则叶节点，而不是先训练完整历史 XGBoost 再用 LLM 修正。",
        "",
        "它和主方法的关系是中间 baseline，而不是替代主方法：",
        "",
        "```text",
        "data-only fitting -> LLM-only -> zero-shot LLM rule -> LLM rule + small historical calibration -> historical model + LLM event adapter",
        "```",
        "",
        "## 设计",
        "",
        "- LLM-style routine rule：按 household size、vehicle ownership、worker count、urban context 分出 routine demand leaves。",
        "- Small historical calibration：只用 2001/2009/2017 的少量样本校准每个 leaf 的 base demand。",
        "- Event adaptation：2022 不使用 `CNTTDHH` 训练或校准，只用已有 LLM trip-suppression pressure 作为固定 event factor。",
        f"- 每个 sample size 重复 `{REPEATS}` 次；event alpha 固定为 `{EVENT_ALPHA}`，不根据 2022 labels 调参。",
        "",
        "## 主要结果",
        "",
        "| 方法 | Weighted MAE | Weighted Bias | Weighted R2 | Delta vs primary |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in [
        "historical_xgboost",
        "llm_only_pressure",
        "zero_shot_llm_rule_tree",
        "llm_rule_small_hist_calibrated_n100",
        "llm_rule_small_hist_calibrated_n500",
        "llm_rule_small_hist_calibrated_n1000",
        "llm_rule_full_history_calibrated",
        "primary_hybrid_gated_adapter",
    ]:
        row = spectrum.loc[spectrum["method"] == method].iloc[0]
        lines.append(
            f"| {method} | {row.weighted_mae:.4f} | {row.weighted_bias:+.4f} | "
            f"{row.weighted_r2:.4f} | {row.weighted_mae_delta_vs_primary:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Zero-shot LLM rule tree wMAE `{zero.weighted_mae:.4f}`，主方法 `{primary.weighted_mae:.4f}`。",
            f"- Rule + 500 historical samples wMAE `{n500.weighted_mae:.4f}`，比 zero-shot rule tree {'更好' if n500.weighted_mae < zero.weighted_mae else '更弱'}。",
            f"- Rule + 1000 historical samples wMAE `{n1000.weighted_mae:.4f}`。",
            f"- Full historical calibration wMAE `{full.weighted_mae:.4f}`。",
            f"- 最好的 small-data calibrated rule 是 `{best_small.method}`，平均 wMAE `{best_small.weighted_mae_mean:.4f}`。",
            "",
            "这个结果把两条路线统一了：LLM 规则 + 少量历史校准是合理的 data-sparse/cold-start 中间方法，但在当前 NHTS 2022 event-shift 主任务上，完整历史模型提供的 household grounding 仍然更强。因此论文/汇报可以把它作为方法谱系中的桥梁，而不是和 hybrid adapter 对立。",
            "",
            "## Artifacts",
            "",
            f"- Run metrics: `{(OUTPUT_DIR / 'small_data_calibration_metrics_by_run.csv').relative_to(PROJECT_ROOT)}`",
            f"- Summary: `{(OUTPUT_DIR / 'small_data_calibration_summary.csv').relative_to(PROJECT_ROOT)}`",
            f"- Method spectrum: `{(OUTPUT_DIR / 'method_spectrum_metrics.csv').relative_to(PROJECT_ROOT)}`",
            f"- Figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
            f"- Leaf values: `{(OUTPUT_DIR / 'small_data_leaf_values.csv').relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history, target_frame = load_frames()
    runs, leaf_values = run_calibration_grid(history, target_frame)
    summary = summarize_runs(runs)
    static_metrics = load_static_method_metrics()
    spectrum = build_method_spectrum(static_metrics, summary)
    figure_path = plot_spectrum(spectrum)

    runs.to_csv(OUTPUT_DIR / "small_data_calibration_metrics_by_run.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "small_data_calibration_summary.csv", index=False)
    spectrum.to_csv(OUTPUT_DIR / "method_spectrum_metrics.csv", index=False)
    leaf_values.to_csv(OUTPUT_DIR / "small_data_leaf_values.csv", index=False)
    report_path = write_report(runs, summary, spectrum, figure_path)
    LOGGER.info("Wrote small-data calibration report: %s", report_path)


if __name__ == "__main__":
    main()
