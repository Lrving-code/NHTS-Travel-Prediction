"""Generate paper-style analysis figures for the final presentation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_project"
MODE_DIR = PROJECT_ROOT / "outputs" / "mode_composition_extension"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "final_label_free_2022"
SUBGROUP_DIR = PROJECT_ROOT / "outputs" / "label_free_llm_adaptation"
OUTPUT_DIR = FINAL_DIR / "figures" / "paper_style"

TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
BASE = "historical_xgboost"
LLM_ONLY = "llm_only_trip_suppression_a1p25"
GLOBAL = "global_trip_suppression_a1p25"
HYBRID = "gated_trip_suppression_a1_d0p15"

LOGGER = logging.getLogger(__name__)

PALETTE = {
    "Historical XGBoost": "#be123c",
    "LLM-only pressure": "#6d28d9",
    "Global event rule": "#00796b",
    "Hybrid gated": "#198754",
    "Historical mean": "#64748b",
    "Trend / indicator": "#ea580c",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_mae(error: np.ndarray, weights: np.ndarray) -> float:
    return weighted_mean(np.abs(error), weights)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    LOGGER.info("Wrote %s", path)


def load_predictions() -> pd.DataFrame:
    path = MODEL_DIR / "final_2022_predictions.csv"
    if not path.exists():
        msg = f"Missing prediction artifact: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def plot_mae_bias_tradeoff() -> Path:
    metrics = pd.read_csv(FINAL_DIR / "final_metrics_summary.csv")
    display = {
        "historical_mean_only": "Historical mean",
        "historical_xgboost": "Historical XGBoost",
        "historical_mean_trend_shift": "Trend / indicator",
        "llm_only_trip_suppression_a1p25": "LLM-only pressure",
        "global_trip_suppression_a1p25": "Global event rule",
        HYBRID: "Hybrid gated",
    }
    frame = metrics[metrics["method"].isin(display)].copy()
    frame["display"] = frame["method"].map(display)
    frame["abs_bias"] = frame["weighted_bias"].abs()

    plt.figure(figsize=(7.3, 4.3))
    ax = sns.scatterplot(
        data=frame,
        x="abs_bias",
        y="weighted_mae",
        hue="display",
        size="weighted_rmse",
        sizes=(80, 240),
        palette=PALETTE,
        legend=False,
    )
    for row in frame.itertuples():
        ax.annotate(row.display, (row.abs_bias, row.weighted_mae), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    ax.set_xlabel("|weighted bias|, trips / household")
    ax.set_ylabel("weighted MAE, trips / household")
    ax.set_title("Accuracy-bias tradeoff across adaptation strategies", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "mae_bias_tradeoff.png"
    save_fig(return_path)
    return return_path


def plot_error_distribution(pred: pd.DataFrame) -> Path:
    weights = pred[WEIGHT].to_numpy(dtype=float)
    series = [
        ("Historical XGBoost", pred[f"error_{BASE}"].to_numpy(dtype=float), PALETTE["Historical XGBoost"]),
        ("LLM-only pressure", pred[f"error_{LLM_ONLY}"].to_numpy(dtype=float), PALETTE["LLM-only pressure"]),
        ("Hybrid gated", pred[f"error_{HYBRID}"].to_numpy(dtype=float), PALETTE["Hybrid gated"]),
    ]
    bins = np.linspace(-8, 12, 81)
    plt.figure(figsize=(7.3, 4.3))
    for label, error, color in series:
        hist, edges = np.histogram(np.clip(error, bins[0], bins[-1]), bins=bins, weights=weights, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, hist, label=label, color=color, linewidth=2)
        plt.fill_between(centers, hist, alpha=0.10, color=color)
    plt.axvline(0, color="#0f172a", linewidth=1, alpha=0.65)
    plt.xlabel("prediction error = predicted - observed trips")
    plt.ylabel("weighted density")
    plt.title("Error distribution reveals the post-pandemic over-prediction shift", fontsize=12, fontweight="bold")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "error_distribution.png"
    save_fig(return_path)
    return return_path


def plot_tolerance_curve(pred: pd.DataFrame) -> Path:
    weights = pred[WEIGHT].to_numpy(dtype=float)
    methods = [
        ("Historical XGBoost", f"error_{BASE}", PALETTE["Historical XGBoost"]),
        ("LLM-only pressure", f"error_{LLM_ONLY}", PALETTE["LLM-only pressure"]),
        ("Global event rule", f"error_{GLOBAL}", PALETTE["Global event rule"]),
        ("Hybrid gated", f"error_{HYBRID}", PALETTE["Hybrid gated"]),
    ]
    tolerance = np.arange(0, 7)
    plt.figure(figsize=(7.3, 4.3))
    for label, column, color in methods:
        abs_error = pred[column].abs().to_numpy(dtype=float)
        values = [weighted_mean((abs_error <= threshold).astype(float), weights) for threshold in tolerance]
        plt.plot(tolerance, values, marker="o", linewidth=2.2, label=label, color=color)
    plt.xlabel("allowed absolute error, trips")
    plt.ylabel("weighted household coverage")
    plt.title("Practical household accuracy under different tolerance levels", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "tolerance_curve.png"
    save_fig(return_path)
    return return_path


def plot_pressure_quintile_gain(pred: pd.DataFrame) -> Path:
    frame = pred.copy()
    frame["pressure_bin"] = pd.qcut(
        frame["llm_trip_suppression_pressure"],
        q=5,
        labels=[f"Q{i}" for i in range(1, 6)],
        duplicates="drop",
    )
    rows = []
    for label, group in frame.groupby("pressure_bin", observed=False):
        weights = group[WEIGHT].to_numpy(dtype=float)
        base_mae = weighted_mae(group[f"error_{BASE}"].to_numpy(dtype=float), weights)
        hybrid_mae = weighted_mae(group[f"error_{HYBRID}"].to_numpy(dtype=float), weights)
        rows.append({"bin": str(label), "Historical XGBoost": base_mae, "Hybrid gated": hybrid_mae})
    result = pd.DataFrame(rows).melt(id_vars="bin", var_name="method", value_name="weighted_mae")

    plt.figure(figsize=(7.3, 4.3))
    ax = sns.barplot(data=result, x="bin", y="weighted_mae", hue="method", palette=PALETTE)
    ax.set_xlabel("LLM trip-suppression pressure quintile")
    ax.set_ylabel("weighted MAE")
    ax.set_title("Adaptation gains persist across event-pressure strata", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "pressure_quintile_gain.png"
    save_fig(return_path)
    return return_path


def plot_event_prior_heatmap(pred: pd.DataFrame) -> Path:
    frame = pred.copy()
    frame["absolute_error_reduction"] = frame[f"error_{BASE}"].abs() - frame[f"error_{HYBRID}"].abs()
    frame["correction_amount"] = frame["prediction_historical_xgboost"] - frame[f"prediction_{HYBRID}"]
    columns = [
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "transit_avoidance_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
        "llm_trip_suppression_pressure",
        "correction_amount",
        "absolute_error_reduction",
    ]
    labels = [
        "trip suppression",
        "remote work",
        "transit avoidance",
        "delivery substitution",
        "recovery sensitivity",
        "event pressure",
        "correction amount",
        "abs-error reduction",
    ]
    corr = frame[columns].corr(method="spearman")
    corr.index = labels
    corr.columns = labels
    plt.figure(figsize=(7.0, 5.3))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", square=True, annot=True, fmt=".2f", annot_kws={"fontsize": 6})
    plt.title("Spearman structure of LLM event priors and correction behavior", fontsize=12, fontweight="bold")
    return_path = OUTPUT_DIR / "event_prior_heatmap.png"
    save_fig(return_path)
    return return_path


def plot_subgroup_gain() -> Path:
    path = SUBGROUP_DIR / "label_free_subgroup_comparison.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["rows"] >= 200].copy()
    frame["label"] = frame["subgroup_column"].astype(str) + "=" + frame["subgroup_value"].astype(str)
    frame = frame.sort_values("llm_vs_historical_mae_delta", ascending=False).head(10)
    plt.figure(figsize=(7.3, 4.3))
    ax = sns.barplot(data=frame, y="label", x="llm_vs_historical_mae_delta", color="#198754")
    ax.set_xlabel("weighted MAE reduction vs historical XGBoost")
    ax.set_ylabel("")
    ax.set_title("Largest subgroup gains from event-aware adaptation", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "subgroup_gain_top.png"
    save_fig(return_path)
    return return_path


def plot_mode_component_mae() -> Path:
    metrics = pd.read_csv(MODE_DIR / "mode_composition_metrics.csv")
    base = metrics.loc[metrics["method"] == "historical_xgboost"].iloc[0]
    llm = metrics.loc[metrics["method"] == "llm_transit_avoidance_a1"].iloc[0]
    modes = ["private_vehicle", "walk", "bike", "transit", "taxi_ridehail", "other"]
    labels = ["private", "walk", "bike", "transit", "taxi", "other"]
    rows = []
    for mode, label in zip(modes, labels, strict=True):
        rows.append({"mode": label, "method": "Historical XGBoost", "weighted_mae": base[f"{mode}_share_weighted_mae"]})
        rows.append({"mode": label, "method": "XGBoost + LLM prior", "weighted_mae": llm[f"{mode}_share_weighted_mae"]})
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(7.3, 4.3))
    ax = sns.barplot(
        data=frame,
        x="mode",
        y="weighted_mae",
        hue="method",
        palette={"Historical XGBoost": "#be123c", "XGBoost + LLM prior": "#198754"},
    )
    ax.set_xlabel("household mode-share component")
    ax.set_ylabel("weighted share MAE")
    ax.set_title("Mode-composition gains concentrate on public-transit behavior", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    return_path = OUTPUT_DIR / "mode_component_mae.png"
    save_fig(return_path)
    return return_path


def main() -> None:
    configure_logging()
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    pred = load_predictions()
    paths = [
        plot_mae_bias_tradeoff(),
        plot_error_distribution(pred),
        plot_tolerance_curve(pred),
        plot_pressure_quintile_gain(pred),
        plot_event_prior_heatmap(pred),
        plot_subgroup_gain(),
        plot_mode_component_mae(),
    ]
    LOGGER.info("Generated %d paper-style figures in %s", len(paths), OUTPUT_DIR)


if __name__ == "__main__":
    main()
