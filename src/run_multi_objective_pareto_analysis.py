"""Analyze Pareto trade-offs for event-adapted household travel prediction."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIP_METRICS = PROJECT_ROOT / "outputs" / "label_free_llm_adaptation" / "label_free_llm_adaptation_metrics.csv"
DEFAULT_MODE_METRICS = PROJECT_ROOT / "outputs" / "mode_composition_extension" / "mode_composition_metrics.csv"
DEFAULT_MODE_TRIP_METRICS = (
    PROJECT_ROOT / "outputs" / "mode_composition_extension" / "mode_specific_trip_count_metrics.csv"
)
DEFAULT_BATCH_SUMMARY = (
    PROJECT_ROOT / "outputs" / "llm_event_features" / "household_cohort_batch_prompt_b15_summary.md"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multi_objective_pareto"
PRIMARY_TRIP_METHOD = "gated_trip_suppression_a1_d0p15"
BEST_MAE_TRIP_METHOD = "llm_trip_suppression_a1p25"
TRADITIONAL_TRIP_METHOD = "historical_xgboost"
PRIMARY_MODE_METHOD = "llm_transit_avoidance_a1"
TRADITIONAL_MODE_METHOD = "historical_xgboost"
PRIMARY_MODE_TRIP_METHOD = "gated_count_x_llm_mode"
TRADITIONAL_MODE_TRIP_METHOD = "traditional_count_x_traditional_mode"


@dataclass(frozen=True)
class PreferenceProfile:
    """Multi-objective deployment preference."""

    name: str
    weighted_mae: float
    abs_weighted_bias: float
    weighted_rmse: float
    llm_request_cost: float
    description: str


PREFERENCE_PROFILES: tuple[PreferenceProfile, ...] = (
    PreferenceProfile(
        name="minimum_mae_sensitivity",
        weighted_mae=1.00,
        abs_weighted_bias=0.00,
        weighted_rmse=0.00,
        llm_request_cost=0.00,
        description="Select the lowest household weighted MAE regardless of calibration or request cost.",
    ),
    PreferenceProfile(
        name="calibration_first",
        weighted_mae=0.35,
        abs_weighted_bias=0.55,
        weighted_rmse=0.10,
        llm_request_cost=0.00,
        description="Prioritize near-zero aggregate bias for planning totals.",
    ),
    PreferenceProfile(
        name="balanced_course_report",
        weighted_mae=0.55,
        abs_weighted_bias=0.35,
        weighted_rmse=0.10,
        llm_request_cost=0.00,
        description="Balance household accuracy and calibration for the main course-report claim.",
    ),
    PreferenceProfile(
        name="low_cost_deployment",
        weighted_mae=0.45,
        abs_weighted_bias=0.20,
        weighted_rmse=0.05,
        llm_request_cost=0.30,
        description="Prefer cheap event-level correction when API budget is binding.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-metrics-path", type=Path, default=DEFAULT_TRIP_METRICS)
    parser.add_argument("--mode-metrics-path", type=Path, default=DEFAULT_MODE_METRICS)
    parser.add_argument("--mode-trip-metrics-path", type=Path, default=DEFAULT_MODE_TRIP_METRICS)
    parser.add_argument("--batch-summary-path", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_batch_request_count(path: Path) -> int:
    if not path.exists():
        return 89
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Batches:\s*`?(\d+)`?", text)
    if not match:
        return 89
    return int(match.group(1))


def method_family(method: str) -> str:
    if method.startswith("historical"):
        return "traditional"
    if method.startswith("random"):
        return "negative_control"
    if method.startswith("global"):
        return "global_event_prior"
    if method.startswith("gated"):
        return "gated_llm_adapter"
    if method.startswith("llm_only"):
        return "llm_only_prior"
    if method.startswith("llm"):
        return "llm_event_adapter"
    return "other"


def request_cost_proxy(method: str, batch_requests: int) -> int:
    family = method_family(method)
    if family in {"traditional", "negative_control"}:
        return 0
    if family == "global_event_prior":
        return 1
    return batch_requests


def aggregate_trip_metrics(metrics: pd.DataFrame, batch_requests: int) -> pd.DataFrame:
    numeric_columns = [
        "mae",
        "rmse",
        "bias",
        "r2",
        "weighted_mae",
        "weighted_rmse",
        "weighted_bias",
        "weighted_r2",
    ]
    agg_spec = {column: "mean" for column in numeric_columns if column in metrics.columns}
    agg_spec["seed"] = "count"
    frame = metrics.groupby("method", as_index=False).agg(agg_spec).rename(columns={"seed": "runs"})
    frame["abs_weighted_bias"] = frame["weighted_bias"].abs()
    frame["method_family"] = frame["method"].map(method_family)
    frame["llm_request_cost"] = frame["method"].map(lambda value: request_cost_proxy(value, batch_requests))
    frame["llm_request_cost_norm"] = normalize_minimize(frame["llm_request_cost"])
    frame["performance_pareto"] = pareto_mask(frame, ["weighted_mae", "abs_weighted_bias", "weighted_rmse"])
    frame["deployment_pareto"] = pareto_mask(
        frame,
        ["weighted_mae", "abs_weighted_bias", "weighted_rmse", "llm_request_cost"],
    )
    return frame.sort_values(["weighted_mae", "abs_weighted_bias"]).reset_index(drop=True)


def normalize_minimize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - minimum) / (maximum - minimum)


def pareto_mask(frame: pd.DataFrame, objective_columns: list[str]) -> np.ndarray:
    objectives = frame[objective_columns].to_numpy(dtype=float)
    efficient = np.ones(len(frame), dtype=bool)
    for index, current in enumerate(objectives):
        dominated_by_other = np.all(objectives <= current, axis=1) & np.any(objectives < current, axis=1)
        dominated_by_other[index] = False
        if dominated_by_other.any():
            efficient[index] = False
    return efficient


def select_operating_points(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ["weighted_mae", "abs_weighted_bias", "weighted_rmse", "llm_request_cost"]:
        normalized[f"{column}_norm"] = normalize_minimize(normalized[column])

    rows: list[dict[str, float | str | bool]] = []
    for profile in PREFERENCE_PROFILES:
        score = (
            profile.weighted_mae * normalized["weighted_mae_norm"]
            + profile.abs_weighted_bias * normalized["abs_weighted_bias_norm"]
            + profile.weighted_rmse * normalized["weighted_rmse_norm"]
            + profile.llm_request_cost * normalized["llm_request_cost_norm"]
        )
        selected = frame.loc[score.idxmin()]
        rows.append(
            {
                "profile": profile.name,
                "description": profile.description,
                "method": selected["method"],
                "method_family": selected["method_family"],
                "score": float(score.loc[selected.name]),
                "weighted_mae": float(selected["weighted_mae"]),
                "weighted_rmse": float(selected["weighted_rmse"]),
                "weighted_bias": float(selected["weighted_bias"]),
                "abs_weighted_bias": float(selected["abs_weighted_bias"]),
                "weighted_r2": float(selected["weighted_r2"]),
                "llm_request_cost": float(selected["llm_request_cost"]),
                "performance_pareto": bool(selected["performance_pareto"]),
                "deployment_pareto": bool(selected["deployment_pareto"]),
            }
        )
    return pd.DataFrame(rows)


def compute_behavior_improvements(
    trip_frame: pd.DataFrame,
    mode_frame: pd.DataFrame,
    mode_trip_frame: pd.DataFrame,
) -> pd.DataFrame:
    trip_base = get_single_row(trip_frame, TRADITIONAL_TRIP_METHOD)
    trip_primary = get_single_row(trip_frame, PRIMARY_TRIP_METHOD)
    mode_base = get_single_row(mode_frame, TRADITIONAL_MODE_METHOD)
    mode_primary = get_single_row(mode_frame, PRIMARY_MODE_METHOD)
    mode_trip_base = get_single_row(mode_trip_frame, TRADITIONAL_MODE_TRIP_METHOD)
    mode_trip_primary = get_single_row(mode_trip_frame, PRIMARY_MODE_TRIP_METHOD)
    rows = [
        metric_reduction_row(
            "trip_count_weighted_mae",
            "Trip generation",
            float(trip_base["weighted_mae"]),
            float(trip_primary["weighted_mae"]),
        ),
        metric_reduction_row(
            "trip_count_abs_weighted_bias",
            "Trip generation calibration",
            abs(float(trip_base["weighted_bias"])),
            abs(float(trip_primary["weighted_bias"])),
        ),
        metric_reduction_row(
            "mode_weighted_total_variation",
            "Mode composition",
            float(mode_base["weighted_total_variation"]),
            float(mode_primary["weighted_total_variation"]),
        ),
        metric_reduction_row(
            "transit_share_weighted_mae",
            "Sustainable mode signal",
            float(mode_base["transit_share_weighted_mae"]),
            float(mode_primary["transit_share_weighted_mae"]),
        ),
        metric_reduction_row(
            "mode_specific_trip_volume_mae",
            "Mode-specific trip volume",
            float(mode_trip_base["weighted_total_mode_trip_mae"]),
            float(mode_trip_primary["weighted_total_mode_trip_mae"]),
        ),
    ]
    return pd.DataFrame(rows)


def get_single_row(frame: pd.DataFrame, method: str) -> pd.Series:
    matches = frame[frame["method"] == method]
    if matches.empty:
        raise ValueError(f"Missing method: {method}")
    return matches.iloc[0]


def metric_reduction_row(metric: str, objective: str, baseline: float, adapted: float) -> dict[str, float | str]:
    reduction = baseline - adapted
    reduction_pct = 100.0 * reduction / baseline if not np.isclose(baseline, 0.0) else float("nan")
    return {
        "metric": metric,
        "objective": objective,
        "baseline": baseline,
        "adapted": adapted,
        "absolute_reduction": reduction,
        "percent_reduction": reduction_pct,
    }


def plot_pareto(frame: pd.DataFrame, operating_points: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "trip_pareto_frontier.png"
    colors = {
        "traditional": "#4E79A7",
        "negative_control": "#BAB0AC",
        "global_event_prior": "#F28E2B",
        "gated_llm_adapter": "#59A14F",
        "llm_only_prior": "#B07AA1",
        "llm_event_adapter": "#E15759",
        "other": "#76B7B2",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for family, group in frame.groupby("method_family"):
        ax.scatter(
            group["weighted_mae"],
            group["abs_weighted_bias"],
            s=70,
            color=colors.get(family, "#76B7B2"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.82,
            label=family.replace("_", " "),
        )
    pareto = frame[frame["performance_pareto"]].sort_values("weighted_mae")
    ax.plot(pareto["weighted_mae"], pareto["abs_weighted_bias"], color="#222222", linewidth=1.4, alpha=0.75)
    ax.scatter(
        pareto["weighted_mae"],
        pareto["abs_weighted_bias"],
        s=135,
        facecolor="none",
        edgecolor="#222222",
        linewidth=1.4,
        label="performance Pareto frontier",
    )
    for method, label in {
        PRIMARY_TRIP_METHOD: "primary gated",
        BEST_MAE_TRIP_METHOD: "lowest MAE",
        TRADITIONAL_TRIP_METHOD: "traditional",
    }.items():
        row = get_single_row(frame, method)
        ax.annotate(
            label,
            xy=(float(row["weighted_mae"]), float(row["abs_weighted_bias"])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    selected_methods = set(operating_points["method"].tolist())
    selected = frame[frame["method"].isin(selected_methods)]
    ax.scatter(
        selected["weighted_mae"],
        selected["abs_weighted_bias"],
        s=180,
        facecolor="none",
        edgecolor="#000000",
        linewidth=2.0,
        label="selected by preference",
    )
    ax.set_xlabel("Weighted MAE, household trips")
    ax.set_ylabel("Absolute weighted bias, household trips")
    ax.set_title("Trip-generation Pareto frontier: accuracy vs calibration")
    ax.grid(True, color="#E6E6E6", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_behavior_improvements(summary: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "behavior_system_improvement.png"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = summary["objective"].tolist()
    values = summary["percent_reduction"].to_numpy(dtype=float)
    positions = np.arange(len(summary))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#76B7B2"]
    ax.barh(positions, values, color=colors[: len(values)])
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Error reduction vs traditional baseline (%)")
    ax.set_title("Multi-output behavior system: where the event adapter helps")
    ax.axvline(0, color="#333333", linewidth=0.8)
    for position, value in zip(positions, values):
        ax.text(value + 0.8, position, f"{value:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    trip_frame: pd.DataFrame,
    operating_points: pd.DataFrame,
    behavior_summary: pd.DataFrame,
    batch_requests: int,
    output_dir: Path,
) -> Path:
    path = output_dir / "multi_objective_pareto_report.md"
    primary = get_single_row(trip_frame, PRIMARY_TRIP_METHOD)
    best_mae = get_single_row(trip_frame, BEST_MAE_TRIP_METHOD)
    traditional = get_single_row(trip_frame, TRADITIONAL_TRIP_METHOD)
    lines = [
        "# Multi-Objective Pareto Analysis",
        "",
        "## Scope",
        "",
        "This analysis turns the existing adapter grid into a planning-oriented multi-objective evaluation. It does not retrain any model. It asks which operating point should be selected when accuracy, calibration, behavior-system fidelity, and LLM request cost are considered together.",
        "",
        "## LLM Cost Proxy",
        "",
        f"- Traditional and random-control methods: `0` LLM requests.",
        f"- Global event prior methods: `1` event-level request proxy.",
        f"- Cohort-aware LLM adapter methods: `{batch_requests}` batched cohort requests with batch size 15.",
        "",
        "## Key Trip-Generation Trade-Off",
        "",
        "| Method | Weighted MAE | Abs weighted bias | Weighted RMSE | Weighted R2 | Request cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [traditional, best_mae, primary]:
        lines.append(
            f"| {row.method} | {row.weighted_mae:.4f} | {row.abs_weighted_bias:.4f} | "
            f"{row.weighted_rmse:.4f} | {row.weighted_r2:.4f} | {int(row.llm_request_cost)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the single-objective sensitivity candidate is slightly more accurate on MAE, but the primary gated adapter has much lower aggregate bias and better weighted R2. This is why the project should present the result as a multi-objective operating-point choice rather than a single leaderboard.",
            "",
            "## Preference-Based Operating Points",
            "",
            "| Profile | Selected method | Family | Score | Weighted MAE | Abs bias | Request cost |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in operating_points.itertuples(index=False):
        lines.append(
            f"| {row.profile} | {row.method} | {row.method_family} | {row.score:.4f} | "
            f"{row.weighted_mae:.4f} | {row.abs_weighted_bias:.4f} | {int(row.llm_request_cost)} |"
        )
    lines.extend(
        [
            "",
            "## Multi-Output Behavior Summary",
            "",
            "| Objective | Metric | Traditional | Event-adapted | Reduction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in behavior_summary.itertuples(index=False):
        lines.append(
            f"| {row.objective} | {row.metric} | {row.baseline:.4f} | {row.adapted:.4f} | "
            f"{row.percent_reduction:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paper Story Implication",
            "",
            "The stronger claim is not that an LLM directly predicts travel better. The stronger claim is that a large-small model system can expose a Pareto set of event-adapted predictions: the small structured model learns routine household heterogeneity, the LLM supplies event mechanisms, and the solver selects an auditable operating point for the planning objective.",
            "",
            "This matches recent LLM-assisted optimization work: use the LLM to parse event context and stakeholder priorities, then use a deterministic solver or adapter grid to make the numerical decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_requests = read_batch_request_count(args.batch_summary_path)
    trip_metrics = pd.read_csv(args.trip_metrics_path)
    mode_metrics = pd.read_csv(args.mode_metrics_path)
    mode_trip_metrics = pd.read_csv(args.mode_trip_metrics_path)

    trip_frame = aggregate_trip_metrics(trip_metrics, batch_requests)
    operating_points = select_operating_points(trip_frame)
    behavior_summary = compute_behavior_improvements(trip_frame, mode_metrics, mode_trip_metrics)

    trip_frame.to_csv(args.output_dir / "trip_pareto_candidates.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    operating_points.to_csv(args.output_dir / "preference_operating_points.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    behavior_summary.to_csv(args.output_dir / "behavior_system_improvement.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pareto_plot = plot_pareto(trip_frame, operating_points, args.output_dir)
    improvement_plot = plot_behavior_improvements(behavior_summary, args.output_dir)
    report_path = write_report(trip_frame, operating_points, behavior_summary, batch_requests, args.output_dir)
    LOGGER.info("Wrote Pareto candidates: %s", args.output_dir / "trip_pareto_candidates.csv")
    LOGGER.info("Wrote operating points: %s", args.output_dir / "preference_operating_points.csv")
    LOGGER.info("Wrote plots: %s and %s", pareto_plot, improvement_plot)
    LOGGER.info("Wrote report: %s", report_path)


if __name__ == "__main__":
    main()
