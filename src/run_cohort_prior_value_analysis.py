"""Analyze where cohort-specific LLM event priors add value over global priors."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "CNTTDHH"
WEIGHT_COLUMN = "WTHHFIN"
ID_COLUMN = "HOUSEID"
SUBGROUP_COLUMNS: tuple[str, ...] = (
    "WRKCOUNT",
    "HHVEHCNT",
    "URBRUR",
    "RAIL",
    "HHFAMINC",
    "CENSUS_R",
    "HHSIZE",
)
METHOD_COLUMNS: dict[str, str] = {
    "historical_xgboost": "prediction_historical_xgboost",
    "global_trip_suppression_a1": "prediction_global_trip_suppression_a1",
    "global_trip_suppression_a1p25": "prediction_global_trip_suppression_a1p25",
    "cohort_trip_suppression_a1": "prediction_cohort_trip_suppression_a1",
    "primary_gated_adapter": "prediction_gated_trip_suppression_a1_d0p15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=Path("outputs/models/final_label_free_2022/final_2022_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cohort_prior_value_analysis"))
    parser.add_argument("--min-subgroup-rows", type=int, default=50)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return weighted_average(np.abs(y_pred - y_true), weights)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(y_pred - y_true), weights=weights)))


def weighted_bias(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return weighted_average(y_pred - y_true, weights)


def add_same_alpha_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fair same-alpha global/cohort predictions used only for value isolation."""
    enriched = frame.copy()
    base_prediction = enriched["base_prediction"].to_numpy(dtype=float)
    weights = enriched[WEIGHT_COLUMN].to_numpy(dtype=float)
    pressure = enriched["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    global_pressure_value = weighted_average(pressure, weights)
    global_pressure = np.full(len(enriched), global_pressure_value, dtype=float)
    enriched["prediction_global_trip_suppression_a1"] = np.maximum(base_prediction * (1.0 - global_pressure), 0.0)
    enriched["prediction_cohort_trip_suppression_a1"] = np.maximum(base_prediction * (1.0 - pressure), 0.0)
    enriched["gated_uses_cohort_prior"] = np.abs(pressure - global_pressure_value) >= 0.15
    return enriched


def load_analysis_frame(dataset_path: Path, prediction_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path, dtype={ID_COLUMN: str})
    household = pd.read_csv(
        dataset_path,
        usecols=[ID_COLUMN, "survey_year", *SUBGROUP_COLUMNS],
        dtype={ID_COLUMN: str},
        low_memory=False,
    )
    household_2022 = household.loc[household["survey_year"] == 2022].drop(columns=["survey_year"])
    frame = predictions.merge(household_2022, on=ID_COLUMN, how="left", validate="one_to_one")
    missing = int(frame[list(SUBGROUP_COLUMNS)].isna().all(axis=1).sum())
    if missing:
        raise ValueError(f"Missing subgroup attributes for {missing} prediction rows.")
    return add_same_alpha_predictions(frame)


def metric_row(frame: pd.DataFrame, method: str, prediction_column: str) -> dict[str, float | int | str]:
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    return {
        "method": method,
        "rows": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "weighted_mae": weighted_mae(y_true, y_pred, weights),
        "weighted_rmse": weighted_rmse(y_true, y_pred, weights),
        "weighted_bias": weighted_bias(y_true, y_pred, weights),
        "abs_weighted_bias": abs(weighted_bias(y_true, y_pred, weights)),
        "weighted_target_mean": weighted_average(y_true, weights),
        "weighted_prediction_mean": weighted_average(y_pred, weights),
    }


def build_overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [metric_row(frame, method, column) for method, column in METHOD_COLUMNS.items()]
    return pd.DataFrame(rows).sort_values("weighted_mae").reset_index(drop=True)


def build_subgroup_metrics(frame: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subgroup_column in SUBGROUP_COLUMNS:
        for subgroup_value, subgroup_frame in frame.groupby(subgroup_column, dropna=False):
            if len(subgroup_frame) < min_rows:
                continue
            for method, prediction_column in METHOD_COLUMNS.items():
                row = metric_row(subgroup_frame, method, prediction_column)
                row["subgroup_column"] = subgroup_column
                row["subgroup_value"] = str(subgroup_value)
                rows.append(row)
    return pd.DataFrame(rows)


def build_comparison(subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["subgroup_column", "subgroup_value", "rows", "weight_sum"]
    value_columns = ["weighted_mae", "weighted_rmse", "weighted_bias", "abs_weighted_bias"]
    pivot = subgroup_metrics.pivot_table(
        index=index_columns,
        columns="method",
        values=value_columns,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
    comparison = pivot.reset_index()
    primary_mae = comparison["weighted_mae_primary_gated_adapter"]
    history_mae = comparison["weighted_mae_historical_xgboost"]
    global_a1_mae = comparison["weighted_mae_global_trip_suppression_a1"]
    global_a1p25_mae = comparison["weighted_mae_global_trip_suppression_a1p25"]
    cohort_a1_mae = comparison["weighted_mae_cohort_trip_suppression_a1"]
    comparison["primary_vs_history_mae_delta"] = history_mae - primary_mae
    comparison["primary_vs_global_a1_mae_delta"] = global_a1_mae - primary_mae
    comparison["primary_vs_global_a1p25_mae_delta"] = global_a1p25_mae - primary_mae
    comparison["cohort_a1_vs_global_a1_mae_delta"] = global_a1_mae - cohort_a1_mae
    comparison["primary_vs_global_a1_abs_bias_delta"] = (
        comparison["abs_weighted_bias_global_trip_suppression_a1"]
        - comparison["abs_weighted_bias_primary_gated_adapter"]
    )
    comparison["primary_beats_global_a1"] = comparison["primary_vs_global_a1_mae_delta"] > 0
    comparison["primary_beats_global_a1p25"] = comparison["primary_vs_global_a1p25_mae_delta"] > 0
    return comparison.sort_values("primary_vs_global_a1_mae_delta", ascending=False).reset_index(drop=True)


def summarize_value(frame: pd.DataFrame, overall: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    primary = overall.loc[overall["method"] == "primary_gated_adapter"].iloc[0]
    global_a1 = overall.loc[overall["method"] == "global_trip_suppression_a1"].iloc[0]
    global_a1p25 = overall.loc[overall["method"] == "global_trip_suppression_a1p25"].iloc[0]
    cohort_share_rows = float(frame["gated_uses_cohort_prior"].mean())
    cohort_share_weight = weighted_average(
        frame["gated_uses_cohort_prior"].astype(float).to_numpy(),
        frame[WEIGHT_COLUMN].to_numpy(dtype=float),
    )
    rows = [
        {
            "quantity": "overall_primary_vs_global_a1_mae_delta",
            "value": float(global_a1["weighted_mae"] - primary["weighted_mae"]),
            "interpretation": "positive means primary gated adapter beats same-alpha global prior",
        },
        {
            "quantity": "overall_primary_vs_global_a1p25_mae_delta",
            "value": float(global_a1p25["weighted_mae"] - primary["weighted_mae"]),
            "interpretation": "positive means primary gated adapter beats reported low-cost global baseline",
        },
        {
            "quantity": "overall_primary_vs_global_a1_abs_bias_delta",
            "value": float(global_a1["abs_weighted_bias"] - primary["abs_weighted_bias"]),
            "interpretation": "positive means primary gated adapter is better calibrated",
        },
        {
            "quantity": "share_subgroup_cells_primary_beats_global_a1",
            "value": float(comparison["primary_beats_global_a1"].mean()),
            "interpretation": "unweighted share across evaluated subgroup cells",
        },
        {
            "quantity": "share_subgroup_cells_primary_beats_global_a1p25",
            "value": float(comparison["primary_beats_global_a1p25"].mean()),
            "interpretation": "unweighted share across evaluated subgroup cells",
        },
        {
            "quantity": "median_primary_vs_global_a1_mae_delta",
            "value": float(comparison["primary_vs_global_a1_mae_delta"].median()),
            "interpretation": "median subgroup-cell MAE gain over same-alpha global prior",
        },
        {
            "quantity": "gated_uses_cohort_prior_share_rows",
            "value": cohort_share_rows,
            "interpretation": "share of households where the gate uses cohort-specific LLM pressure",
        },
        {
            "quantity": "gated_uses_cohort_prior_share_weighted",
            "value": cohort_share_weight,
            "interpretation": "survey-weighted share where the gate uses cohort-specific LLM pressure",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def plot_top_groups(comparison: pd.DataFrame, output_path: Path) -> None:
    eligible = comparison.copy()
    top = eligible.head(8)
    bottom = eligible.tail(6).sort_values("primary_vs_global_a1_mae_delta", ascending=True)
    plot_frame = pd.concat([bottom, top], ignore_index=True)
    labels = plot_frame["subgroup_column"].astype(str) + "=" + plot_frame["subgroup_value"].astype(str)
    values = plot_frame["primary_vs_global_a1_mae_delta"].to_numpy(dtype=float)
    colors = np.where(values >= 0, "#16A34A", "#F97316")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Weighted MAE gain vs same-alpha global prior")
    ax.set_ylabel("Subgroup cell")
    ax.set_title("Where cohort-specific LLM priors add value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report(
    overall: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    path = output_dir / "cohort_prior_value_report.md"
    top_columns = [
        "subgroup_column",
        "subgroup_value",
        "rows",
        "primary_vs_global_a1_mae_delta",
        "primary_vs_global_a1p25_mae_delta",
        "primary_vs_global_a1_abs_bias_delta",
    ]
    top_groups = comparison.head(10)
    weak_groups = comparison.tail(10).sort_values("primary_vs_global_a1_mae_delta", ascending=True)
    overall_columns = ["method", "weighted_mae", "weighted_rmse", "weighted_bias", "abs_weighted_bias"]
    summary_lookup = summary.set_index("quantity")["value"]
    same_alpha_delta = float(summary_lookup["overall_primary_vs_global_a1_mae_delta"])
    reported_global_delta = float(summary_lookup["overall_primary_vs_global_a1p25_mae_delta"])
    subgroup_win_share = float(summary_lookup["share_subgroup_cells_primary_beats_global_a1"])
    weighted_gate_share = float(summary_lookup["gated_uses_cohort_prior_share_weighted"])
    lines = [
        "# Cohort-Prior Value Analysis",
        "",
        "## Scope",
        "",
        "This analysis addresses the reviewer question: if a global post-pandemic downscaling prior is already strong, where do cohort-specific LLM event priors add value?",
        "",
        "The main comparison uses the final `primary_gated_adapter` and a same-alpha `global_trip_suppression_a1` control to isolate cohort-specific assignment from correction strength. It also reports the previously used low-cost `global_trip_suppression_a1p25` baseline for consistency with the main result table.",
        "",
        "## Overall Metrics",
        "",
        *markdown_table(overall, overall_columns),
        "",
        "## Summary",
        "",
        *markdown_table(summary, ["quantity", "value", "interpretation"]),
        "",
        "Main takeaway:",
        "",
        (
            f"- The primary gated adapter is `{same_alpha_delta:.4f}` weighted-MAE lower than the same-alpha global prior "
            f"and `{reported_global_delta:.4f}` lower than the reported global-a1.25 prior."
        ),
        (
            f"- It beats the same-alpha global prior in `{subgroup_win_share:.1%}` of evaluated subgroup cells, "
            f"while using cohort-specific pressure for `{weighted_gate_share:.1%}` of survey-weighted households."
        ),
        "",
        "## Subgroups Where Cohort-Specific Priors Help Most",
        "",
        *markdown_table(top_groups, top_columns),
        "",
        "## Subgroups Where Global Prior Is More Competitive",
        "",
        *markdown_table(weak_groups, top_columns),
        "",
        "## Interpretation",
        "",
        "- The global event prior remains a strong baseline and should stay visible in the paper.",
        "- The final gated adapter is not a claim that cohort-specific ranking explains all gains; it is a calibration-first mechanism that uses cohort-specific priors only when they differ meaningfully from the global event pressure.",
        "- Positive subgroup deltas show where differentiated LLM priors add value beyond event-level downscaling; negative deltas show groups where the common event correction is enough or more stable.",
        "- These subgroup cells overlap across variables, so the shares should be interpreted as diagnostic coverage rather than independent population partitions.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame(args.dataset_path, args.prediction_path)
    overall = build_overall_metrics(frame)
    subgroup_metrics = build_subgroup_metrics(frame, args.min_subgroup_rows)
    comparison = build_comparison(subgroup_metrics)
    summary = summarize_value(frame, overall, comparison)
    overall.to_csv(args.output_dir / "cohort_prior_value_overall_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    subgroup_metrics.to_csv(args.output_dir / "cohort_prior_value_subgroup_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    comparison.to_csv(args.output_dir / "cohort_prior_value_comparison.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.output_dir / "cohort_prior_value_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    plot_top_groups(comparison, args.output_dir / "cohort_prior_value_top_groups.png")
    report_path = write_report(overall, comparison, summary, args.output_dir)
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
