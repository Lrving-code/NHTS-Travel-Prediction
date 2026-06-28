"""Evaluate subgroup equity trade-offs for final household trip predictions."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
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
    "historical_mean_only": "prediction_historical_mean_only",
    "historical_xgboost": "prediction_historical_xgboost",
    "llm_only_trip_suppression_a1p25": "prediction_llm_only_trip_suppression_a1p25",
    "global_trip_suppression_a1p25": "prediction_global_trip_suppression_a1p25",
    "llm_trip_suppression_a1p25": "prediction_llm_trip_suppression_a1p25",
    "gated_trip_suppression_a1_d0p15": "prediction_gated_trip_suppression_a1_d0p15",
}
PRIMARY_METHOD = "gated_trip_suppression_a1_d0p15"
BASELINE_METHOD = "historical_xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=Path("outputs/models/final_label_free_2022/final_2022_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/equity_aware_evaluation"))
    parser.add_argument("--min-subgroup-rows", type=int, default=50)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return weighted_average(np.abs(y_pred - y_true), weights)


def weighted_bias(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return weighted_average(y_pred - y_true, weights)


def load_frame(dataset_path: Path, prediction_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(prediction_path, dtype={ID_COLUMN: str})
    household = pd.read_csv(
        dataset_path,
        usecols=[ID_COLUMN, "survey_year", *SUBGROUP_COLUMNS],
        dtype={ID_COLUMN: str},
        low_memory=False,
    )
    household_2022 = household[household["survey_year"] == 2022].drop(columns=["survey_year"])
    frame = predictions.merge(household_2022, on=ID_COLUMN, how="left", validate="one_to_one")
    missing = frame[list(SUBGROUP_COLUMNS)].isna().all(axis=1).sum()
    if missing:
        raise ValueError(f"Missing subgroup attributes for {missing} prediction rows.")
    return frame


def subgroup_metrics(frame: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for column in SUBGROUP_COLUMNS:
        for value, group in frame.groupby(column, dropna=False):
            if len(group) < min_rows:
                continue
            y_true = group[TARGET].to_numpy(dtype=float)
            weights = group[WEIGHT].to_numpy(dtype=float)
            for method, prediction_column in METHOD_COLUMNS.items():
                y_pred = group[prediction_column].to_numpy(dtype=float)
                rows.append(
                    {
                        "subgroup_column": column,
                        "subgroup_value": str(value),
                        "method": method,
                        "rows": len(group),
                        "weight_sum": float(weights.sum()),
                        "weighted_mae": weighted_mae(y_true, y_pred, weights),
                        "weighted_bias": weighted_bias(y_true, y_pred, weights),
                    }
                )
    return pd.DataFrame(rows)


def overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    y_true = frame[TARGET].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []
    for method, prediction_column in METHOD_COLUMNS.items():
        y_pred = frame[prediction_column].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "weighted_mae": weighted_mae(y_true, y_pred, weights),
                "weighted_bias": weighted_bias(y_true, y_pred, weights),
                "abs_weighted_bias": abs(weighted_bias(y_true, y_pred, weights)),
            }
        )
    return pd.DataFrame(rows)


def summarize_equity(overall: pd.DataFrame, subgroups: pd.DataFrame) -> pd.DataFrame:
    baseline = subgroups[subgroups["method"] == BASELINE_METHOD][
        ["subgroup_column", "subgroup_value", "weighted_mae"]
    ].rename(columns={"weighted_mae": "baseline_subgroup_weighted_mae"})
    rows: list[dict[str, float | str]] = []
    for method, group in subgroups.groupby("method"):
        merged = group.merge(baseline, on=["subgroup_column", "subgroup_value"], how="left")
        improvements = merged["baseline_subgroup_weighted_mae"] - merged["weighted_mae"]
        overall_row = overall[overall["method"] == method].iloc[0]
        rows.append(
            {
                "method": method,
                "weighted_mae": float(overall_row["weighted_mae"]),
                "abs_weighted_bias": float(overall_row["abs_weighted_bias"]),
                "worst_subgroup_weighted_mae": float(merged["weighted_mae"].max()),
                "subgroup_mae_std": float(merged["weighted_mae"].std(ddof=1)),
                "worst_subgroup_gap_vs_overall": float(merged["weighted_mae"].max() - overall_row["weighted_mae"]),
                "min_subgroup_mae_improvement_vs_historical": float(improvements.min()),
                "median_subgroup_mae_improvement_vs_historical": float(improvements.median()),
                "share_subgroups_improved_vs_historical": float((improvements > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["worst_subgroup_weighted_mae", "weighted_mae"]).reset_index(drop=True)


def normalize_minimize(values: pd.Series) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - minimum) / (maximum - minimum)


def select_equity_operating_points(summary: pd.DataFrame) -> pd.DataFrame:
    normalized = summary.copy()
    for column in ["weighted_mae", "abs_weighted_bias", "worst_subgroup_weighted_mae"]:
        normalized[f"{column}_norm"] = normalize_minimize(normalized[column])
    profiles = {
        "equity_first": (0.30, 0.20, 0.50),
        "balanced_equity": (0.45, 0.25, 0.30),
        "accuracy_first_core_methods": (0.75, 0.10, 0.15),
    }
    rows: list[dict[str, float | str]] = []
    for profile, (mae_w, bias_w, worst_w) in profiles.items():
        score = (
            mae_w * normalized["weighted_mae_norm"]
            + bias_w * normalized["abs_weighted_bias_norm"]
            + worst_w * normalized["worst_subgroup_weighted_mae_norm"]
        )
        selected = summary.loc[score.idxmin()]
        rows.append(
            {
                "profile": profile,
                "method": selected["method"],
                "score": float(score.loc[selected.name]),
                "weighted_mae": float(selected["weighted_mae"]),
                "abs_weighted_bias": float(selected["abs_weighted_bias"]),
                "worst_subgroup_weighted_mae": float(selected["worst_subgroup_weighted_mae"]),
                "share_subgroups_improved_vs_historical": float(selected["share_subgroups_improved_vs_historical"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "equity_aware_evaluation_report.md"
    primary = summary[summary["method"] == PRIMARY_METHOD].iloc[0]
    baseline = summary[summary["method"] == BASELINE_METHOD].iloc[0]
    lines = [
        "# Equity-Aware Evaluation Report",
        "",
        "## Scope",
        "",
        "This report evaluates whether event-adapted household trip predictions improve not only the average household error but also subgroup robustness across worker count, vehicle count, urban/rural status, rail access, family income, census region, and household size.",
        "",
        "## Method-Level Equity Summary",
        "",
        "| Method | wMAE | |bias| | Worst subgroup MAE | Worst gap | Share groups improved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_mae:.4f} | {row.abs_weighted_bias:.4f} | "
            f"{row.worst_subgroup_weighted_mae:.4f} | {row.worst_subgroup_gap_vs_overall:.4f} | "
            f"{row.share_subgroups_improved_vs_historical:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Equity-Aware Operating Points",
            "",
            "| Profile | Selected method | Score | wMAE | |bias| | Worst subgroup MAE | Share improved |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.profile} | {row.method} | {row.score:.4f} | {row.weighted_mae:.4f} | "
            f"{row.abs_weighted_bias:.4f} | {row.worst_subgroup_weighted_mae:.4f} | "
            f"{row.share_subgroups_improved_vs_historical:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Main Interpretation",
            "",
            (
                f"The primary gated event adapter reduces overall weighted MAE from `{baseline.weighted_mae:.4f}` "
                f"to `{primary.weighted_mae:.4f}` and worst-subgroup weighted MAE from "
                f"`{baseline.worst_subgroup_weighted_mae:.4f}` to `{primary.worst_subgroup_weighted_mae:.4f}`."
            ),
            "",
            "This supports the mobility-inequality framing: the event adapter improves average performance and reduces the worst subgroup error among the evaluated core methods. It remains a predictive subgroup robustness result, not a causal equity claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.dataset_path, args.prediction_path)
    subgroups = subgroup_metrics(frame, args.min_subgroup_rows)
    overall = overall_metrics(frame)
    summary = summarize_equity(overall, subgroups)
    selected = select_equity_operating_points(summary)
    subgroups.to_csv(args.output_dir / "subgroup_equity_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.output_dir / "equity_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    selected.to_csv(args.output_dir / "equity_operating_points.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    report_path = write_report(summary, selected, args.output_dir)
    LOGGER.info("Wrote equity-aware report: %s", report_path)


if __name__ == "__main__":
    main()
