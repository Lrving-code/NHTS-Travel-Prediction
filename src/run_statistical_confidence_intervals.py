"""Bootstrap confidence intervals for final household behavior metrics."""

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
RANDOM_SEED = 20260610
TRIP_METHODS: dict[str, str] = {
    "traditional_supervised_baseline": "prediction_historical_xgboost",
    "llm_only_pressure": "prediction_llm_only_trip_suppression_a1p25",
    "global_event_prior": "prediction_global_trip_suppression_a1p25",
    "llm_trip_suppression": "prediction_llm_trip_suppression_a1p25",
    "gated_llm_correction": "prediction_gated_trip_suppression_a1_d0p15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trip-predictions-path",
        type=Path,
        default=Path("outputs/models/final_label_free_2022/final_2022_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/statistical_validation"),
    )
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_average(y_true, weights)
    numerator = np.average(np.square(y_true - y_pred), weights=weights)
    denominator = np.average(np.square(y_true - mean), weights=weights)
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def evaluate_trip_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    y_true = frame[TARGET].to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    return {
        "weighted_mae": weighted_average(abs_error, weights),
        "weighted_rmse": float(np.sqrt(weighted_average(np.square(error), weights))),
        "weighted_bias": weighted_average(error, weights),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_within_1_trip": weighted_average((abs_error <= 1.0).astype(float), weights),
        "weighted_within_2_trips": weighted_average((abs_error <= 2.0).astype(float), weights),
        "weighted_within_3_trips": weighted_average((abs_error <= 3.0).astype(float), weights),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def bootstrap_trip_metrics(frame: pd.DataFrame, runs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    point_rows: list[dict[str, float | str]] = []
    boot_rows: list[dict[str, float | str | int]] = []
    for method, column in TRIP_METHODS.items():
        point_rows.append({"method": method, **evaluate_trip_metrics(frame, column)})

    row_count = len(frame)
    for run in range(1, runs + 1):
        indices = rng.integers(0, row_count, size=row_count)
        sample = frame.iloc[indices]
        for method, column in TRIP_METHODS.items():
            boot_rows.append({"run": run, "method": method, **evaluate_trip_metrics(sample, column)})
    return pd.DataFrame(point_rows), pd.DataFrame(boot_rows)


def summarize_bootstrap(point: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in point.columns if column != "method"]
    rows: list[dict[str, float | str]] = []
    for method, group in boot.groupby("method"):
        point_row = point[point["method"] == method].iloc[0]
        for metric in metric_columns:
            summary = summarize(group[metric].to_numpy(dtype=float))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "point": float(point_row[metric]),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def paired_improvements(boot: pd.DataFrame, baseline: str, target: str) -> pd.DataFrame:
    baseline_frame = boot[boot["method"] == baseline].set_index("run")
    target_frame = boot[boot["method"] == target].set_index("run")
    rows: list[dict[str, float | str]] = []
    for metric in ["weighted_mae", "weighted_rmse", "weighted_bias", "weighted_within_2_trips"]:
        if metric == "weighted_bias":
            values = np.abs(baseline_frame[metric].to_numpy(dtype=float)) - np.abs(
                target_frame[metric].to_numpy(dtype=float)
            )
            label = "absolute_bias_reduction"
        elif metric == "weighted_within_2_trips":
            values = target_frame[metric].to_numpy(dtype=float) - baseline_frame[metric].to_numpy(dtype=float)
            label = "within2_gain"
        else:
            values = baseline_frame[metric].to_numpy(dtype=float) - target_frame[metric].to_numpy(dtype=float)
            label = f"{metric}_reduction"
        rows.append({"comparison": f"{target}_vs_{baseline}", "metric": label, **summarize(values)})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "confidence_interval_report.md"
    main = summary[
        (summary["method"].isin(["traditional_supervised_baseline", "gated_llm_correction"]))
        & (summary["metric"].isin(["weighted_mae", "weighted_bias", "weighted_within_2_trips"]))
    ]
    lines = [
        "# Statistical Validation Report",
        "",
        "## Scope",
        "",
        "This report uses household bootstrap resampling to estimate uncertainty for the final 2022 trip-count metrics. It keeps survey weights inside each bootstrap sample.",
        "",
        "## Key Confidence Intervals",
        "",
        "| Method | Metric | Point | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in main.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.metric} | {row.point:.4f} | "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] |"
        )
    lines.extend(["", "## Paired Improvement vs Traditional Baseline", "", "| Metric | Mean | 95% CI |", "|---|---:|---:|"])
    for row in paired.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.mean:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The main result should be presented with uncertainty rather than a single point estimate. If the paired MAE-reduction interval stays positive, it supports the claim that the hybrid event adapter robustly improves household trip-count prediction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.trip_predictions_path)
    point, boot = bootstrap_trip_metrics(frame, args.bootstrap_runs, args.seed)
    summary = summarize_bootstrap(point, boot)
    paired = paired_improvements(boot, "traditional_supervised_baseline", "gated_llm_correction")
    point.to_csv(args.output_dir / "trip_metric_points.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    boot.to_csv(args.output_dir / "trip_metric_bootstrap_samples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.output_dir / "trip_metric_confidence_intervals.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    paired.to_csv(args.output_dir / "paired_improvement_confidence_intervals.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    report_path = write_report(summary, paired, args.output_dir)
    LOGGER.info("Wrote confidence interval report: %s", report_path)


if __name__ == "__main__":
    main()
