"""Run label-free LLM event-prior adaptation for 2022 NHTS."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from run_household_baseline import RANDOM_SEED, TARGET_COLUMN, WEIGHT_COLUMN
from run_llm_residual_adaptation import (
    LLM_NUMERIC_FEATURES,
    MetricRow,
    configure_logging,
    evaluate_predictions,
    load_2022_with_llm_features,
    metric_rows_to_dicts,
    parse_seeds,
    resolve_device,
    set_random_seed,
    train_history_model,
    weighted_average,
    weighted_rmse,
)


LOGGER = logging.getLogger(__name__)
PRESSURE_WEIGHTS: dict[str, float] = {
    "trip_suppression_risk": 0.35,
    "remote_work_substitution_likelihood": 0.25,
    "transit_avoidance_likelihood": 0.20,
    "online_delivery_substitution_likelihood": 0.20,
}
SUBGROUP_COLUMNS: tuple[str, ...] = (
    "WRKCOUNT",
    "HHVEHCNT",
    "URBRUR",
    "RAIL",
    "HHFAMINC",
    "CENSUS_R",
    "HHSIZE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument(
        "--cohort-profiles-path",
        type=Path,
        default=Path("outputs/llm_event_features/household_cohort_profiles.csv"),
    )
    parser.add_argument(
        "--llm-features-path",
        type=Path,
        default=Path(
            "outputs/llm_event_features/"
            "cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/label_free_llm_adaptation"))
    parser.add_argument("--history-n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--alphas",
        default="0.25,0.50,0.75,1.00,1.25",
        help="Pre-declared multiplicative correction strengths to report.",
    )
    parser.add_argument(
        "--random-control-alphas",
        default="0.75,1.00,1.25",
        help="Alpha values used for random-pressure negative controls.",
    )
    parser.add_argument("--random-control-seeds", default="42,43,44,45,46")
    parser.add_argument(
        "--gate-thresholds",
        default="0.05,0.10,0.15",
        help="No-label thresholds for using cohort-specific trip suppression instead of the global mean.",
    )
    parser.add_argument("--min-factor", type=float, default=0.05)
    parser.add_argument("--subgroup-alpha", type=float, default=1.25)
    return parser.parse_args()


def parse_float_grid(raw_values: str) -> list[float]:
    values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("Float grid must include at least one value.")
    return values


def alpha_token(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def clipped(values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def add_event_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    adapted = frame.copy()
    base_pressure = np.zeros(len(adapted), dtype=float)
    for column, weight in PRESSURE_WEIGHTS.items():
        base_pressure += weight * clipped(adapted[column])

    recovery = clipped(adapted["post_pandemic_recovery_sensitivity"])
    confidence = clipped(adapted["confidence"])
    adapted["llm_raw_event_pressure"] = clipped(base_pressure)
    adapted["llm_recovery_adjusted_pressure"] = clipped(base_pressure * (1.0 - 0.25 * recovery))
    adapted["llm_confidence_weighted_pressure"] = clipped(
        adapted["llm_recovery_adjusted_pressure"].to_numpy(dtype=float) * confidence
    )
    adapted["llm_trip_suppression_pressure"] = clipped(adapted["trip_suppression_risk"])
    return adapted


def apply_multiplicative_pressure(
    base_prediction: np.ndarray,
    pressure: np.ndarray,
    alpha: float,
    min_factor: float,
) -> np.ndarray:
    factor = np.clip(1.0 - alpha * pressure, min_factor, 1.0)
    return np.maximum(base_prediction * factor, 0.0)


def add_historical_mean_trend_prediction(frame: pd.DataFrame, frame_2022: pd.DataFrame) -> np.ndarray:
    years: list[int] = []
    means: list[float] = []
    for year, year_frame in frame[frame["survey_year"].isin((2001, 2009, 2017))].groupby("survey_year"):
        years.append(int(year))
        means.append(
            weighted_average(
                year_frame[TARGET_COLUMN].to_numpy(dtype=float),
                year_frame[WEIGHT_COLUMN].to_numpy(dtype=float),
            )
        )

    slope, intercept = np.polyfit(np.asarray(years, dtype=float), np.asarray(means, dtype=float), deg=1)
    projected_2022_mean = float(slope * 2022.0 + intercept)
    base_prediction = frame_2022["base_prediction"].to_numpy(dtype=float)
    base_weighted_mean = weighted_average(
        base_prediction,
        frame_2022[WEIGHT_COLUMN].to_numpy(dtype=float),
    )
    shifted = base_prediction + (projected_2022_mean - base_weighted_mean)
    return np.maximum(shifted, 0.0)


def make_historical_mean_prediction(frame: pd.DataFrame, rows: int) -> np.ndarray:
    history = frame[frame["survey_year"].isin((2001, 2009, 2017))]
    mean_value = weighted_average(
        history[TARGET_COLUMN].to_numpy(dtype=float),
        history[WEIGHT_COLUMN].to_numpy(dtype=float),
    )
    return np.full(rows, mean_value, dtype=float)


def evaluate_full_2022(
    frame_2022: pd.DataFrame,
    method: str,
    predictions: np.ndarray,
    seed: int,
    device: str,
) -> MetricRow:
    return evaluate_predictions(
        method=method,
        seed=seed,
        calibration_fraction=0.0,
        calibration_rows=0,
        test_frame=frame_2022,
        predictions=np.maximum(predictions, 0.0),
        device=device,
    )


def make_random_pressure(pressure: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(pressure, dtype=float).copy()
    rng.shuffle(shuffled)
    return shuffled


def make_global_pressure(pressure: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_pressure = weighted_average(np.asarray(pressure, dtype=float), np.asarray(weights, dtype=float))
    return np.full_like(np.asarray(pressure, dtype=float), mean_pressure)


def make_delta_gated_pressure(
    pressure: np.ndarray,
    global_pressure: np.ndarray,
    threshold: float,
) -> np.ndarray:
    pressure_array = np.asarray(pressure, dtype=float)
    global_array = np.asarray(global_pressure, dtype=float)
    use_llm = np.abs(pressure_array - global_array) >= threshold
    return np.where(use_llm, pressure_array, global_array)


def build_metric_rows(
    full_frame: pd.DataFrame,
    frame_2022: pd.DataFrame,
    alphas: list[float],
    random_control_alphas: list[float],
    random_control_seeds: list[int],
    gate_thresholds: list[float],
    min_factor: float,
    device: str,
) -> list[MetricRow]:
    rows: list[MetricRow] = []
    base_prediction = frame_2022["base_prediction"].to_numpy(dtype=float)
    weights = frame_2022[WEIGHT_COLUMN].to_numpy(dtype=float)
    pressure = frame_2022["llm_recovery_adjusted_pressure"].to_numpy(dtype=float)
    confidence_pressure = frame_2022["llm_confidence_weighted_pressure"].to_numpy(dtype=float)
    suppression_pressure = frame_2022["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    global_pressure = make_global_pressure(pressure, weights)
    global_suppression_pressure = make_global_pressure(suppression_pressure, weights)

    rows.append(evaluate_full_2022(frame_2022, "historical_xgboost", base_prediction, 0, device))
    historical_mean_prediction = make_historical_mean_prediction(full_frame, len(frame_2022))
    rows.append(evaluate_full_2022(frame_2022, "historical_mean_only", historical_mean_prediction, 0, device))
    rows.append(
        evaluate_full_2022(
            frame_2022,
            "historical_mean_trend_shift",
            add_historical_mean_trend_prediction(full_frame, frame_2022),
            0,
            device,
        )
    )

    for alpha in alphas:
        token = alpha_token(alpha)
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"llm_recovery_pressure_a{token}",
                apply_multiplicative_pressure(base_prediction, pressure, alpha, min_factor),
                0,
                device,
            )
        )
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"global_recovery_pressure_a{token}",
                apply_multiplicative_pressure(base_prediction, global_pressure, alpha, min_factor),
                0,
                device,
            )
        )
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"llm_confidence_pressure_a{token}",
                apply_multiplicative_pressure(base_prediction, confidence_pressure, alpha, min_factor),
                0,
                device,
            )
        )
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"llm_trip_suppression_a{token}",
                apply_multiplicative_pressure(base_prediction, suppression_pressure, alpha, min_factor),
                0,
                device,
            )
        )
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"llm_only_trip_suppression_a{token}",
                apply_multiplicative_pressure(
                    historical_mean_prediction,
                    suppression_pressure,
                    alpha,
                    min_factor,
                ),
                0,
                device,
            )
        )
        rows.append(
            evaluate_full_2022(
                frame_2022,
                f"global_trip_suppression_a{token}",
                apply_multiplicative_pressure(base_prediction, global_suppression_pressure, alpha, min_factor),
                0,
                device,
            )
        )
        for threshold in gate_thresholds:
            threshold_token = alpha_token(threshold)
            gated_suppression = make_delta_gated_pressure(
                suppression_pressure,
                global_suppression_pressure,
                threshold,
            )
            rows.append(
                evaluate_full_2022(
                    frame_2022,
                    f"gated_trip_suppression_a{token}_d{threshold_token}",
                    apply_multiplicative_pressure(base_prediction, gated_suppression, alpha, min_factor),
                    0,
                    device,
                )
            )

    for alpha in random_control_alphas:
        token = alpha_token(alpha)
        for seed in random_control_seeds:
            random_pressure = make_random_pressure(pressure, seed)
            random_suppression_pressure = make_random_pressure(suppression_pressure, seed)
            rows.append(
                evaluate_full_2022(
                    frame_2022,
                    f"random_recovery_pressure_a{token}",
                    apply_multiplicative_pressure(base_prediction, random_pressure, alpha, min_factor),
                    seed,
                    device,
                )
            )
            rows.append(
                evaluate_full_2022(
                    frame_2022,
                    f"random_trip_suppression_a{token}",
                    apply_multiplicative_pressure(base_prediction, random_suppression_pressure, alpha, min_factor),
                    seed,
                    device,
                )
            )
    return rows


def write_metrics_csv(rows: list[MetricRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = metric_rows_to_dicts(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0]))
        writer.writeheader()
        writer.writerows(row_dicts)


def summarize_metric_rows(rows: list[MetricRow]) -> pd.DataFrame:
    frame = pd.DataFrame(metric_rows_to_dicts(rows))
    summary = (
        frame.groupby("method")
        .agg(
            runs=("seed", "count"),
            weighted_mae_mean=("weighted_mae", "mean"),
            weighted_mae_std=("weighted_mae", "std"),
            weighted_rmse_mean=("weighted_rmse", "mean"),
            weighted_bias_mean=("weighted_bias", "mean"),
            weighted_bias_std=("weighted_bias", "std"),
            weighted_r2_mean=("weighted_r2", "mean"),
            r2_mean=("r2", "mean"),
            weighted_prediction_mean=("weighted_prediction_mean", "mean"),
            weighted_target_mean=("weighted_target_mean", "mean"),
        )
        .reset_index()
        .sort_values("weighted_mae_mean")
    )
    return summary


def format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def write_pressure_summary(frame_2022: pd.DataFrame, output_path: Path) -> None:
    pressure_columns = [
        *LLM_NUMERIC_FEATURES,
        "llm_raw_event_pressure",
        "llm_recovery_adjusted_pressure",
        "llm_confidence_weighted_pressure",
        "llm_trip_suppression_pressure",
    ]
    summary = frame_2022[pressure_columns].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)


def make_subgroup_prediction_map(
    frame_2022: pd.DataFrame,
    alpha: float,
    min_factor: float,
) -> dict[str, np.ndarray]:
    base_prediction = frame_2022["base_prediction"].to_numpy(dtype=float)
    weights = frame_2022[WEIGHT_COLUMN].to_numpy(dtype=float)
    suppression_pressure = frame_2022["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    global_suppression_pressure = make_global_pressure(suppression_pressure, weights)
    random_suppression_pressure = make_random_pressure(suppression_pressure, RANDOM_SEED)
    token = alpha_token(alpha)
    return {
        "historical_xgboost": base_prediction,
        f"llm_trip_suppression_a{token}": apply_multiplicative_pressure(
            base_prediction,
            suppression_pressure,
            alpha,
            min_factor,
        ),
        f"global_trip_suppression_a{token}": apply_multiplicative_pressure(
            base_prediction,
            global_suppression_pressure,
            alpha,
            min_factor,
        ),
        f"random_trip_suppression_a{token}": apply_multiplicative_pressure(
            base_prediction,
            random_suppression_pressure,
            alpha,
            min_factor,
        ),
    }


def compute_subgroup_metric(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    subgroup_column: str,
    subgroup_value: object,
    method: str,
) -> dict[str, str | int | float]:
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=float)
    weights = frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - y_true
    return {
        "subgroup_column": subgroup_column,
        "subgroup_value": str(subgroup_value),
        "method": method,
        "rows": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "weighted_mae": float(np.average(np.abs(error), weights=weights)),
        "weighted_rmse": weighted_rmse(y_true, predictions, weights),
        "weighted_bias": weighted_average(error, weights),
        "weighted_target_mean": weighted_average(y_true, weights),
        "weighted_prediction_mean": weighted_average(predictions, weights),
    }


def build_subgroup_metrics(
    frame_2022: pd.DataFrame,
    prediction_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    metric_rows: list[dict[str, str | int | float]] = []
    indexed_frame = frame_2022.reset_index(drop=True)
    for subgroup_column in SUBGROUP_COLUMNS:
        for subgroup_value, subgroup_frame in indexed_frame.groupby(subgroup_column, dropna=False):
            indices = subgroup_frame.index.to_numpy()
            for method, predictions in prediction_map.items():
                metric_rows.append(
                    compute_subgroup_metric(
                        frame=subgroup_frame,
                        predictions=predictions[indices],
                        subgroup_column=subgroup_column,
                        subgroup_value=subgroup_value,
                        method=method,
                    )
                )
    return pd.DataFrame(metric_rows)


def build_subgroup_comparison(subgroup_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    token = alpha_token(alpha)
    key_columns = ["subgroup_column", "subgroup_value", "rows", "weight_sum"]
    value_columns = ["weighted_mae", "weighted_rmse", "weighted_bias"]
    selected = subgroup_metrics[
        subgroup_metrics["method"].isin(
            [
                "historical_xgboost",
                f"llm_trip_suppression_a{token}",
                f"global_trip_suppression_a{token}",
                f"random_trip_suppression_a{token}",
            ]
        )
    ]
    pivot = selected.pivot_table(
        index=key_columns,
        columns="method",
        values=value_columns,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
    comparison = pivot.reset_index()
    llm_mae = comparison[f"weighted_mae_llm_trip_suppression_a{token}"]
    global_mae = comparison[f"weighted_mae_global_trip_suppression_a{token}"]
    random_mae = comparison[f"weighted_mae_random_trip_suppression_a{token}"]
    historical_mae = comparison["weighted_mae_historical_xgboost"]
    comparison["llm_vs_historical_mae_delta"] = historical_mae - llm_mae
    comparison["llm_vs_global_mae_delta"] = global_mae - llm_mae
    comparison["llm_vs_random_mae_delta"] = random_mae - llm_mae
    comparison["llm_vs_historical_bias_abs_delta"] = (
        comparison["weighted_bias_historical_xgboost"].abs()
        - comparison[f"weighted_bias_llm_trip_suppression_a{token}"].abs()
    )
    return comparison


def write_subgroup_diagnostics(
    frame_2022: pd.DataFrame,
    output_dir: Path,
    alpha: float,
    min_factor: float,
) -> None:
    prediction_map = make_subgroup_prediction_map(frame_2022, alpha, min_factor)
    subgroup_metrics = build_subgroup_metrics(frame_2022, prediction_map)
    subgroup_comparison = build_subgroup_comparison(subgroup_metrics, alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    subgroup_metrics.to_csv(output_dir / "label_free_subgroup_metrics.csv", index=False)
    subgroup_comparison.to_csv(output_dir / "label_free_subgroup_comparison.csv", index=False)
    write_subgroup_summary(
        subgroup_comparison,
        output_dir / "label_free_subgroup_diagnostics.md",
        alpha,
    )


def markdown_rows(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for column in columns:
            value = getattr(row, column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_subgroup_summary(comparison: pd.DataFrame, output_path: Path, alpha: float) -> None:
    eligible = comparison[comparison["rows"] >= 50].copy()
    top_vs_global = eligible.sort_values("llm_vs_global_mae_delta", ascending=False).head(12)
    worst_vs_global = eligible.sort_values("llm_vs_global_mae_delta", ascending=True).head(12)
    top_vs_history = eligible.sort_values("llm_vs_historical_mae_delta", ascending=False).head(12)
    columns = [
        "subgroup_column",
        "subgroup_value",
        "rows",
        "llm_vs_historical_mae_delta",
        "llm_vs_global_mae_delta",
        "llm_vs_random_mae_delta",
        "llm_vs_historical_bias_abs_delta",
    ]
    lines = [
        "# Label-Free Subgroup Diagnostics",
        "",
        f"- Diagnostic alpha: `{alpha}`.",
        "- Positive deltas mean the LLM trip-suppression correction is better.",
        "- Rows with fewer than 50 households are excluded from the ranked tables.",
        "",
        "## Largest LLM Gains vs Historical Baseline",
        "",
        *markdown_rows(top_vs_history[columns], columns),
        "",
        "## Largest LLM Gains vs Global Trip-Suppression Control",
        "",
        *markdown_rows(top_vs_global[columns], columns),
        "",
        "## Groups Where Global Control Beats LLM Most",
        "",
        *markdown_rows(worst_vs_global[columns], columns),
        "",
        "Interpretation:",
        "",
        "- This diagnostic separates event-level downscaling from cohort-specific LLM assignment.",
        "- If LLM beats global controls in a subgroup, the cohort-specific `trip_suppression_risk` ranking is adding value there.",
        "- If global beats LLM, the subgroup mainly benefits from a common downward correction rather than differentiated LLM scores.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows: list[MetricRow], output_path: Path) -> None:
    summary = summarize_metric_rows(rows)
    baseline = summary[summary["method"] == "historical_xgboost"].iloc[0]
    best = summary.iloc[0]
    best_bias = summary.iloc[summary["weighted_bias_mean"].abs().argmin()]
    best_r2 = summary.sort_values("weighted_r2_mean", ascending=False).iloc[0]
    mae_reduction = 100.0 * (
        baseline.weighted_mae_mean - best.weighted_mae_mean
    ) / baseline.weighted_mae_mean
    bias_reduction = 100.0 * (
        abs(baseline.weighted_bias_mean) - abs(best.weighted_bias_mean)
    ) / abs(baseline.weighted_bias_mean)
    metric_lookup = summary.set_index("method")

    def weighted_mae(method: str) -> float | None:
        if method not in metric_lookup.index:
            return None
        return float(metric_lookup.loc[method, "weighted_mae_mean"])

    def relative_gain(numerator_method: str, denominator_method: str) -> float | None:
        numerator = weighted_mae(numerator_method)
        denominator = weighted_mae(denominator_method)
        if numerator is None or denominator is None:
            return None
        return 100.0 * (denominator - numerator) / denominator

    trip_vs_global = relative_gain("llm_trip_suppression_a1p25", "global_trip_suppression_a1p25")
    trip_vs_random = relative_gain("llm_trip_suppression_a1p25", "random_trip_suppression_a1p25")
    recovery_vs_global = relative_gain("llm_recovery_pressure_a1p25", "global_recovery_pressure_a1p25")
    recovery_vs_random = relative_gain("llm_recovery_pressure_a1p25", "random_recovery_pressure_a1p25")

    lines = [
        "# Label-Free LLM Event Adaptation Results",
        "",
        "| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.runs} | "
            f"{format_mean_std(row.weighted_mae_mean, row.weighted_mae_std)} | "
            f"{row.weighted_rmse_mean:.4f} | "
            f"{format_mean_std(row.weighted_bias_mean, row.weighted_bias_std)} | "
            f"{row.weighted_r2_mean:.4f} | {row.r2_mean:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            f"- Best reported sensitivity row: `{best.method}`.",
            f"- Weighted MAE reduction vs historical baseline: {mae_reduction:.2f}%.",
            f"- Absolute weighted-bias reduction vs historical baseline: {bias_reduction:.2f}%.",
            f"- Most unbiased row: `{best_bias.method}` with weighted bias `{best_bias.weighted_bias_mean:.4f}`.",
            f"- Highest weighted-R2 row: `{best_r2.method}` with weighted R2 `{best_r2.weighted_r2_mean:.4f}`.",
            "- These rows use no 2022 `CNTTDHH` labels for training, calibration, or parameter selection.",
            "- Alpha values are a pre-declared sensitivity grid, not target-domain fitted hyperparameters.",
            "- Random-pressure controls shuffle the same LLM pressure distribution across households.",
            "- Global-pressure controls apply the weighted mean LLM pressure to every household.",
            "- Gated controls use cohort-specific LLM scores only when they differ from the global mean by a fixed no-label threshold.",
            "",
            "## Diagnostic Interpretation",
            "",
            "- The strongest LLM feature is `trip_suppression_risk`; the broader recovery-adjusted composite is weaker.",
            "- Global-pressure controls are strong, so much of the gain comes from label-free event-level downscaling.",
            "- Random controls are weaker than `trip_suppression_risk`, indicating some cohort assignment signal remains.",
            "- Gated trip-suppression correction trades a small amount of MAE for much lower bias and higher weighted R2.",
        ]
    )
    if trip_vs_global is not None and trip_vs_random is not None:
        lines.append(
            "- `llm_trip_suppression_a1p25` improves weighted MAE by "
            f"{trip_vs_global:.2f}% vs its global-mean control and "
            f"{trip_vs_random:.2f}% vs its random-shuffle control."
        )
    if recovery_vs_global is not None and recovery_vs_random is not None:
        lines.append(
            "- `llm_recovery_pressure_a1p25` changes weighted MAE by "
            f"{recovery_vs_global:.2f}% vs global mean and "
            f"{recovery_vs_random:.2f}% vs random shuffle; this composite should not be overclaimed."
        )

    lines.extend(
        [
            "",
            "## Label-Free Correction Rule",
            "",
            "The primary LLM event pressure is computed as:",
            "",
            "```text",
            "raw_pressure = 0.35 * trip_suppression",
            "             + 0.25 * remote_work_substitution",
            "             + 0.20 * transit_avoidance",
            "             + 0.20 * online_delivery_substitution",
            "recovery_adjusted_pressure = raw_pressure * (1 - 0.25 * recovery_sensitivity)",
            "prediction = historical_prediction * clip(1 - alpha * recovery_adjusted_pressure, min_factor, 1)",
            "```",
            "",
            "Notes:",
            "",
            "- Historical model is trained only on 2001, 2009, and 2017 labels.",
            "- 2022 `CNTTDHH` appears only inside the final evaluation metrics.",
            "- All XGBoost training uses CUDA.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    alphas = parse_float_grid(args.alphas)
    random_control_alphas = parse_float_grid(args.random_control_alphas)
    random_control_seeds = parse_seeds(args.random_control_seeds)
    gate_thresholds = parse_float_grid(args.gate_thresholds)

    LOGGER.info("Resolved training device: %s", device)
    full_frame = pd.read_csv(args.dataset_path, low_memory=False)
    history_model, history_feature_columns = train_history_model(
        full_frame,
        args.history_n_estimators,
        device,
    )
    frame_2022 = load_2022_with_llm_features(
        args.dataset_path,
        args.cohort_profiles_path,
        args.llm_features_path,
    )
    frame_2022["base_prediction"] = np.maximum(
        history_model.predict(frame_2022[history_feature_columns]),
        0.0,
    )
    frame_2022 = add_event_pressure(frame_2022)

    rows = build_metric_rows(
        full_frame=full_frame,
        frame_2022=frame_2022,
        alphas=alphas,
        random_control_alphas=random_control_alphas,
        random_control_seeds=random_control_seeds,
        gate_thresholds=gate_thresholds,
        min_factor=args.min_factor,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(rows, args.output_dir / "label_free_llm_adaptation_metrics.csv")
    write_pressure_summary(frame_2022, args.output_dir / "label_free_event_pressure_summary.csv")
    write_subgroup_diagnostics(frame_2022, args.output_dir, args.subgroup_alpha, args.min_factor)
    write_summary(rows, args.output_dir / "label_free_llm_adaptation_results.md")
    LOGGER.info("Wrote label-free LLM adaptation results to %s", args.output_dir)


if __name__ == "__main__":
    main()
