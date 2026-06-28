"""Run irrelevant pseudo-event placebo controls for the event adapter."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "models" / "final_label_free_2022" / "final_2022_predictions.csv"
HOUSEHOLD_PATH = PROJECT_ROOT / "data" / "processed" / "household_harmonized.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "irrelevant_pseudo_event_placebo"

TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
BASE_PREDICTION = "base_prediction"
ACTUAL_PRESSURE = "llm_trip_suppression_pressure"
PRIMARY_PREDICTION = "prediction_gated_trip_suppression_a1_d0p15"
LLM_ONLY_PREDICTION = "prediction_llm_only_trip_suppression_a1p25"
RANDOM_SEED = 20260610

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PressureSpec:
    """A deterministic pseudo-event pressure specification."""

    name: str
    description: str
    score_column: str


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_average(values, weights)
    return float(np.sqrt(np.average(np.square(values - mean), weights=weights)))


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
        "weighted_prediction_mean": weighted_average(y_pred, weights),
        "exact_rounded_accuracy": float(np.mean(np.rint(y_pred) == np.rint(y_true))),
        "within_1_trip": float(np.mean(abs_error <= 1.0)),
        "within_2_trips": float(np.mean(abs_error <= 2.0)),
        "within_3_trips": float(np.mean(abs_error <= 3.0)),
        "weighted_within_1_trip": weighted_average((abs_error <= 1.0).astype(float), weights),
        "weighted_within_2_trips": weighted_average((abs_error <= 2.0).astype(float), weights),
        "weighted_within_3_trips": weighted_average((abs_error <= 3.0).astype(float), weights),
    }


def apply_pressure(base_prediction: np.ndarray, pressure: np.ndarray, alpha: float, min_factor: float = 0.05) -> np.ndarray:
    factor = np.clip(1.0 - alpha * pressure, min_factor, 1.0)
    return np.maximum(base_prediction * factor, 0.0)


def global_pressure(pressure: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.full_like(pressure, weighted_average(pressure, weights), dtype=float)


def gated_pressure(pressure: np.ndarray, weights: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    global_value = weighted_average(pressure, weights)
    global_values = np.full_like(pressure, global_value, dtype=float)
    return np.where(np.abs(pressure - global_value) >= threshold, pressure, global_values)


def stable_uniform_from_id(value: object, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(16**16 - 1)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(values.median())


def minmax(values: pd.Series) -> pd.Series:
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def rank_match_distribution(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    matched = np.empty_like(reference, dtype=float)
    matched[order] = np.sort(reference)
    return matched


def mean_adjust_pressure(pressure: np.ndarray, target_mean: float, weights: np.ndarray) -> np.ndarray:
    adjusted = pressure.astype(float).copy()
    for _ in range(8):
        adjusted = np.clip(adjusted + target_mean - weighted_average(adjusted, weights), 0.02, 0.95)
    return adjusted


def load_analysis_frame() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"HOUSEID": str})
    needed = [
        "HOUSEID",
        "survey_year",
        "HHFAMINC",
        "HOMEOWN",
        "HHSIZE",
        "URBAN",
        "URBRUR",
        "TRAVDAY",
        "travel_month",
    ]
    households = pd.read_csv(HOUSEHOLD_PATH, usecols=needed, dtype={"HOUSEID": str}, low_memory=False)
    households = households.loc[households["survey_year"] == 2022].drop_duplicates("HOUSEID")
    merged = predictions.merge(households.drop(columns=["survey_year"]), on="HOUSEID", how="left", validate="one_to_one")
    if merged[["HHFAMINC", "HOMEOWN", "HHSIZE", "URBAN", "URBRUR", "travel_month"]].isna().all(axis=None):
        raise ValueError("Merged household covariates are empty; cannot build pseudo-event controls.")
    return merged


def add_pseudo_event_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    income = minmax(numeric_series(output, "HHFAMINC"))
    household_size = minmax(numeric_series(output, "HHSIZE"))
    homeownership = (numeric_series(output, "HOMEOWN") == 1).astype(float)
    urban = (numeric_series(output, "URBAN") == 1).astype(float)
    month = numeric_series(output, "travel_month").clip(1, 12)
    month_wave = (np.sin((month - 1.0) / 12.0 * 2.0 * np.pi) + 1.0) / 2.0
    weekday_wave = (np.cos((numeric_series(output, "TRAVDAY") - 1.0) / 7.0 * 2.0 * np.pi) + 1.0) / 2.0

    hash_media = output["HOUSEID"].map(lambda value: stable_uniform_from_id(value, "media-upgrade"))
    hash_calendar = output["HOUSEID"].map(lambda value: stable_uniform_from_id(value, "calendar-app"))

    output["score_pseudo_media_upgrade"] = (
        0.30 * income
        + 0.20 * homeownership
        + 0.15 * household_size
        + 0.10 * urban
        + 0.10 * month_wave
        + 0.15 * hash_media
    )
    output["score_pseudo_calendar_app"] = 0.50 * month_wave + 0.25 * weekday_wave + 0.25 * hash_calendar
    output["score_pseudo_random_hash"] = output["HOUSEID"].map(lambda value: stable_uniform_from_id(value, "unrelated"))
    return output


def pressure_specs() -> tuple[PressureSpec, ...]:
    return (
        PressureSpec(
            name="pseudo_media_upgrade",
            description=(
                "A non-travel pseudo-event about household digital media upgrade adoption, built from household "
                "demographics and deterministic ID noise."
            ),
            score_column="score_pseudo_media_upgrade",
        ),
        PressureSpec(
            name="pseudo_calendar_app",
            description=(
                "A non-travel pseudo-event about calendar-app adoption, built from travel-day timing waves and "
                "deterministic ID noise."
            ),
            score_column="score_pseudo_calendar_app",
        ),
        PressureSpec(
            name="pseudo_random_hash",
            description="A deterministic non-mechanistic pressure ranking from hashed household IDs.",
            score_column="score_pseudo_random_hash",
        ),
    )


def build_metric_row(
    method: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    pressure: np.ndarray | None = None,
    description: str = "",
    alpha: float | None = None,
    gate_threshold: float | None = None,
) -> dict[str, float | str | None]:
    row: dict[str, float | str | None] = {
        "method": method,
        "family": family,
        "description": description,
        "alpha": alpha,
        "gate_threshold": gate_threshold,
        **evaluate(y_true, y_pred, weights),
    }
    if pressure is None:
        row["pressure_weighted_mean"] = None
        row["pressure_weighted_std"] = None
    else:
        row["pressure_weighted_mean"] = weighted_average(pressure, weights)
        row["pressure_weighted_std"] = weighted_std(pressure, weights)
    return row


def build_placebo_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = frame[TARGET].to_numpy(dtype=float)
    weights = frame[WEIGHT].to_numpy(dtype=float)
    base = frame[BASE_PREDICTION].to_numpy(dtype=float)
    actual_pressure = frame[ACTUAL_PRESSURE].to_numpy(dtype=float)
    target_pressure_mean = weighted_average(actual_pressure, weights)

    rows: list[dict[str, float | str | None]] = [
        build_metric_row(
            "historical_xgboost",
            "traditional_baseline",
            y_true,
            base,
            weights,
            description="Historical routine model without post-pandemic event adaptation.",
        ),
        build_metric_row(
            "global_actual_event_a1",
            "global_event_prior",
            y_true,
            apply_pressure(base, global_pressure(actual_pressure, weights), alpha=1.0),
            weights,
            pressure=global_pressure(actual_pressure, weights),
            description="Global weighted mean of the actual LLM trip-suppression pressure.",
            alpha=1.0,
        ),
        build_metric_row(
            "primary_gated_event_a1_d0p15",
            "main_method",
            y_true,
            frame[PRIMARY_PREDICTION].to_numpy(dtype=float),
            weights,
            pressure=gated_pressure(actual_pressure, weights, threshold=0.15),
            description="Primary fixed no-label gated event adapter.",
            alpha=1.0,
            gate_threshold=0.15,
        ),
        build_metric_row(
            "llm_only_pressure_a1p25",
            "llm_only_control",
            y_true,
            frame[LLM_ONLY_PREDICTION].to_numpy(dtype=float),
            weights,
            pressure=actual_pressure,
            description="LLM pressure without the routine household baseline.",
            alpha=1.25,
        ),
    ]

    pressure_rows: list[dict[str, float | str]] = []
    for spec in pressure_specs():
        raw_scores = frame[spec.score_column].to_numpy(dtype=float)
        pseudo_pressure = rank_match_distribution(raw_scores, actual_pressure)
        pseudo_pressure = mean_adjust_pressure(pseudo_pressure, target_pressure_mean, weights)
        pseudo_global = global_pressure(pseudo_pressure, weights)
        pseudo_gated = gated_pressure(pseudo_pressure, weights, threshold=0.15)

        rows.extend(
            [
                build_metric_row(
                    f"{spec.name}_global_same_mean_a1",
                    "pseudo_event_global",
                    y_true,
                    apply_pressure(base, pseudo_global, alpha=1.0),
                    weights,
                    pressure=pseudo_global,
                    description=f"{spec.description} Uses only the placebo weighted mean.",
                    alpha=1.0,
                ),
                build_metric_row(
                    f"{spec.name}_rank_matched_a1",
                    "pseudo_event_ranked",
                    y_true,
                    apply_pressure(base, pseudo_pressure, alpha=1.0),
                    weights,
                    pressure=pseudo_pressure,
                    description=f"{spec.description} Distribution-matched to actual LLM pressure.",
                    alpha=1.0,
                ),
                build_metric_row(
                    f"{spec.name}_rank_matched_gated_a1_d0p15",
                    "pseudo_event_gated",
                    y_true,
                    apply_pressure(base, pseudo_gated, alpha=1.0),
                    weights,
                    pressure=pseudo_gated,
                    description=f"{spec.description} Distribution-matched and passed through the same gate.",
                    alpha=1.0,
                    gate_threshold=0.15,
                ),
            ]
        )
        pressure_rows.append(
            {
                "pseudo_event": spec.name,
                "score_column": spec.score_column,
                "description": spec.description,
                "pressure_weighted_mean": f"{weighted_average(pseudo_pressure, weights):.8f}",
                "pressure_weighted_std": f"{weighted_std(pseudo_pressure, weights):.8f}",
                "actual_pressure_weighted_mean": f"{weighted_average(actual_pressure, weights):.8f}",
                "actual_pressure_weighted_std": f"{weighted_std(actual_pressure, weights):.8f}",
            }
        )

    metrics = pd.DataFrame(rows)
    global_mae = float(metrics.loc[metrics["method"] == "global_actual_event_a1", "weighted_mae"].iloc[0])
    primary_mae = float(metrics.loc[metrics["method"] == "primary_gated_event_a1_d0p15", "weighted_mae"].iloc[0])
    metrics["mae_delta_vs_global_actual_event_a1"] = metrics["weighted_mae"] - global_mae
    metrics["mae_delta_vs_primary"] = metrics["weighted_mae"] - primary_mae
    return metrics, pd.DataFrame(pressure_rows)


def plot_placebo_metrics(metrics: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "irrelevant_pseudo_event_placebo_mae.png"
    selected = metrics.loc[
        metrics["method"].isin(
            [
                "historical_xgboost",
                "global_actual_event_a1",
                "primary_gated_event_a1_d0p15",
                "pseudo_media_upgrade_rank_matched_a1",
                "pseudo_calendar_app_rank_matched_a1",
                "pseudo_random_hash_rank_matched_a1",
            ]
        )
    ].copy()
    labels = {
        "historical_xgboost": "Historical\nXGBoost",
        "global_actual_event_a1": "Global\nevent prior",
        "primary_gated_event_a1_d0p15": "Primary\ngated event",
        "pseudo_media_upgrade_rank_matched_a1": "Pseudo\nmedia",
        "pseudo_calendar_app_rank_matched_a1": "Pseudo\ncalendar",
        "pseudo_random_hash_rank_matched_a1": "Pseudo\nhash",
    }
    colors = {
        "historical_xgboost": "#94a3b8",
        "global_actual_event_a1": "#38bdf8",
        "primary_gated_event_a1_d0p15": "#16a34a",
        "pseudo_media_upgrade_rank_matched_a1": "#f59e0b",
        "pseudo_calendar_app_rank_matched_a1": "#f97316",
        "pseudo_random_hash_rank_matched_a1": "#dc2626",
    }
    selected["label"] = selected["method"].map(labels)
    selected["color"] = selected["method"].map(colors)

    plt.figure(figsize=(8.8, 4.8))
    bars = plt.bar(selected["label"], selected["weighted_mae"], color=selected["color"], edgecolor="#334155")
    plt.ylabel("Weighted MAE")
    plt.title("Irrelevant pseudo-event priors do not replace mechanism-aligned event adaptation", fontsize=12)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.035, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def write_report(metrics: pd.DataFrame, pressure_summary: pd.DataFrame, figure_path: Path) -> Path:
    path = OUTPUT_DIR / "irrelevant_pseudo_event_placebo_report.md"
    baseline = metrics.loc[metrics["method"] == "historical_xgboost"].iloc[0]
    global_event = metrics.loc[metrics["method"] == "global_actual_event_a1"].iloc[0]
    primary = metrics.loc[metrics["method"] == "primary_gated_event_a1_d0p15"].iloc[0]
    pseudo_ranked = metrics.loc[metrics["family"] == "pseudo_event_ranked"].copy()
    pseudo_gated = metrics.loc[metrics["family"] == "pseudo_event_gated"].copy()
    best_pseudo = pseudo_ranked.sort_values("weighted_mae").iloc[0]
    best_pseudo_gated = pseudo_gated.sort_values("weighted_mae").iloc[0]

    lines = [
        "# Irrelevant Pseudo-Event Placebo Control",
        "",
        "## Purpose",
        "",
        "This is a negative-control experiment. It asks whether the 2022 improvement can be reproduced by any "
        "cohort-level score that has the same marginal pressure distribution as the LLM event prior, even when "
        "the score is unrelated to post-pandemic travel mechanisms.",
        "",
        "The answer should not be read as causal proof. It is a guardrail against the weaker explanation that the "
        "method works merely because it applies an arbitrary household-level shrinkage score.",
        "",
        "## Design",
        "",
        "- Target and evaluation year: 2022 household `CNTTDHH`.",
        "- Base model: historical XGBoost routine prediction trained without 2022 labels.",
        "- True event prior: LLM trip-suppression pressure, used only through fixed no-label correction rules.",
        "- Pseudo-event priors: deterministic non-travel rankings for media-upgrade adoption, calendar-app adoption, "
        "and household-ID hash noise.",
        "- Distribution matching: every pseudo-event pressure is rank-matched to the actual LLM pressure distribution "
        "and mean-adjusted to the same weighted pressure mean.",
        "",
        "## Key Results",
        "",
        f"- Historical XGBoost wMAE: `{baseline.weighted_mae:.4f}`.",
        f"- Global actual event prior wMAE: `{global_event.weighted_mae:.4f}`.",
        f"- Primary gated event adapter wMAE: `{primary.weighted_mae:.4f}`.",
        f"- Best irrelevant ranked pseudo-event wMAE: `{best_pseudo.weighted_mae:.4f}` "
        f"(`{best_pseudo.method}`).",
        f"- Best irrelevant gated pseudo-event wMAE: `{best_pseudo_gated.weighted_mae:.4f}` "
        f"(`{best_pseudo_gated.method}`).",
        f"- Best irrelevant ranked pseudo-event is `{best_pseudo.weighted_mae - primary.weighted_mae:+.4f}` wMAE "
        "relative to the primary method.",
        "",
        "## Interpretation",
        "",
        "A global event downshift is a strong baseline because 2022 has a broad post-pandemic travel suppression. "
        "However, irrelevant cohort rankings do not replace the mechanism-aligned event prior. The defensible claim "
        "is therefore not that any LLM-generated score works, but that a scoped event prior can adapt a routine "
        "household model under a known societal shock while preserving label-free calibration discipline.",
        "",
        "## Pressure Matching Check",
        "",
        "| Pseudo-event | Weighted mean | Weighted std | Actual weighted mean | Actual weighted std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pressure_summary.itertuples(index=False):
        lines.append(
            f"| {row.pseudo_event} | {row.pressure_weighted_mean} | {row.pressure_weighted_std} | "
            f"{row.actual_pressure_weighted_mean} | {row.actual_pressure_weighted_std} |"
        )

    lines.extend(
        [
            "",
            "## Full Metric Table",
            "",
            "| Method | Family | wMAE | wRMSE | wBias | wR2 | Delta vs primary |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.sort_values(["family", "weighted_mae"]).itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.family} | {row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} | {row.mae_delta_vs_primary:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Artifact",
            "",
            f"- Figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
            f"- Metrics: `{(OUTPUT_DIR / 'irrelevant_pseudo_event_placebo_metrics.csv').relative_to(PROJECT_ROOT)}`",
            f"- Pressure summary: `{(OUTPUT_DIR / 'irrelevant_pseudo_event_pressure_summary.csv').relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_pseudo_event_scores(load_analysis_frame())
    metrics, pressure_summary = build_placebo_metrics(frame)
    metrics.to_csv(OUTPUT_DIR / "irrelevant_pseudo_event_placebo_metrics.csv", index=False)
    pressure_summary.to_csv(OUTPUT_DIR / "irrelevant_pseudo_event_pressure_summary.csv", index=False)
    figure_path = plot_placebo_metrics(metrics)
    report_path = write_report(metrics, pressure_summary, figure_path)
    LOGGER.info("Wrote pseudo-event placebo metrics: %s", OUTPUT_DIR / "irrelevant_pseudo_event_placebo_metrics.csv")
    LOGGER.info("Wrote pseudo-event placebo report: %s", report_path)


if __name__ == "__main__":
    main()
