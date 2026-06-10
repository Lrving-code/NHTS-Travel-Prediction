"""Apply 2022-style event correction to a pre-COVID target year as placebo."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from run_household_baseline import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    Experiment,
    create_pipeline,
    evaluate_predictions,
    resolve_device,
    set_random_seed,
)


LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
PLACEBO_EXPERIMENT = Experiment(
    name="pre_covid_2001_2009_to_2017_placebo",
    train_years=(2001, 2009),
    test_year=2017,
    include_survey_year=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument(
        "--llm-features-path",
        type=Path,
        default=Path(
            "outputs/llm_event_features/"
            "cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pre_covid_placebo_event_correction"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--alphas", default="0.25,0.50,0.75,1.00,1.25")
    parser.add_argument("--min-factor", type=float, default=0.05)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_alphas(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--alphas must include at least one value.")
    return values


def alpha_token(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def global_trip_suppression(llm_features_path: Path) -> float:
    features = pd.read_csv(llm_features_path, usecols=["trip_suppression_risk"])
    return float(features["trip_suppression_risk"].mean())


def apply_global_suppression(predictions: np.ndarray, pressure: float, alpha: float, min_factor: float) -> np.ndarray:
    factor = float(np.clip(1.0 - alpha * pressure, min_factor, 1.0))
    return np.maximum(predictions * factor, 0.0)


def train_placebo_model(frame: pd.DataFrame, n_estimators: int, device: str) -> tuple[pd.DataFrame, np.ndarray]:
    train = frame[frame["survey_year"].isin(PLACEBO_EXPERIMENT.train_years)].copy()
    test = frame[frame["survey_year"] == PLACEBO_EXPERIMENT.test_year].copy()
    model = create_pipeline(PLACEBO_EXPERIMENT.include_survey_year, n_estimators, device)
    numeric_cols = list(model.named_steps["preprocessor"].transformers[0][2])
    categorical_cols = list(model.named_steps["preprocessor"].transformers[1][2])
    feature_cols = numeric_cols + categorical_cols
    LOGGER.info("Training placebo routine model on %s rows; testing on %s rows.", len(train), len(test))
    model.fit(train[feature_cols], train[TARGET_COLUMN], model__sample_weight=train[WEIGHT_COLUMN])
    predictions = np.maximum(model.predict(test[feature_cols]), 0.0)
    return test, predictions


def metric_row(test: pd.DataFrame, predictions: np.ndarray, method: str, device: str) -> dict[str, float | int | str]:
    row = evaluate_predictions(PLACEBO_EXPERIMENT, test, test, predictions, device).__dict__
    row["method"] = method
    row.pop("experiment", None)
    row.pop("train_years", None)
    row.pop("test_year", None)
    row.pop("train_rows", None)
    return row


def write_report(metrics: pd.DataFrame, pressure: float, output_dir: Path) -> Path:
    path = output_dir / "pre_covid_placebo_event_correction_report.md"
    base = metrics[metrics["method"] == "routine_2001_2009_to_2017"].iloc[0]
    rows = metrics.sort_values("weighted_mae")
    lines = [
        "# Pre-COVID Placebo Event-Correction Report",
        "",
        "## Scope",
        "",
        "This placebo applies the 2022 LLM-derived global trip-suppression prior to a pre-COVID transfer task: train on 2001+2009 and predict 2017. It tests whether pandemic event correction behaves like a generic error fix or a context-specific shock adapter.",
        "",
        f"- Global 2022 trip-suppression pressure used in placebo: `{pressure:.4f}`.",
        "- 2017 labels are used only for evaluation.",
        "",
        "## Results",
        "",
        "| Method | Weighted MAE | Weighted bias | Weighted R2 | Prediction mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_mae:.4f} | {row.weighted_bias:+.4f} | "
            f"{row.r2:.4f} | {row.weighted_prediction_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The routine pre-COVID baseline has weighted MAE `{base.weighted_mae:.4f}` and "
                f"weighted bias `{base.weighted_bias:+.4f}`. Small global downshifts may compensate for this ordinary "
                "positive transfer bias, but stronger pandemic-style suppression should not be treated as a universal correction."
            ),
            "",
            "For the paper, this placebo should be reported as a guardrail rather than a proof of causality: event priors need context and strength discipline, and 2022 target labels must remain outside the adaptation step.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset_path, low_memory=False)
    test, base_predictions = train_placebo_model(frame, args.n_estimators, device)
    pressure = global_trip_suppression(args.llm_features_path)
    rows = [metric_row(test, base_predictions, "routine_2001_2009_to_2017", device)]
    for alpha in parse_alphas(args.alphas):
        predictions = apply_global_suppression(base_predictions, pressure, alpha, args.min_factor)
        rows.append(metric_row(test, predictions, f"placebo_global_2022_suppression_a{alpha_token(alpha)}", device))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "pre_covid_placebo_event_correction_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    report_path = write_report(metrics, pressure, args.output_dir)
    LOGGER.info("Wrote placebo report: %s", report_path)


if __name__ == "__main__":
    main()
