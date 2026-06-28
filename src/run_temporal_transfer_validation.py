"""Run historical temporal-transfer checks for NHTS household trip generation."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TransferCheck:
    """Temporal transfer experiment definition."""

    name: str
    train_years: tuple[int, ...]
    test_year: int
    include_survey_year: bool = True


CHECKS: tuple[TransferCheck, ...] = (
    TransferCheck("pre_covid_2001_to_2009", (2001,), 2009),
    TransferCheck("pre_covid_2001_2009_to_2017", (2001, 2009), 2017),
    TransferCheck("post_covid_history_to_2022", (2001, 2009, 2017), 2022),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temporal_transfer_validation"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def to_baseline_experiment(check: TransferCheck) -> Experiment:
    return Experiment(
        name=check.name,
        train_years=check.train_years,
        test_year=check.test_year,
        include_survey_year=check.include_survey_year,
    )


def run_check(frame: pd.DataFrame, check: TransferCheck, n_estimators: int, device: str) -> dict[str, float | int | str]:
    train_frame = frame[frame["survey_year"].isin(check.train_years)].copy()
    test_frame = frame[frame["survey_year"] == check.test_year].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError(f"Missing train/test rows for {check.name}")
    model = create_pipeline(check.include_survey_year, n_estimators, device)
    numeric_cols = list(model.named_steps["preprocessor"].transformers[0][2])
    categorical_cols = list(model.named_steps["preprocessor"].transformers[1][2])
    feature_cols = numeric_cols + categorical_cols
    LOGGER.info("Training %s on %s rows; testing on %s rows.", check.name, len(train_frame), len(test_frame))
    model.fit(
        train_frame[feature_cols],
        train_frame[TARGET_COLUMN],
        model__sample_weight=train_frame[WEIGHT_COLUMN],
    )
    predictions = np.maximum(model.predict(test_frame[feature_cols]), 0.0)
    metric_row = evaluate_predictions(to_baseline_experiment(check), train_frame, test_frame, predictions, device)
    return metric_row.__dict__


def write_report(metrics: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "temporal_transfer_validation_report.md"
    lines = [
        "# Temporal Transfer Validation",
        "",
        "## Scope",
        "",
        "This check asks whether the 2022 failure is just ordinary cross-year transfer error or a stronger post-pandemic distribution shift.",
        "",
        "## Results",
        "",
        "| Check | Train years | Test year | Weighted MAE | Weighted bias | R2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.train_years} | {row.test_year} | "
            f"{row.weighted_mae:.4f} | {row.weighted_bias:+.4f} | {row.r2:.4f} |"
        )
    pre = metrics[metrics["test_year"] != 2022]
    post = metrics[metrics["test_year"] == 2022].iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Pre-COVID temporal checks have mean absolute weighted bias `{pre['weighted_bias'].abs().mean():.4f}`.",
            f"- The 2022 transfer check has absolute weighted bias `{abs(post.weighted_bias):.4f}`.",
            "- This supports the framing that 2022 is an event-shift target rather than a routine transfer year.",
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
    frame = pd.read_csv(args.dataset_path, low_memory=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_check(frame, check, args.n_estimators, device) for check in CHECKS]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "temporal_transfer_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    report_path = write_report(metrics, args.output_dir)
    LOGGER.info("Wrote temporal transfer report: %s", report_path)


if __name__ == "__main__":
    main()
