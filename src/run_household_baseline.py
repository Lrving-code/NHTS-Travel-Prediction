"""Run household-level NHTS baseline transfer experiments."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


LOGGER = logging.getLogger(__name__)
RANDOM_SEED = 42
TARGET_COLUMN = "CNTTDHH"
WEIGHT_COLUMN = "WTHHFIN"
ID_COLUMN = "HOUSEID"

BASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "DRVRCNT",
    "HHSIZE",
    "HHVEHCNT",
    "NUMADLT",
    "RESP_CNT",
    "WRKCOUNT",
    "travel_month",
)
BASE_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "CDIVMSAR",
    "CENSUS_D",
    "CENSUS_R",
    "HBHTNRNT",
    "HBHUR",
    "HBPPOPDN",
    "HHFAMINC",
    "HOMEOWN",
    "HTEEMPDN",
    "HTHTNRNT",
    "HTPPOPDN",
    "LIF_CYC",
    "MSACAT",
    "MSASIZE",
    "RAIL",
    "TRAVDAY",
    "URBAN",
    "URBRUR",
)


@dataclass(frozen=True)
class Experiment:
    """Baseline experiment definition."""

    name: str
    train_years: tuple[int, ...]
    test_year: int
    include_survey_year: bool


@dataclass(frozen=True)
class MetricRow:
    """Evaluation metrics for one experiment."""

    experiment: str
    train_years: str
    test_year: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    bias: float
    r2: float
    weighted_mae: float
    weighted_rmse: float
    weighted_bias: float
    weighted_target_mean: float
    weighted_prediction_mean: float
    device: str


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        name="direct_2017_to_2022",
        train_years=(2017,),
        test_year=2022,
        include_survey_year=False,
    ),
    Experiment(
        name="pooled_history_to_2022",
        train_years=(2001, 2009, 2017),
        test_year=2022,
        include_survey_year=False,
    ),
    Experiment(
        name="pooled_history_with_year_to_2022",
        train_years=(2001, 2009, 2017),
        test_year=2022,
        include_survey_year=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/household_harmonized.csv"),
        help="Harmonized household dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/household_baseline"),
        help="Directory where baseline results will be written.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of XGBoost boosting rounds.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. Auto uses CUDA when an NVIDIA GPU is visible.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        if not has_nvidia_gpu():
            raise RuntimeError("No NVIDIA GPU detected. CUDA training was requested by project policy.")
        return "cuda"
    return requested_device


def get_feature_columns(include_survey_year: bool) -> tuple[list[str], list[str]]:
    numeric_features = list(BASE_NUMERIC_FEATURES)
    categorical_features = list(BASE_CATEGORICAL_FEATURES)
    if include_survey_year:
        numeric_features.append("survey_year")
    return numeric_features, categorical_features


def create_pipeline(include_survey_year: bool, n_estimators: int, device: str) -> Pipeline:
    numeric_features, categorical_features = get_feature_columns(include_survey_year)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        device=device,
        random_state=RANDOM_SEED,
        n_jobs=4,
        eval_metric="rmse",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    squared_error = np.square(y_pred - y_true)
    return float(np.sqrt(np.average(squared_error, weights=weights)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def evaluate_predictions(
    experiment: Experiment,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    device: str,
) -> MetricRow:
    y_true = test_frame[TARGET_COLUMN].to_numpy(dtype=float)
    weights = test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - y_true

    return MetricRow(
        experiment=experiment.name,
        train_years="+".join(str(year) for year in experiment.train_years),
        test_year=experiment.test_year,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=rmse(y_true, predictions),
        bias=float(np.mean(error)),
        r2=float(r2_score(y_true, predictions)),
        weighted_mae=float(mean_absolute_error(y_true, predictions, sample_weight=weights)),
        weighted_rmse=weighted_rmse(y_true, predictions, weights),
        weighted_bias=weighted_average(error, weights),
        weighted_target_mean=weighted_average(y_true, weights),
        weighted_prediction_mean=weighted_average(predictions, weights),
        device=device,
    )


def run_experiment(
    frame: pd.DataFrame,
    experiment: Experiment,
    n_estimators: int,
    device: str,
) -> MetricRow:
    train_frame = frame[frame["survey_year"].isin(experiment.train_years)].copy()
    test_frame = frame[frame["survey_year"] == experiment.test_year].copy()
    numeric_features, categorical_features = get_feature_columns(experiment.include_survey_year)
    feature_columns = numeric_features + categorical_features

    LOGGER.info(
        "Running %s with %s train rows and %s test rows.",
        experiment.name,
        len(train_frame),
        len(test_frame),
    )
    pipeline = create_pipeline(experiment.include_survey_year, n_estimators, device)
    pipeline.fit(
        train_frame[feature_columns],
        train_frame[TARGET_COLUMN],
        model__sample_weight=train_frame[WEIGHT_COLUMN],
    )
    actual_device = device
    predictions = pipeline.predict(test_frame[feature_columns])
    predictions = np.maximum(predictions, 0.0)
    return evaluate_predictions(experiment, train_frame, test_frame, predictions, actual_device)


def metric_rows_to_dicts(rows: list[MetricRow]) -> list[dict[str, str | int | float]]:
    return [
        {
            "experiment": row.experiment,
            "train_years": row.train_years,
            "test_year": row.test_year,
            "train_rows": row.train_rows,
            "test_rows": row.test_rows,
            "mae": row.mae,
            "rmse": row.rmse,
            "bias": row.bias,
            "r2": row.r2,
            "weighted_mae": row.weighted_mae,
            "weighted_rmse": row.weighted_rmse,
            "weighted_bias": row.weighted_bias,
            "weighted_target_mean": row.weighted_target_mean,
            "weighted_prediction_mean": row.weighted_prediction_mean,
            "device": row.device,
        }
        for row in rows
    ]


def write_metrics_csv(rows: list[MetricRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metric_rows_to_dicts(rows)[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows_to_dicts(rows))


def write_summary(rows: list[MetricRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Household Baseline Results",
        "",
        "| Experiment | Train Years | Test Year | Device | Weighted MAE | Weighted RMSE | Weighted Bias | R2 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.experiment} | {row.train_years} | {row.test_year} | {row.device} | "
            f"{row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:.4f} | {row.r2:.4f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- Target: `CNTTDHH`.")
    lines.append("- Training uses NHTS household final weights `WTHHFIN`.")
    lines.append("- Metrics include both unweighted values in CSV and weighted values in this summary.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    LOGGER.info("Requested device: %s; resolved training device: %s", args.device, device)

    frame = pd.read_csv(args.dataset_path, low_memory=False)
    rows = [
        run_experiment(frame, experiment, args.n_estimators, device)
        for experiment in EXPERIMENTS
    ]
    write_metrics_csv(rows, args.output_dir / "household_baseline_metrics.csv")
    write_summary(rows, args.output_dir / "household_baseline_results.md")

    LOGGER.info("Wrote household baseline results to %s", args.output_dir)


if __name__ == "__main__":
    main()
