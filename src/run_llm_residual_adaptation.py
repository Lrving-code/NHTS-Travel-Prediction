"""Run LLM-guided residual adaptation experiments for 2022 NHTS."""

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

from build_llm_household_profiles import DEFAULT_COHORT_COLUMNS, bin_count
from run_household_baseline import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    ID_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)


LOGGER = logging.getLogger(__name__)
LLM_NUMERIC_FEATURES: tuple[str, ...] = (
    "trip_suppression_risk",
    "remote_work_substitution_likelihood",
    "transit_avoidance_likelihood",
    "online_delivery_substitution_likelihood",
    "post_pandemic_recovery_sensitivity",
    "confidence",
)


@dataclass(frozen=True)
class MetricRow:
    """Evaluation metrics for one residual-adaptation method."""

    method: str
    seed: int
    calibration_fraction: float
    calibration_rows: int
    test_rows: int
    mae: float
    rmse: float
    bias: float
    r2: float
    weighted_mae: float
    weighted_rmse: float
    weighted_bias: float
    weighted_r2: float
    weighted_target_mean: float
    weighted_prediction_mean: float
    device: str


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_residual_adaptation"))
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--history-n-estimators", type=int, default=200)
    parser.add_argument("--residual-n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
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
            raise RuntimeError("No NVIDIA GPU detected. CUDA training is required for this project.")
        return "cuda"
    return requested_device


def parse_seeds(raw_seeds: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed.")
    return seeds


def create_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    device: str,
    random_state: int,
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
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
        random_state=random_state,
        n_jobs=4,
        eval_metric="rmse",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(y_pred - y_true), weights=weights)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weighted_mean = weighted_average(y_true, weights)
    numerator = np.average(np.square(y_true - y_pred), weights=weights)
    denominator = np.average(np.square(y_true - weighted_mean), weights=weights)
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def evaluate_predictions(
    method: str,
    seed: int,
    calibration_fraction: float,
    calibration_rows: int,
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    device: str,
) -> MetricRow:
    y_true = test_frame[TARGET_COLUMN].to_numpy(dtype=float)
    weights = test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - y_true
    return MetricRow(
        method=method,
        seed=seed,
        calibration_fraction=calibration_fraction,
        calibration_rows=calibration_rows,
        test_rows=len(test_frame),
        mae=float(mean_absolute_error(y_true, predictions)),
        rmse=rmse(y_true, predictions),
        bias=float(np.mean(error)),
        r2=float(r2_score(y_true, predictions)),
        weighted_mae=float(mean_absolute_error(y_true, predictions, sample_weight=weights)),
        weighted_rmse=weighted_rmse(y_true, predictions, weights),
        weighted_bias=weighted_average(error, weights),
        weighted_r2=weighted_r2(y_true, predictions, weights),
        weighted_target_mean=weighted_average(y_true, weights),
        weighted_prediction_mean=weighted_average(predictions, weights),
        device=device,
    )


def add_cohort_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["HHVEHCNT_BIN"] = keyed["HHVEHCNT"].map(bin_count)
    keyed["WRKCOUNT_BIN"] = keyed["WRKCOUNT"].map(bin_count)
    for column in DEFAULT_COHORT_COLUMNS:
        keyed[column] = keyed[column].astype(str)
    return keyed


def load_2022_with_llm_features(
    dataset_path: Path,
    cohort_profiles_path: Path,
    llm_features_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path, low_memory=False)
    household_2022 = frame[frame["survey_year"] == 2022].copy().reset_index(drop=True)
    household_2022 = add_cohort_keys(household_2022)

    profile_columns = ["cohort_id", *DEFAULT_COHORT_COLUMNS]
    profiles = pd.read_csv(cohort_profiles_path, usecols=profile_columns, dtype=str)
    features = pd.read_csv(llm_features_path)
    cohort_features = profiles.merge(features, on="cohort_id", how="inner", validate="one_to_one")

    merged = household_2022.merge(
        cohort_features,
        on=list(DEFAULT_COHORT_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    if merged["cohort_id"].isna().any():
        missing_count = int(merged["cohort_id"].isna().sum())
        raise ValueError(f"Missing LLM cohort features for {missing_count} 2022 household rows.")
    return merged


def train_history_model(
    frame: pd.DataFrame,
    n_estimators: int,
    device: str,
) -> tuple[Pipeline, list[str]]:
    train_frame = frame[frame["survey_year"].isin((2001, 2009, 2017))].copy()
    numeric_features = [*BASE_NUMERIC_FEATURES, "survey_year"]
    categorical_features = list(BASE_CATEGORICAL_FEATURES)
    feature_columns = numeric_features + categorical_features
    model = create_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=n_estimators,
        device=device,
        random_state=RANDOM_SEED,
    )
    LOGGER.info("Training historical routine model on %s rows.", len(train_frame))
    model.fit(
        train_frame[feature_columns],
        train_frame[TARGET_COLUMN],
        model__sample_weight=train_frame[WEIGHT_COLUMN],
    )
    return model, feature_columns


def split_calibration_test(frame: pd.DataFrame, calibration_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be in (0, 1).")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    calibration_count = max(1, int(round(len(frame) * calibration_fraction)))
    calibration_indices = indices[:calibration_count]
    test_indices = indices[calibration_count:]
    return frame.iloc[calibration_indices].copy(), frame.iloc[test_indices].copy()


def fit_residual_model(
    calibration_frame: pd.DataFrame,
    residual_column: str,
    numeric_features: list[str],
    categorical_features: list[str],
    n_estimators: int,
    device: str,
    seed: int,
) -> tuple[Pipeline, list[str]]:
    feature_columns = numeric_features + categorical_features
    model = create_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_estimators=n_estimators,
        device=device,
        random_state=seed,
    )
    model.fit(
        calibration_frame[feature_columns],
        calibration_frame[residual_column],
        model__sample_weight=calibration_frame[WEIGHT_COLUMN],
    )
    return model, feature_columns


def run_seed(
    frame_2022: pd.DataFrame,
    calibration_fraction: float,
    seed: int,
    residual_n_estimators: int,
    device: str,
) -> list[MetricRow]:
    calibration_frame, test_frame = split_calibration_test(frame_2022, calibration_fraction, seed)
    rows: list[MetricRow] = []
    calibration_residuals = calibration_frame["base_residual"].to_numpy(dtype=float)
    calibration_weights = calibration_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    weighted_residual_mean = weighted_average(calibration_residuals, calibration_weights)

    method_predictions: dict[str, np.ndarray] = {
        "historical_xgboost": test_frame["base_prediction"].to_numpy(dtype=float),
        "bias_shift_calibration": test_frame["base_prediction"].to_numpy(dtype=float) + weighted_residual_mean,
    }

    residual_targets = calibration_frame.copy()
    residual_targets["residual_target"] = residual_targets["base_residual"]
    tabular_numeric = [*BASE_NUMERIC_FEATURES, "survey_year"]
    tabular_categorical = list(BASE_CATEGORICAL_FEATURES)

    residual_specs = [
        ("residual_llm_only", list(LLM_NUMERIC_FEATURES), []),
        ("residual_tabular_only", tabular_numeric, tabular_categorical),
        ("residual_tabular_plus_llm", [*tabular_numeric, *LLM_NUMERIC_FEATURES], tabular_categorical),
    ]
    for method, numeric_features, categorical_features in residual_specs:
        residual_model, feature_columns = fit_residual_model(
            calibration_frame=residual_targets,
            residual_column="residual_target",
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_estimators=residual_n_estimators,
            device=device,
            seed=seed,
        )
        residual_prediction = residual_model.predict(test_frame[feature_columns])
        method_predictions[method] = test_frame["base_prediction"].to_numpy(dtype=float) + residual_prediction

    for method, predictions in method_predictions.items():
        clipped_predictions = np.maximum(predictions, 0.0)
        rows.append(
            evaluate_predictions(
                method=method,
                seed=seed,
                calibration_fraction=calibration_fraction,
                calibration_rows=len(calibration_frame),
                test_frame=test_frame,
                predictions=clipped_predictions,
                device=device,
            )
        )
    return rows


def metric_rows_to_dicts(rows: list[MetricRow]) -> list[dict[str, str | int | float]]:
    return [row.__dict__ for row in rows]


def write_metrics_csv(rows: list[MetricRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = metric_rows_to_dicts(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0]))
        writer.writeheader()
        writer.writerows(row_dicts)


def write_joined_features(frame: pd.DataFrame, output_path: Path) -> None:
    columns = [
        ID_COLUMN,
        "cohort_id",
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        "base_prediction",
        "base_residual",
        *LLM_NUMERIC_FEATURES,
        "primary_event_mechanism",
        "short_explanation",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, columns].to_csv(output_path, index=False)


def summarize_metrics(rows: list[MetricRow]) -> pd.DataFrame:
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
    )
    return summary.sort_values("weighted_mae_mean")


def write_summary(rows: list[MetricRow], output_path: Path) -> None:
    summary = summarize_metrics(rows)
    lines = [
        "# LLM Residual Adaptation Results",
        "",
        "| Method | Runs | Weighted MAE | Weighted RMSE | Weighted Bias | Weighted R2 | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.runs} | "
            f"{row.weighted_mae_mean:.4f} +/- {row.weighted_mae_std:.4f} | "
            f"{row.weighted_rmse_mean:.4f} | {row.weighted_bias_mean:.4f} +/- {row.weighted_bias_std:.4f} | "
            f"{row.weighted_r2_mean:.4f} | {row.r2_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Historical model is trained on 2001, 2009, and 2017 households.",
            "- 2022 is split into calibration/test subsets for each seed.",
            "- Residual target is `CNTTDHH - historical_prediction`.",
            "- All XGBoost models use CUDA.",
            "- Lower MAE/RMSE and bias closer to zero are better.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Resolved training device: %s", device)

    full_frame = pd.read_csv(args.dataset_path, low_memory=False)
    history_model, history_feature_columns = train_history_model(full_frame, args.history_n_estimators, device)
    frame_2022 = load_2022_with_llm_features(
        args.dataset_path,
        args.cohort_profiles_path,
        args.llm_features_path,
    )
    frame_2022["base_prediction"] = np.maximum(
        history_model.predict(frame_2022[history_feature_columns]),
        0.0,
    )
    frame_2022["base_residual"] = frame_2022[TARGET_COLUMN] - frame_2022["base_prediction"]
    write_joined_features(frame_2022, args.output_dir / "household_2022_with_llm_features.csv")

    all_rows: list[MetricRow] = []
    for seed in seeds:
        LOGGER.info("Running residual adaptation split with seed %s.", seed)
        all_rows.extend(
            run_seed(
                frame_2022=frame_2022,
                calibration_fraction=args.calibration_fraction,
                seed=seed,
                residual_n_estimators=args.residual_n_estimators,
                device=device,
            )
        )

    write_metrics_csv(all_rows, args.output_dir / "llm_residual_adaptation_metrics.csv")
    write_summary(all_rows, args.output_dir / "llm_residual_adaptation_results.md")
    LOGGER.info("Wrote residual adaptation results to %s", args.output_dir)


if __name__ == "__main__":
    main()
