"""Export reproducibility artifacts for the final label-free experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from run_household_baseline import ID_COLUMN, RANDOM_SEED, TARGET_COLUMN, WEIGHT_COLUMN
from run_label_free_llm_adaptation import (
    add_event_pressure,
    apply_multiplicative_pressure,
    evaluate_full_2022,
    make_historical_mean_prediction,
    make_delta_gated_pressure,
    make_global_pressure,
)
from run_llm_residual_adaptation import (
    configure_logging,
    load_2022_with_llm_features,
    metric_rows_to_dicts,
    resolve_device,
    set_random_seed,
    train_history_model,
)


LOGGER = logging.getLogger(__name__)
PRIMARY_METHOD = "gated_trip_suppression_a1_d0p15"
BEST_MAE_SENSITIVITY_METHOD = "llm_trip_suppression_a1p25"


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/final_label_free_2022"))
    parser.add_argument("--history-n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--primary-alpha", type=float, default=1.0)
    parser.add_argument("--primary-delta-threshold", type=float, default=0.15)
    parser.add_argument("--sensitivity-alpha", type=float, default=1.25)
    parser.add_argument("--min-factor", type=float, default=0.05)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return f"unavailable: {output}"
    return output


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


def make_predictions(
    full_frame: pd.DataFrame,
    frame_2022: pd.DataFrame,
    primary_alpha: float,
    primary_delta_threshold: float,
    sensitivity_alpha: float,
    min_factor: float,
) -> dict[str, np.ndarray]:
    base_prediction = frame_2022["base_prediction"].to_numpy(dtype=float)
    weights = frame_2022[WEIGHT_COLUMN].to_numpy(dtype=float)
    trip_pressure = frame_2022["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    global_pressure = make_global_pressure(trip_pressure, weights)
    gated_pressure = make_delta_gated_pressure(trip_pressure, global_pressure, primary_delta_threshold)
    historical_mean_prediction = make_historical_mean_prediction(full_frame, len(frame_2022))
    return {
        "historical_mean_only": historical_mean_prediction,
        "historical_xgboost": base_prediction,
        "llm_only_trip_suppression_a1p25": apply_multiplicative_pressure(
            historical_mean_prediction,
            trip_pressure,
            sensitivity_alpha,
            min_factor,
        ),
        "global_trip_suppression_a1p25": apply_multiplicative_pressure(
            base_prediction,
            global_pressure,
            sensitivity_alpha,
            min_factor,
        ),
        BEST_MAE_SENSITIVITY_METHOD: apply_multiplicative_pressure(
            base_prediction,
            trip_pressure,
            sensitivity_alpha,
            min_factor,
        ),
        PRIMARY_METHOD: apply_multiplicative_pressure(
            base_prediction,
            gated_pressure,
            primary_alpha,
            min_factor,
        ),
    }


def write_predictions(frame_2022: pd.DataFrame, predictions: dict[str, np.ndarray], output_path: Path) -> None:
    columns = [
        ID_COLUMN,
        "cohort_id",
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        "base_prediction",
        "llm_trip_suppression_pressure",
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "transit_avoidance_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
        "confidence",
    ]
    export = frame_2022.loc[:, columns].copy()
    for method, values in predictions.items():
        export[f"prediction_{method}"] = values
        export[f"error_{method}"] = values - export[TARGET_COLUMN].to_numpy(dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False)


def write_metadata(
    output_path: Path,
    args: argparse.Namespace,
    feature_columns: list[str],
    metrics_path: Path,
    model_path: Path,
) -> None:
    metadata = {
        "task": "2022 household trip-count prediction",
        "target": TARGET_COLUMN,
        "weight": WEIGHT_COLUMN,
        "random_seed": RANDOM_SEED,
        "primary_method": PRIMARY_METHOD,
        "primary_rule": {
            "alpha": args.primary_alpha,
            "delta_threshold": args.primary_delta_threshold,
            "min_factor": args.min_factor,
            "description": "Use cohort LLM trip-suppression pressure only when it differs from the global weighted mean by the fixed threshold; otherwise use the global pressure.",
        },
        "best_mae_sensitivity_method": BEST_MAE_SENSITIVITY_METHOD,
        "sensitivity_rule": {
            "alpha": args.sensitivity_alpha,
            "min_factor": args.min_factor,
            "description": "Reported as sensitivity analysis, not the strict primary no-label selection rule.",
        },
        "label_usage": {
            "train_years": [2001, 2009, 2017],
            "target_year": 2022,
            "target_year_labels_used_for_training_or_calibration": False,
            "target_year_labels_used_for_final_metrics": True,
        },
        "paths": {
            "dataset_path": str(args.dataset_path),
            "cohort_profiles_path": str(args.cohort_profiles_path),
            "llm_features_path": str(args.llm_features_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
        "hashes": {
            "dataset_sha256": sha256_file(args.dataset_path),
            "cohort_profiles_sha256": sha256_file(args.cohort_profiles_path),
            "llm_features_sha256": sha256_file(args.llm_features_path),
        },
        "feature_columns": feature_columns,
        "environment": package_versions(),
        "gpu": run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    }
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics(
    frame_2022: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    device: str,
    output_path: Path,
) -> None:
    rows = [
        evaluate_full_2022(frame_2022, method, values, RANDOM_SEED, device)
        for method, values in predictions.items()
    ]
    pd.DataFrame(metric_rows_to_dicts(rows)).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    LOGGER.info("Exporting final artifacts with device=%s.", device)

    full_frame = pd.read_csv(args.dataset_path, low_memory=False)
    history_model, feature_columns = train_history_model(full_frame, args.history_n_estimators, device)
    frame_2022 = load_2022_with_llm_features(
        args.dataset_path,
        args.cohort_profiles_path,
        args.llm_features_path,
    )
    frame_2022["base_prediction"] = np.maximum(history_model.predict(frame_2022[feature_columns]), 0.0)
    frame_2022 = add_event_pressure(frame_2022)
    predictions = make_predictions(
        full_frame=full_frame,
        frame_2022=frame_2022,
        primary_alpha=args.primary_alpha,
        primary_delta_threshold=args.primary_delta_threshold,
        sensitivity_alpha=args.sensitivity_alpha,
        min_factor=args.min_factor,
    )

    model_path = args.output_dir / "historical_routine_model.joblib"
    metrics_path = args.output_dir / "final_artifact_metrics.csv"
    prediction_path = args.output_dir / "final_2022_predictions.csv"
    metadata_path = args.output_dir / "metadata.json"
    freeze_path = args.output_dir / "environment_freeze.txt"

    joblib.dump(history_model, model_path)
    write_metrics(frame_2022, predictions, device, metrics_path)
    write_predictions(frame_2022, predictions, prediction_path)
    write_metadata(metadata_path, args, feature_columns, metrics_path, model_path)
    freeze_path.write_text(run_command([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    LOGGER.info("Wrote final reproducibility artifacts to %s.", args.output_dir)


if __name__ == "__main__":
    main()
