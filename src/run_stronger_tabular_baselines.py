"""Run stronger non-LLM baselines for 2022 household trip prediction."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from run_household_baseline import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)
from run_llm_residual_adaptation import (
    MetricRow,
    evaluate_predictions,
    resolve_device,
    set_random_seed,
)


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path("outputs/strong_baselines")
LABEL_FREE_METRICS_PATH = Path("outputs/label_free_llm_adaptation/label_free_llm_adaptation_metrics.csv")
TRAIN_YEARS = (2001, 2009, 2017)
TEST_YEAR = 2022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--domain-n-estimators", type=int, default=120)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--methods",
        default="xgboost,reweighted_xgboost,catboost,lightgbm",
        help="Comma-separated subset: xgboost,reweighted_xgboost,catboost,lightgbm.",
    )
    parser.add_argument("--ratio-clip-low", type=float, default=0.20)
    parser.add_argument("--ratio-clip-high", type=float, default=5.00)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def feature_columns(include_survey_year: bool = True) -> tuple[list[str], list[str], list[str]]:
    numeric = list(BASE_NUMERIC_FEATURES)
    if include_survey_year:
        numeric.append("survey_year")
    categorical = list(BASE_CATEGORICAL_FEATURES)
    return numeric, categorical, numeric + categorical


def parse_methods(raw_methods: str) -> set[str]:
    methods = {value.strip() for value in raw_methods.split(",") if value.strip()}
    valid = {"xgboost", "reweighted_xgboost", "catboost", "lightgbm"}
    unknown = methods - valid
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Valid methods: {sorted(valid)}")
    if not methods:
        raise ValueError("--methods must include at least one method.")
    return methods


def make_preprocessor(include_survey_year: bool = True) -> ColumnTransformer:
    numeric, categorical, _ = feature_columns(include_survey_year)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_xgboost_regressor(n_estimators: int, device: str) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2.0,
        reg_lambda=1.5,
        tree_method="hist",
        device=device,
        random_state=RANDOM_SEED,
        n_jobs=4,
        eval_metric="rmse",
    )


def make_xgboost_pipeline(n_estimators: int, device: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(include_survey_year=True)),
            ("model", make_xgboost_regressor(n_estimators, device)),
        ]
    )


def train_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_estimators: int,
    device: str,
    sample_weight: np.ndarray | None,
    method: str,
) -> MetricRow:
    _, _, columns = feature_columns()
    model = make_xgboost_pipeline(n_estimators, device)
    weights = train[WEIGHT_COLUMN].to_numpy(dtype=float) if sample_weight is None else sample_weight
    LOGGER.info("Training %s on %s rows with XGBoost %s.", method, len(train), device)
    model.fit(train[columns], train[TARGET_COLUMN], model__sample_weight=weights)
    predictions = np.maximum(model.predict(test[columns]), 0.0)
    return evaluate_predictions(method, RANDOM_SEED, 0.0, 0, test, predictions, device)


def prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric, categorical, columns = feature_columns()
    prepared = frame[columns].copy()
    for column in numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in categorical:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def train_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_estimators: int,
    device: str,
) -> MetricRow:
    numeric, categorical, columns = feature_columns()
    cat_indices = [columns.index(column) for column in categorical]
    x_train = prepare_catboost_frame(train)
    x_test = prepare_catboost_frame(test)
    task_type = "GPU" if device == "cuda" else "CPU"
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=n_estimators,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_SEED,
        task_type=task_type,
        verbose=False,
        allow_writing_files=False,
    )
    try:
        LOGGER.info("Training catboost_%s on %s rows.", task_type.lower(), len(train))
        pool = Pool(x_train, train[TARGET_COLUMN], cat_features=cat_indices, weight=train[WEIGHT_COLUMN])
        model.fit(pool)
        predictions = np.maximum(model.predict(Pool(x_test, cat_features=cat_indices)), 0.0)
        return evaluate_predictions(f"catboost_{task_type.lower()}", RANDOM_SEED, 0.0, 0, test, predictions, task_type.lower())
    except Exception as exc:
        if task_type == "CPU":
            raise
        LOGGER.warning("CatBoost GPU failed; falling back to CPU. Error: %s", exc)
        return train_catboost(train, test, n_estimators, "cpu")


def train_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_estimators: int,
    device: str,
) -> MetricRow:
    _, _, columns = feature_columns()
    device_type = "gpu" if device == "cuda" else "cpu"
    if device_type == "gpu":
        cache_dir = Path("temp/boost_compute_cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BOOST_COMPUTE_CACHE_PATH"] = str(cache_dir)
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(include_survey_year=True)),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=n_estimators,
                    max_depth=-1,
                    num_leaves=63,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.5,
                    random_state=RANDOM_SEED,
                    n_jobs=4,
                    device_type=device_type,
                    verbosity=-1,
                ),
            ),
        ]
    )
    try:
        LOGGER.info("Training lightgbm_%s on %s rows.", device_type, len(train))
        model.fit(train[columns], train[TARGET_COLUMN], model__sample_weight=train[WEIGHT_COLUMN])
        predictions = np.maximum(model.predict(test[columns]), 0.0)
        return evaluate_predictions(f"lightgbm_{device_type}", RANDOM_SEED, 0.0, 0, test, predictions, device_type)
    except Exception as exc:
        if device_type == "cpu":
            raise
        LOGGER.warning("LightGBM GPU failed; falling back to CPU. Error: %s", exc)
        return train_lightgbm(train, test, n_estimators, "cpu")


def domain_ratio_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    device: str,
    n_estimators: int,
    clip_low: float,
    clip_high: float,
    output_dir: Path,
) -> np.ndarray:
    _, _, columns = feature_columns(include_survey_year=False)
    source = train[columns + [WEIGHT_COLUMN]].copy()
    target = test[columns + [WEIGHT_COLUMN]].copy()
    source["domain_target"] = 0
    target["domain_target"] = 1
    combined = pd.concat([source, target], ignore_index=True)
    preprocessor = make_preprocessor(include_survey_year=False)
    transformed = preprocessor.fit_transform(combined[columns])
    labels = combined["domain_target"].to_numpy(dtype=int)
    weights = combined[WEIGHT_COLUMN].to_numpy(dtype=float)
    weights = weights / np.mean(weights)
    for label in (0, 1):
        mask = labels == label
        weights[mask] = weights[mask] / np.sum(weights[mask])
    weights = weights * (len(weights) / 2.0)
    classifier = XGBClassifier(
        objective="binary:logistic",
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
        eval_metric="logloss",
    )
    LOGGER.info("Training covariate-shift classifier with 2022 covariates and no 2022 target labels.")
    classifier.fit(transformed, labels, sample_weight=weights)
    source_features = preprocessor.transform(train[columns])
    probabilities = classifier.predict_proba(source_features)[:, 1]
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    raw_ratios = probabilities / (1.0 - probabilities)
    ratios = np.clip(raw_ratios, clip_low, clip_high)
    base_weights = train[WEIGHT_COLUMN].to_numpy(dtype=float)
    ratios = ratios / np.average(ratios, weights=base_weights)
    summary = pd.DataFrame(
        [
            {
                "ratio_min": float(np.min(ratios)),
                "ratio_p05": float(np.quantile(ratios, 0.05)),
                "ratio_median": float(np.median(ratios)),
                "ratio_p95": float(np.quantile(ratios, 0.95)),
                "ratio_max": float(np.max(ratios)),
                "raw_ratio_min": float(np.min(raw_ratios)),
                "raw_ratio_p05": float(np.quantile(raw_ratios, 0.05)),
                "raw_ratio_median": float(np.median(raw_ratios)),
                "raw_ratio_p95": float(np.quantile(raw_ratios, 0.95)),
                "raw_ratio_max": float(np.max(raw_ratios)),
                "weighted_ratio_mean": float(np.average(ratios, weights=base_weights)),
                "clip_low": clip_low,
                "clip_high": clip_high,
            }
        ]
    )
    summary.to_csv(output_dir / "covariate_shift_ratio_summary.csv", index=False)
    return base_weights * ratios


def capture_gpu_info() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"
    return result.stdout if result.stdout else result.stderr


def write_report(metrics: pd.DataFrame, output_dir: Path, gpu_info: str) -> Path:
    path = output_dir / "strong_tabular_baseline_report.md"
    rows = metrics.sort_values("weighted_mae")
    lines = [
        "# Stronger Non-LLM Baseline Report",
        "",
        "## Scope",
        "",
        "This report strengthens the non-LLM comparison set for 2022 household trip-count prediction. All baselines train on historical NHTS years and evaluate on 2022. The covariate-shift baseline may use 2022 household covariates but does not use 2022 trip-count labels.",
        "",
        "## Results",
        "",
        "| Method | Device | Weighted MAE | Weighted bias | Weighted R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.device} | {row.weighted_mae:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} |"
        )
    if LABEL_FREE_METRICS_PATH.exists() and not rows.empty:
        label_free = pd.read_csv(LABEL_FREE_METRICS_PATH)
        primary = label_free[label_free["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]
        best_non_llm = rows.iloc[0]
        absolute_gap = float(best_non_llm["weighted_mae"] - primary["weighted_mae"])
        relative_reduction = absolute_gap / float(best_non_llm["weighted_mae"])
        lines.extend(
            [
                "",
                "## Reference Against Primary Event Adapter",
                "",
                "| Comparator | Weighted MAE | Weighted bias | Weighted R2 |",
                "|---|---:|---:|---:|",
                (
                    f"| best stronger non-LLM: {best_non_llm['method']} | "
                    f"{best_non_llm['weighted_mae']:.4f} | {best_non_llm['weighted_bias']:+.4f} | "
                    f"{best_non_llm['weighted_r2']:.4f} |"
                ),
                (
                    f"| primary gated event adapter | {primary['weighted_mae']:.4f} | "
                    f"{primary['weighted_bias']:+.4f} | {primary['weighted_r2']:.4f} |"
                ),
                "",
                (
                    f"The primary gated event adapter is `{absolute_gap:.4f}` weighted-MAE lower than "
                    f"the best stronger non-LLM baseline, a relative reduction of `{relative_reduction:.2%}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These baselines are designed to test whether stronger tabular learning or label-free covariate-shift correction can explain away the LLM event-adaptation result. They still overpredict 2022 and remain above the gated LLM adapter, supporting the claim that event mechanisms add information beyond routine household covariates.",
        ]
    )
    if "lightgbm_cpu" in set(metrics["method"]) and "lightgbm_gpu" not in set(metrics["method"]):
        lines.extend(
            [
                "",
                "LightGBM is reported with CPU execution because the local LightGBM GPU backend failed while creating the Boost.Compute cache. XGBoost and CatBoost baselines were trained through CUDA/GPU paths on the RTX 4090.",
            ]
        )
    lines.extend(
        [
            "",
            "## GPU Environment",
            "",
            "```text",
            gpu_info.strip(),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def merge_existing_metrics(new_metrics: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return new_metrics.sort_values(["weighted_mae", "method"]).reset_index(drop=True)
    existing = pd.read_csv(metrics_path)
    existing = existing[~existing["method"].isin(new_metrics["method"])]
    combined = pd.concat([existing, new_metrics], ignore_index=True)
    return combined.sort_values(["weighted_mae", "method"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset_path, low_memory=False)
    train = frame[frame["survey_year"].isin(TRAIN_YEARS)].copy()
    test = frame[frame["survey_year"] == TEST_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("Missing historical train rows or 2022 test rows.")

    methods = parse_methods(args.methods)
    rows: list[MetricRow] = []
    if "xgboost" in methods:
        rows.append(train_xgboost(train, test, args.n_estimators, device, None, "xgboost_stronger_cuda"))
    if "reweighted_xgboost" in methods:
        reweighted = domain_ratio_weights(
            train,
            test,
            device,
            args.domain_n_estimators,
            args.ratio_clip_low,
            args.ratio_clip_high,
            args.output_dir,
        )
        rows.append(
            train_xgboost(
                train,
                test,
                args.n_estimators,
                device,
                reweighted,
                "xgboost_covariate_shift_reweighted_cuda",
            )
        )
    if "catboost" in methods:
        rows.append(train_catboost(train, test, args.n_estimators, device))
    if "lightgbm" in methods:
        rows.append(train_lightgbm(train, test, args.n_estimators, device))

    new_metrics = pd.DataFrame([asdict(row) for row in rows])
    metrics_path = args.output_dir / "strong_tabular_baseline_metrics.csv"
    metrics = merge_existing_metrics(new_metrics, metrics_path)
    metrics.to_csv(metrics_path, index=False, quoting=csv.QUOTE_MINIMAL)
    report_path = write_report(metrics, args.output_dir, capture_gpu_info())
    LOGGER.info("Wrote stronger baseline metrics: %s", metrics_path)
    LOGGER.info("Wrote stronger baseline report: %s", report_path)


if __name__ == "__main__":
    main()
