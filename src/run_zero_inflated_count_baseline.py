"""Run zero-inflated count-model baselines for 2022 trip prediction."""

from __future__ import annotations

import argparse
import csv
import logging
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson

from run_household_baseline import (
    BASE_NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)
from run_llm_residual_adaptation import MetricRow, evaluate_predictions, set_random_seed


LOGGER = logging.getLogger(__name__)
TRAIN_YEARS = (2001, 2009, 2017)
TEST_YEAR = 2022
OUTPUT_DIR = Path("outputs/zero_inflated_count_baseline")
LABEL_FREE_METRICS_PATH = Path("outputs/label_free_llm_adaptation/label_free_llm_adaptation_metrics.csv")
COUNT_BASELINE_METRICS_PATH = Path("outputs/count_model_baselines/count_model_baseline_metrics.csv")
NB_BASELINE_METRICS_PATH = Path("outputs/negative_binomial_baseline/negative_binomial_2022_metrics.csv")

COUNT_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "URBRUR",
    "URBAN",
    "RAIL",
    "TRAVDAY",
    "HHFAMINC",
    "LIF_CYC",
    "MSACAT",
)
INFLATION_NUMERIC_FEATURES: tuple[str, ...] = (
    "HHSIZE",
    "HHVEHCNT",
    "WRKCOUNT",
    "DRVRCNT",
)
INFLATION_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "URBRUR",
    "URBAN",
    "HHFAMINC",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--methods", default="zip")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def summarize_warnings(messages: list[str], max_items: int = 6) -> tuple[int, str]:
    """Return a compact warning summary suitable for committed diagnostics."""
    if not messages:
        return 0, ""
    counter = Counter(messages)
    parts = []
    for message, count in counter.most_common(max_items):
        label = message.replace("\n", " ").strip()
        parts.append(f"{label} (x{count})")
    omitted = len(counter) - len(parts)
    if omitted > 0:
        parts.append(f"{omitted} additional warning types omitted")
    return len(messages), " | ".join(parts)


def parse_methods(raw: str) -> list[str]:
    methods = [value.strip().lower() for value in raw.split(",") if value.strip()]
    valid = {"zip", "zinb"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unsupported zero-inflated methods: {unknown}")
    return methods


def count_numeric_features() -> list[str]:
    return [*BASE_NUMERIC_FEATURES, "survey_year"]


def count_feature_columns() -> list[str]:
    return [*count_numeric_features(), *COUNT_CATEGORICAL_FEATURES]


def inflation_feature_columns() -> list[str]:
    return [*INFLATION_NUMERIC_FEATURES, *INFLATION_CATEGORICAL_FEATURES]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def transform_designs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count_preprocessor = make_preprocessor(count_numeric_features(), list(COUNT_CATEGORICAL_FEATURES))
    inflation_preprocessor = make_preprocessor(list(INFLATION_NUMERIC_FEATURES), list(INFLATION_CATEGORICAL_FEATURES))
    count_train = count_preprocessor.fit_transform(train[count_feature_columns()])
    count_test = count_preprocessor.transform(test[count_feature_columns()])
    inflation_train = inflation_preprocessor.fit_transform(train[inflation_feature_columns()])
    inflation_test = inflation_preprocessor.transform(test[inflation_feature_columns()])
    return (
        sm.add_constant(np.asarray(count_train, dtype=float), prepend=True, has_constant="add"),
        sm.add_constant(np.asarray(count_test, dtype=float), prepend=True, has_constant="add"),
        sm.add_constant(np.asarray(inflation_train, dtype=float), prepend=True, has_constant="add"),
        sm.add_constant(np.asarray(inflation_test, dtype=float), prepend=True, has_constant="add"),
    )


def model_factory(method: str) -> Callable[..., ZeroInflatedPoisson | ZeroInflatedNegativeBinomialP]:
    if method == "zip":
        return ZeroInflatedPoisson
    if method == "zinb":
        return ZeroInflatedNegativeBinomialP
    raise ValueError(f"Unsupported method: {method}")


def fit_one_method(
    method: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_iter: int,
) -> tuple[MetricRow | None, dict[str, object]]:
    exog_train, exog_test, exog_infl_train, exog_infl_test = transform_designs(train, test)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    model_class = model_factory(method)
    LOGGER.info(
        "Fitting %s on %s rows, count columns=%s, inflation columns=%s.",
        method,
        len(train),
        exog_train.shape[1],
        exog_infl_train.shape[1],
    )
    warning_messages: list[str] = []
    diagnostic: dict[str, object] = {
        "method": method,
        "status": "ok",
        "max_iter": max_iter,
        "count_columns": int(exog_train.shape[1]),
        "inflation_columns": int(exog_infl_train.shape[1]),
        "converged": False,
        "n_iter": -1,
        "warning_message": "",
        "error_message": "",
    }
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = model_class(y_train, exog_train, exog_infl=exog_infl_train, inflation="logit")
            result = model.fit(method="bfgs", maxiter=max_iter, disp=0)
            warning_messages = [str(item.message) for item in captured]
        predictions = np.asarray(result.predict(exog=exog_test, exog_infl=exog_infl_test, which="mean"), dtype=float)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=100.0, neginf=0.0)
        predictions = np.clip(predictions, 0.0, max(float(train[TARGET_COLUMN].quantile(0.999) * 2.0), 1.0))
        metric = evaluate_predictions(
            method=f"zero_inflated_{method}_compact",
            seed=RANDOM_SEED,
            calibration_fraction=0.0,
            calibration_rows=0,
            test_frame=test,
            predictions=predictions,
            device="cpu",
        )
        mle_retvals = getattr(result, "mle_retvals", {}) or {}
        warning_count, warning_summary = summarize_warnings(warning_messages)
        diagnostic.update(
            {
                "converged": bool(mle_retvals.get("converged", False)),
                "n_iter": int(mle_retvals.get("iterations", -1)),
                "warning_count": warning_count,
                "warning_message": warning_summary,
                "aic": float(getattr(result, "aic", np.nan)),
                "llf": float(getattr(result, "llf", np.nan)),
                "weighted_mae": metric.weighted_mae,
                "weighted_bias": metric.weighted_bias,
            }
        )
        return metric, diagnostic
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        warning_count, warning_summary = summarize_warnings(warning_messages)
        diagnostic.update(
            {
                "status": "failed",
                "error_message": str(error),
                "warning_count": warning_count,
                "warning_message": warning_summary,
                "weighted_mae": np.nan,
                "weighted_bias": np.nan,
            }
        )
        LOGGER.warning("%s failed: %s", method, error)
        return None, diagnostic


def write_metrics(rows: list[MetricRow], output_dir: Path) -> pd.DataFrame:
    metrics = pd.DataFrame([asdict(row) for row in rows])
    if not metrics.empty:
        metrics = metrics.sort_values(["weighted_mae", "method"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "zero_inflated_count_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return metrics


def write_diagnostics(diagnostics: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(diagnostics)
    if not frame.empty:
        frame = frame.sort_values("method").reset_index(drop=True)
    frame.to_csv(output_dir / "zero_inflated_count_diagnostics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def load_reference_row(path: Path, method: str | None = None) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if method is not None:
        frame = frame[frame["method"] == method]
    if frame.empty:
        return None
    return frame.sort_values("weighted_mae").iloc[0]


def fmt(value: object, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "NA"
    numeric = float(value)
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


def write_report(metrics: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "zero_inflated_count_baseline_report.md"
    primary = load_reference_row(LABEL_FREE_METRICS_PATH, "gated_trip_suppression_a1_d0p15")
    poisson = load_reference_row(COUNT_BASELINE_METRICS_PATH, "poisson_glm_l2")
    nb = load_reference_row(NB_BASELINE_METRICS_PATH)
    lines = [
        "# Zero-Inflated Count Baseline Report",
        "",
        "## Scope",
        "",
        "This experiment adds zero-inflated count-model baselines because household daily trip counts can contain structural zeros. Models train only on 2001, 2009, and 2017 NHTS households and evaluate once on 2022. No 2022 labels are used for training or calibration.",
        "",
        "The implementation uses a compact statsmodels design matrix rather than the full one-hot household design, because full zero-inflated maximum-likelihood optimization is slow and numerically fragile on this survey table. Fits are unweighted due to statsmodels discrete zero-inflated model limitations; all reported metrics are still survey-weighted. The default committed run uses zero-inflated Poisson; zero-inflated negative binomial remains available through `--methods zinb` but is not the default because it is much less stable on this design.",
        "",
        "## Results",
        "",
        "| Method | Weighted MAE | Weighted RMSE | Weighted bias | Weighted R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    if metrics.empty:
        lines.append("| no successful model | NA | NA | NA | NA |")
    else:
        for row in metrics.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
                f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} |"
            )
    lines.extend(["", "## Diagnostics", "", "| Method | Status | Converged | Iterations | Weighted MAE | Weighted bias |", "|---|---|---|---:|---:|---:|"])
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.status} | {row.converged} | {row.n_iter} | "
            f"{fmt(row.weighted_mae)} | {fmt(row.weighted_bias, signed=True)} |"
        )
    if not metrics.empty:
        best = metrics.sort_values("weighted_mae").iloc[0]
        lines.extend(["", "## Reference Comparisons", "", "| Comparator | Weighted MAE | Weighted bias | Weighted R2 |", "|---|---:|---:|---:|"])
        lines.append(
            f"| best zero-inflated count: {best['method']} | {fmt(best['weighted_mae'])} | "
            f"{fmt(best['weighted_bias'], signed=True)} | {fmt(best['weighted_r2'])} |"
        )
        if poisson is not None:
            lines.append(f"| Poisson GLM | {fmt(poisson['weighted_mae'])} | {fmt(poisson['weighted_bias'], signed=True)} | {fmt(poisson['weighted_r2'])} |")
        if nb is not None:
            lines.append(f"| Negative-binomial GLM | {fmt(nb['weighted_mae'])} | {fmt(nb['weighted_bias'], signed=True)} | {fmt(nb['weighted_r2'])} |")
        if primary is not None:
            gap = float(best["weighted_mae"] - primary["weighted_mae"])
            reduction = gap / float(best["weighted_mae"])
            lines.append(
                f"| primary gated event adapter | {fmt(primary['weighted_mae'])} | "
                f"{fmt(primary['weighted_bias'], signed=True)} | {fmt(primary['weighted_r2'])} |"
            )
            lines.append("")
            lines.append(
                f"The primary gated event adapter is `{gap:.4f}` weighted-MAE lower than the best zero-inflated count baseline, "
                f"a relative reduction of `{reduction:.2%}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The zero-inflated models are a reviewer-facing stress test for count-data adequacy. If they still overpredict 2022 or underperform the event adapter, that supports the paper's claim that the central issue is target-year event shift rather than merely choosing a count-family likelihood.",
            "",
            "Do not present this as the final word on all possible travel-demand count models. It is a compact, transparent robustness baseline with explicit solver and weighting caveats.",
        ]
    )
    if diagnostics["warning_message"].fillna("").astype(str).str.len().gt(0).any() or (~diagnostics["converged"].fillna(False)).any():
        lines.extend(
            [
                "",
                "## Solver Caveat",
                "",
                "At least one zero-inflated fit emitted warnings or did not report convergence. The diagnostics CSV preserves this state so the paper can report the baseline honestly.",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.dataset_path, low_memory=False)
    train = frame[frame["survey_year"].isin(TRAIN_YEARS)].copy()
    test = frame[frame["survey_year"] == TEST_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError("Missing historical training rows or 2022 test rows.")
    metrics: list[MetricRow] = []
    diagnostics: list[dict[str, object]] = []
    for method in parse_methods(args.methods):
        metric, diagnostic = fit_one_method(method, train, test, max_iter=args.max_iter)
        if metric is not None:
            metrics.append(metric)
        diagnostics.append(diagnostic)
    metrics_frame = write_metrics(metrics, args.output_dir)
    diagnostics_frame = write_diagnostics(diagnostics, args.output_dir)
    report_path = write_report(metrics_frame, diagnostics_frame, args.output_dir)
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
