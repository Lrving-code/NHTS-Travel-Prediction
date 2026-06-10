"""Run transparent count-model baselines for 2022 household trip prediction."""

from __future__ import annotations

import argparse
import csv
import logging
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

from run_household_baseline import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)
from run_llm_residual_adaptation import MetricRow, evaluate_predictions, set_random_seed


LOGGER = logging.getLogger(__name__)
TRAIN_YEARS = (2001, 2009, 2017)
TEST_YEAR = 2022
OUTPUT_DIR = Path("outputs/count_model_baselines")
LABEL_FREE_METRICS_PATH = Path("outputs/label_free_llm_adaptation/label_free_llm_adaptation_metrics.csv")
STRONG_BASELINE_METRICS_PATH = Path("outputs/strong_baselines/strong_tabular_baseline_metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1e-4)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def feature_columns() -> tuple[list[str], list[str], list[str]]:
    numeric = [*BASE_NUMERIC_FEATURES, "survey_year"]
    categorical = list(BASE_CATEGORICAL_FEATURES)
    return numeric, categorical, [*numeric, *categorical]


def make_preprocessor() -> ColumnTransformer:
    numeric, categorical, _ = feature_columns()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
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
        ],
        sparse_threshold=1.0,
    )


def make_pipeline(method: str, alpha: float, max_iter: int) -> Pipeline:
    if method == "poisson_glm_l2":
        model = PoissonRegressor(alpha=alpha, max_iter=max_iter, tol=1e-7)
    elif method == "tweedie_glm_p1p5":
        model = TweedieRegressor(power=1.5, link="log", alpha=alpha, max_iter=max_iter, tol=1e-7)
    else:
        raise ValueError(f"Unsupported count model method: {method}")
    return Pipeline(steps=[("preprocessor", make_preprocessor()), ("model", model)])


def train_count_model(
    method: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    max_iter: int,
) -> tuple[MetricRow, dict[str, object]]:
    _, _, columns = feature_columns()
    model = make_pipeline(method, alpha=alpha, max_iter=max_iter)
    LOGGER.info("Training %s on %s historical rows.", method, len(train))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[columns], train[TARGET_COLUMN], model__sample_weight=train[WEIGHT_COLUMN])
    convergence_messages = [
        str(item.message)
        for item in captured
        if issubclass(item.category, ConvergenceWarning)
    ]
    predictions = np.maximum(model.predict(test[columns]), 0.0)
    metric = evaluate_predictions(
        method=method,
        seed=RANDOM_SEED,
        calibration_fraction=0.0,
        calibration_rows=0,
        test_frame=test,
        predictions=predictions,
        device="cpu",
    )
    estimator = model.named_steps["model"]
    diagnostic = {
        "method": method,
        "alpha": alpha,
        "max_iter": max_iter,
        "n_iter": int(getattr(estimator, "n_iter_", -1)),
        "convergence_warning": bool(convergence_messages),
        "warning_message": " | ".join(convergence_messages),
    }
    return metric, diagnostic


def write_metrics(rows: list[MetricRow], output_dir: Path) -> pd.DataFrame:
    metrics = pd.DataFrame([asdict(row) for row in rows])
    metrics = metrics.sort_values(["weighted_mae", "method"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "count_model_baseline_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return metrics


def write_diagnostics(diagnostics: list[dict[str, object]], output_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(diagnostics).sort_values("method").reset_index(drop=True)
    frame.to_csv(output_dir / "count_model_solver_diagnostics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


def reference_rows() -> tuple[pd.Series | None, pd.Series | None]:
    primary = None
    catboost = None
    if LABEL_FREE_METRICS_PATH.exists():
        label_free = pd.read_csv(LABEL_FREE_METRICS_PATH)
        match = label_free[label_free["method"] == "gated_trip_suppression_a1_d0p15"]
        if not match.empty:
            primary = match.iloc[0]
    if STRONG_BASELINE_METRICS_PATH.exists():
        strong = pd.read_csv(STRONG_BASELINE_METRICS_PATH)
        match = strong[strong["method"] == "catboost_gpu"]
        if not match.empty:
            catboost = match.iloc[0]
    return primary, catboost


def write_report(metrics: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path, alpha: float, max_iter: int) -> Path:
    primary, catboost = reference_rows()
    path = output_dir / "count_model_baseline_report.md"
    best = metrics.sort_values("weighted_mae").iloc[0]
    lines = [
        "# Count-Model Baseline Report",
        "",
        "## Scope",
        "",
        "This experiment adds transparent count-model baselines familiar to transportation and travel-demand reviewers. Models train only on 2001, 2009, and 2017 NHTS household labels and evaluate on 2022. No 2022 `CNTTDHH` labels are used for training, calibration, model selection, or preprocessing beyond final evaluation.",
        "",
        "The implemented baselines are linear generalized count-model families over the same harmonized household features used by the ML baselines:",
        "",
        "- `poisson_glm_l2`: Poisson GLM with log link and L2 regularization.",
        "- `tweedie_glm_p1p5`: Tweedie GLM with log link and power 1.5, included as an overdispersed count-like baseline.",
        "",
        f"Regularization alpha: `{alpha}`; max iterations: `{max_iter}`.",
        "",
        "## Results",
        "",
        "| Method | Weighted MAE | Weighted RMSE | Weighted bias | Weighted R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} |"
        )
    lines.extend(["", "## Solver Diagnostics", "", "| Method | n_iter | convergence warning |", "|---|---:|---|"])
    for row in diagnostics.itertuples(index=False):
        warning = "yes" if row.convergence_warning else "no"
        lines.append(f"| {row.method} | {row.n_iter} | {warning} |")
    if primary is not None or catboost is not None:
        lines.extend(["", "## Reference Comparisons", "", "| Comparator | Weighted MAE | Weighted bias | Weighted R2 |", "|---|---:|---:|---:|"])
        lines.append(
            f"| best count model: {best['method']} | {best['weighted_mae']:.4f} | "
            f"{best['weighted_bias']:+.4f} | {best['weighted_r2']:.4f} |"
        )
        if catboost is not None:
            lines.append(
                f"| strongest non-LLM tabular: catboost_gpu | {catboost['weighted_mae']:.4f} | "
                f"{catboost['weighted_bias']:+.4f} | {catboost['weighted_r2']:.4f} |"
            )
        if primary is not None:
            gap = float(best["weighted_mae"] - primary["weighted_mae"])
            reduction = gap / float(best["weighted_mae"])
            lines.append(
                f"| primary gated event adapter | {primary['weighted_mae']:.4f} | "
                f"{primary['weighted_bias']:+.4f} | {primary['weighted_r2']:.4f} |"
            )
            lines.extend(
                [
                    "",
                    (
                        f"The primary gated event adapter is `{gap:.4f}` weighted-MAE lower than the best count model, "
                        f"a relative reduction of `{reduction:.2%}`."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The count-model baselines are intentionally transparent rather than high-capacity. Their persistent positive bias on 2022 supports the paper's main framing: historically fitted routine travel-demand models, including classical count families, do not encode the post-pandemic event mechanisms needed to avoid overpredicting 2022 household travel.",
            "",
            "This result should be used as a reviewer-facing baseline, not as a claim that Poisson/Tweedie models are the strongest possible transportation models.",
        ]
    )
    if diagnostics["convergence_warning"].any():
        lines.extend(
            [
                "",
                "## Solver Caveat",
                "",
                "The sklearn L-BFGS solver reached the iteration limit for at least one count model. The results are retained as transparent reviewer-facing baselines, and the diagnostic CSV records the warning. They should not be presented as exhaustively optimized count models.",
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
        raise ValueError("Missing historical train rows or 2022 test rows.")

    outputs = [
        train_count_model("poisson_glm_l2", train, test, alpha=args.alpha, max_iter=args.max_iter),
        train_count_model("tweedie_glm_p1p5", train, test, alpha=args.alpha, max_iter=args.max_iter),
    ]
    rows = [metric for metric, _diagnostic in outputs]
    diagnostics = [diagnostic for _metric, diagnostic in outputs]
    metrics = write_metrics(rows, args.output_dir)
    diagnostics_frame = write_diagnostics(diagnostics, args.output_dir)
    report_path = write_report(metrics, diagnostics_frame, args.output_dir, alpha=args.alpha, max_iter=args.max_iter)
    LOGGER.info("Wrote %s", args.output_dir / "count_model_baseline_metrics.csv")
    LOGGER.info("Wrote %s", args.output_dir / "count_model_solver_diagnostics.csv")
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
