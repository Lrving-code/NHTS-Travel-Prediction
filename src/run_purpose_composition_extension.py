"""Run household-level trip-purpose composition transfer experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_household_baseline import BASE_CATEGORICAL_FEATURES, BASE_NUMERIC_FEATURES, RANDOM_SEED, WEIGHT_COLUMN
from run_llm_residual_adaptation import create_pipeline, load_2022_with_llm_features, resolve_device, weighted_average


LOGGER = logging.getLogger(__name__)
HOUSEHOLD_ID = "HOUSEID"
YEAR_COLUMN = "survey_year"
FIGURE_DIR_NAME = "figures"
PURPOSE_COLUMNS: tuple[str, ...] = (
    "work_share",
    "shopping_share",
    "social_recreation_share",
    "other_home_based_share",
    "non_home_based_share",
)
PURPOSE_LABELS: dict[str, str] = {
    "work_share": "Work",
    "shopping_share": "Shopping",
    "social_recreation_share": "Social / recreation",
    "other_home_based_share": "Other home-based",
    "non_home_based_share": "Non-home-based",
}
FEATURE_COLUMNS = tuple(BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class TripSource:
    """Trip-level source table used to build household purpose targets."""

    year: int
    path: Path


TRIP_SOURCES: tuple[TripSource, ...] = (
    TripSource(2017, Path("data/raw/nhts_2017/csv/trippub.csv")),
    TripSource(2022, Path("data/raw/nhts_2022/csv/tripv2pub.csv")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-path", type=Path, default=Path("data/processed/household_harmonized.csv"))
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/purpose_composition_extension"))
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--alphas", default="0.50,0.75,1.00")
    parser.add_argument("--min-factor", type=float, default=0.05)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_float_grid(raw_values: str) -> list[float]:
    values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one alpha is required.")
    return values


def alpha_token(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def map_trip_purpose(year: int, value: object) -> str | None:
    if year == 2017:
        raw = str(value).strip().upper()
        return {
            "HBW": "work",
            "HBSHOP": "shopping",
            "HBSOCREC": "social_recreation",
            "HBO": "other_home_based",
            "NHB": "non_home_based",
        }.get(raw)
    code = pd.to_numeric(value, errors="coerce")
    if pd.isna(code):
        return None
    return {
        1: "work",
        2: "shopping",
        3: "social_recreation",
        4: "other_home_based",
        5: "non_home_based",
    }.get(int(code))


def load_trip_purposes(source: TripSource) -> pd.DataFrame:
    if not source.path.exists():
        raise FileNotFoundError(f"Missing trip source: {source.path}")
    LOGGER.info("Reading %s trip purposes from %s", source.year, source.path)
    frame = pd.read_csv(source.path, usecols=[HOUSEHOLD_ID, "TRIPPURP"], low_memory=False)
    frame["purpose_category"] = frame["TRIPPURP"].map(lambda value: map_trip_purpose(source.year, value))
    frame = frame.dropna(subset=[HOUSEHOLD_ID, "purpose_category"]).copy()
    frame[HOUSEHOLD_ID] = frame[HOUSEHOLD_ID].astype(str)
    frame[YEAR_COLUMN] = source.year
    return frame[[HOUSEHOLD_ID, YEAR_COLUMN, "purpose_category"]]


def aggregate_household_purposes(trip_frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        trip_frame.groupby([YEAR_COLUMN, HOUSEHOLD_ID, "purpose_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    categories = ["work", "shopping", "social_recreation", "other_home_based", "non_home_based"]
    for category in categories:
        if category not in counts.columns:
            counts[category] = 0
    counts["valid_purpose_trip_count"] = counts[categories].sum(axis=1)
    counts = counts[counts["valid_purpose_trip_count"] > 0].copy()
    for category in categories:
        counts[f"{category}_share"] = counts[category] / counts["valid_purpose_trip_count"]
        counts[f"{category}_trips"] = counts[category]
    keep_columns = [YEAR_COLUMN, HOUSEHOLD_ID, "valid_purpose_trip_count"] + list(PURPOSE_COLUMNS)
    keep_columns += [f"{category}_trips" for category in categories]
    return counts[keep_columns].reset_index(drop=True)


def build_purpose_dataset(household_path: Path) -> pd.DataFrame:
    household = pd.read_csv(household_path, low_memory=False)
    household = household[household[YEAR_COLUMN].isin((2017, 2022))].copy()
    household[HOUSEHOLD_ID] = household[HOUSEHOLD_ID].astype(str)
    trip_purposes = pd.concat([load_trip_purposes(source) for source in TRIP_SOURCES], ignore_index=True)
    targets = aggregate_household_purposes(trip_purposes)
    merged = household.merge(targets, on=[YEAR_COLUMN, HOUSEHOLD_ID], how="inner")
    missing_features = set(FEATURE_COLUMNS).difference(merged.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")
    return merged.reset_index(drop=True)


def normalize_shares(predictions: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictions, 0.0, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = np.divide(clipped, row_sums, out=np.zeros_like(clipped), where=row_sums > 1e-12)
    empty_rows = np.where(row_sums.reshape(-1) <= 1e-12)[0]
    if len(empty_rows) > 0:
        normalized[empty_rows] = fallback
    return normalized


def weighted_mean_prediction(train_frame: pd.DataFrame, rows: int) -> np.ndarray:
    weights = train_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    means = [weighted_average(train_frame[column].to_numpy(dtype=float), weights) for column in PURPOSE_COLUMNS]
    mean_array = np.asarray(means, dtype=float)
    mean_array = mean_array / mean_array.sum()
    return np.tile(mean_array, (rows, 1))


def fit_xgboost_share_models(frame: pd.DataFrame, n_estimators: int, device: str) -> np.ndarray:
    train_frame = frame[frame[YEAR_COLUMN] == 2017].copy()
    test_frame = frame[frame[YEAR_COLUMN] == 2022].copy()
    numeric_features = [column for column in BASE_NUMERIC_FEATURES if column in frame.columns]
    categorical_features = [column for column in BASE_CATEGORICAL_FEATURES if column in frame.columns]
    predictions: list[np.ndarray] = []
    for target in PURPOSE_COLUMNS:
        LOGGER.info("Training purpose-share model for %s on %s rows.", target, len(train_frame))
        model = create_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_estimators=n_estimators,
            device=device,
            random_state=RANDOM_SEED,
        )
        model.fit(
            train_frame[numeric_features + categorical_features],
            train_frame[target],
            model__sample_weight=train_frame[WEIGHT_COLUMN],
        )
        predictions.append(model.predict(test_frame[numeric_features + categorical_features]))
    fallback = weighted_mean_prediction(train_frame, len(test_frame))
    return normalize_shares(np.vstack(predictions).T, fallback[0])


def load_llm_event_features(frame_2022: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    llm_frame = load_2022_with_llm_features(args.household_path, args.cohort_profiles_path, args.llm_features_path)
    llm_frame[HOUSEHOLD_ID] = llm_frame[HOUSEHOLD_ID].astype(str)
    columns = [
        HOUSEHOLD_ID,
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
    ]
    merged = frame_2022[[HOUSEHOLD_ID]].merge(llm_frame[columns], on=HOUSEHOLD_ID, how="left")
    for column in columns[1:]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return merged


def purpose_pressure_matrix(features: pd.DataFrame) -> np.ndarray:
    trip = features["trip_suppression_risk"].to_numpy(dtype=float)
    remote = features["remote_work_substitution_likelihood"].to_numpy(dtype=float)
    delivery = features["online_delivery_substitution_likelihood"].to_numpy(dtype=float)
    recovery = features["post_pandemic_recovery_sensitivity"].to_numpy(dtype=float)
    return np.vstack(
        [
            remote,
            delivery,
            np.clip(0.6 * trip * (1.0 - 0.25 * recovery), 0.0, 1.0),
            np.clip(0.35 * trip, 0.0, 1.0),
            np.clip(0.25 * trip, 0.0, 1.0),
        ]
    ).T


def apply_purpose_priors(
    base_predictions: np.ndarray,
    pressure_matrix: np.ndarray,
    alpha: float,
    min_factor: float,
) -> np.ndarray:
    adapted = base_predictions * np.clip(1.0 - alpha * pressure_matrix, min_factor, 1.0)
    fallback = np.mean(base_predictions, axis=0)
    fallback = fallback / fallback.sum()
    return normalize_shares(adapted, fallback)


def evaluate_predictions(
    method: str,
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    device: str,
) -> dict[str, float | int | str]:
    truth = test_frame[list(PURPOSE_COLUMNS)].to_numpy(dtype=float)
    weights = test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - truth
    abs_error = np.abs(error)
    total_variation = 0.5 * abs_error.sum(axis=1)
    true_purpose = np.argmax(truth, axis=1)
    pred_purpose = np.argmax(predictions, axis=1)
    row: dict[str, float | int | str] = {
        "method": method,
        "train_year": 2017,
        "test_year": 2022,
        "test_rows": int(len(test_frame)),
        "device": device,
        "weighted_total_variation": weighted_average(total_variation, weights),
        "weighted_mean_share_mae": weighted_average(abs_error.mean(axis=1), weights),
        "weighted_dominant_purpose_accuracy": weighted_average((true_purpose == pred_purpose).astype(float), weights),
    }
    for index, column in enumerate(PURPOSE_COLUMNS):
        row[f"{column}_weighted_mae"] = weighted_average(abs_error[:, index], weights)
        row[f"{column}_weighted_bias"] = weighted_average(error[:, index], weights)
    return row


def distribution_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, year_frame in frame.groupby(YEAR_COLUMN):
        weights = year_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
        row: dict[str, float | int] = {
            "survey_year": int(year),
            "households_with_trips": int(len(year_frame)),
            "weighted_valid_purpose_trip_count": weighted_average(
                year_frame["valid_purpose_trip_count"].to_numpy(dtype=float),
                weights,
            ),
        }
        for column in PURPOSE_COLUMNS:
            row[column] = weighted_average(year_frame[column].to_numpy(dtype=float), weights)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("survey_year")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def save_prediction_frame(
    test_frame: pd.DataFrame,
    predictions_by_method: dict[str, np.ndarray],
    path: Path,
) -> None:
    output = test_frame[[HOUSEHOLD_ID, WEIGHT_COLUMN, "valid_purpose_trip_count", *PURPOSE_COLUMNS]].copy()
    for method, prediction in predictions_by_method.items():
        for index, column in enumerate(PURPOSE_COLUMNS):
            output[f"{method}_{column}"] = prediction[:, index]
    save_csv(output, path)


def save_distribution_figure(distribution: pd.DataFrame, output_dir: Path) -> Path:
    figure_dir = output_dir / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "purpose_distribution_shift.png"
    labels = [PURPOSE_LABELS[column] for column in PURPOSE_COLUMNS]
    x = np.arange(len(PURPOSE_COLUMNS))
    width = 0.36
    dist = distribution.set_index("survey_year")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, [dist.loc[2017, column] for column in PURPOSE_COLUMNS], width, label="2017")
    ax.bar(x + width / 2, [dist.loc[2022, column] for column in PURPOSE_COLUMNS], width, label="2022")
    ax.set_title("Weighted Household Purpose Composition")
    ax.set_ylabel("Average household purpose share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metrics_figure(metrics: pd.DataFrame, output_dir: Path) -> Path:
    figure_dir = output_dir / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "purpose_metric_comparison.png"
    methods = ["historical_mean_2017", "historical_xgboost", "llm_purpose_prior_a1", "global_purpose_prior_a1"]
    labels = ["2017 mean", "Traditional XGBoost", "LLM purpose prior", "Global purpose prior"]
    plot_frame = metrics.set_index("method").loc[methods].reset_index()
    x = np.arange(len(plot_frame))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(x, plot_frame["weighted_total_variation"], color="#3B82F6")
    axes[0].set_title("Purpose Composition Error")
    axes[0].set_ylabel("Weighted total variation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].bar(x, plot_frame["work_share_weighted_mae"], color="#F97316")
    axes[1].set_title("Work Share Error")
    axes[1].set_ylabel("Weighted MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(metrics: pd.DataFrame, distribution: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "purpose_composition_report.md"
    baseline = metrics[metrics["method"] == "historical_xgboost"].iloc[0]
    best = metrics.sort_values("weighted_total_variation").iloc[0]
    best_gain = 100.0 * (
        baseline.weighted_total_variation - best.weighted_total_variation
    ) / baseline.weighted_total_variation
    lines = [
        "# Household Purpose Composition Extension",
        "",
        "## Scope",
        "",
        "This extension predicts household-level trip-purpose shares. It adds a third behavior dimension beyond trip generation and mode composition.",
        "",
        "## Purpose Mapping",
        "",
        "- 2017 `TRIPPURP`: `HBW`, `HBSHOP`, `HBSOCREC`, `HBO`, `NHB`.",
        "- 2022 `TRIPPURP`: `1`, `2`, `3`, `4`, `5`, mapped to the same five broad categories using cross-tabs with `WHYTRP90` and `WHYTRP1S`.",
        "",
        "## Weighted Purpose Distribution",
        "",
        "| Year | Households | Trips/HH | Work | Shopping | Social/Rec | Other HB | NHB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.itertuples(index=False):
        lines.append(
            f"| {row.survey_year} | {row.households_with_trips} | "
            f"{row.weighted_valid_purpose_trip_count:.2f} | {row.work_share:.3f} | "
            f"{row.shopping_share:.3f} | {row.social_recreation_share:.3f} | "
            f"{row.other_home_based_share:.3f} | {row.non_home_based_share:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main Metrics",
            "",
            "| Method | Weighted TV | Mean share MAE | Dominant purpose acc. | Work MAE | Shopping MAE | Social MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_total_variation:.4f} | "
            f"{row.weighted_mean_share_mae:.4f} | {row.weighted_dominant_purpose_accuracy:.4f} | "
            f"{row.work_share_weighted_mae:.4f} | {row.shopping_share_weighted_mae:.4f} | "
            f"{row.social_recreation_share_weighted_mae:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Traditional XGBoost weighted TV: `{baseline.weighted_total_variation:.4f}`.",
            f"- Best reported row: `{best.method}`, weighted TV `{best.weighted_total_variation:.4f}`.",
            f"- Best row changes weighted TV by `{best_gain:.2f}%` vs traditional XGBoost.",
            "- This purpose task is useful for scope expansion even if the LLM prior is not the main source of improvement.",
            "",
            "## Figures",
            "",
            "- `figures/purpose_distribution_shift.png`",
            "- `figures/purpose_metric_comparison.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    alphas = parse_float_grid(args.alphas)
    dataset = build_purpose_dataset(args.household_path)
    save_csv(dataset, output_dir / "household_purpose_composition.csv")
    distribution = distribution_by_year(dataset)
    save_csv(distribution, output_dir / "purpose_distribution_by_year.csv")

    train_frame = dataset[dataset[YEAR_COLUMN] == 2017].copy()
    test_frame = dataset[dataset[YEAR_COLUMN] == 2022].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("Purpose dataset must include both 2017 training rows and 2022 test rows.")

    mean_prediction = weighted_mean_prediction(train_frame, len(test_frame))
    xgb_prediction = fit_xgboost_share_models(dataset, args.n_estimators, device)
    metrics = [
        evaluate_predictions("historical_mean_2017", test_frame, mean_prediction, "none"),
        evaluate_predictions("historical_xgboost", test_frame, xgb_prediction, device),
    ]
    predictions_by_method: dict[str, np.ndarray] = {
        "historical_mean_2017": mean_prediction,
        "historical_xgboost": xgb_prediction,
    }

    features = load_llm_event_features(test_frame, args)
    pressure = purpose_pressure_matrix(features)
    global_pressure = np.tile(
        np.average(pressure, axis=0, weights=test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)),
        (len(test_frame), 1),
    )
    for alpha in alphas:
        token = alpha_token(alpha)
        llm_prediction = apply_purpose_priors(xgb_prediction, pressure, alpha, args.min_factor)
        global_prediction = apply_purpose_priors(xgb_prediction, global_pressure, alpha, args.min_factor)
        metrics.append(evaluate_predictions(f"llm_purpose_prior_a{token}", test_frame, llm_prediction, device))
        metrics.append(evaluate_predictions(f"global_purpose_prior_a{token}", test_frame, global_prediction, device))
        predictions_by_method[f"llm_purpose_prior_a{token}"] = llm_prediction
        predictions_by_method[f"global_purpose_prior_a{token}"] = global_prediction

    metrics_frame = pd.DataFrame(metrics)
    save_csv(metrics_frame, output_dir / "purpose_composition_metrics.csv")
    save_prediction_frame(test_frame, predictions_by_method, output_dir / "purpose_composition_predictions.csv")
    save_distribution_figure(distribution, output_dir)
    save_metrics_figure(metrics_frame, output_dir)
    report_path = write_report(metrics_frame, distribution, output_dir)
    LOGGER.info("Wrote purpose-composition report: %s", report_path)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_plot_style()
    run_experiment(args)


if __name__ == "__main__":
    main()
