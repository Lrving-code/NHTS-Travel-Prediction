"""Run household-level mode-composition transfer experiments."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_household_baseline import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    RANDOM_SEED,
    WEIGHT_COLUMN,
)
from run_llm_residual_adaptation import (
    create_pipeline,
    load_2022_with_llm_features,
    resolve_device,
    weighted_average,
)


LOGGER = logging.getLogger(__name__)
HOUSEHOLD_ID = "HOUSEID"
YEAR_COLUMN = "survey_year"
MODE_COLUMNS: tuple[str, ...] = (
    "private_vehicle_share",
    "walk_share",
    "bike_share",
    "transit_share",
    "taxi_ridehail_share",
    "other_share",
)
MODE_LABELS: dict[str, str] = {
    "private_vehicle_share": "Private vehicle",
    "walk_share": "Walk",
    "bike_share": "Bike",
    "transit_share": "Transit",
    "taxi_ridehail_share": "Taxi / ridehail",
    "other_share": "Other",
}
FEATURE_COLUMNS = tuple(BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES)
FIGURE_DIR_NAME = "figures"


@dataclass(frozen=True)
class ModeSource:
    """Trip-level source table used to build household mode targets."""

    year: int
    path: Path


TRIP_SOURCES: tuple[ModeSource, ...] = (
    ModeSource(2017, Path("data/raw/nhts_2017/csv/trippub.csv")),
    ModeSource(2022, Path("data/raw/nhts_2022/csv/tripv2pub.csv")),
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mode_composition_extension"))
    parser.add_argument(
        "--trip-predictions-path",
        type=Path,
        default=Path("outputs/models/final_label_free_2022/final_2022_predictions.csv"),
    )
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--alphas", default="0.50,0.75,1.00")
    parser.add_argument("--min-transit-factor", type=float, default=0.05)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


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


def map_2017_trptrans(value: object) -> str | None:
    code = pd.to_numeric(value, errors="coerce")
    if pd.isna(code):
        return None
    code_int = int(code)
    if code_int == 1:
        return "walk"
    if code_int == 2:
        return "bike"
    if code_int in {3, 4, 5, 6, 7, 8, 9, 18}:
        return "private_vehicle"
    if code_int == 17:
        return "taxi_ridehail"
    if code_int in {11, 12, 13, 14, 15, 16, 20}:
        return "transit"
    if code_int in {-9, -8, -7, 10, 19, 97, 98, 99}:
        return "other"
    return "other"


def map_2022_trptrans(value: object) -> str | None:
    code = pd.to_numeric(value, errors="coerce")
    if pd.isna(code):
        return None
    code_int = int(code)
    if code_int in {1, 2, 3, 4, 6, 7}:
        return "private_vehicle"
    if code_int in {15, 16}:
        return "taxi_ridehail"
    if code_int == 20:
        return "walk"
    if code_int == 18:
        return "bike"
    if code_int in {8, 10, 11, 12, 13, 17}:
        return "transit"
    if code_int in {9, 14, 19, 21}:
        return "other"
    return "other"


def load_trip_modes(source: ModeSource) -> pd.DataFrame:
    if not source.path.exists():
        raise FileNotFoundError(f"Missing trip source: {source.path}")
    usecols = [HOUSEHOLD_ID, "TRPTRANS"]
    LOGGER.info("Reading %s trip modes from %s", source.year, source.path)
    frame = pd.read_csv(source.path, usecols=usecols, low_memory=False)
    if source.year == 2022:
        frame["mode_category"] = frame["TRPTRANS"].map(map_2022_trptrans)
    else:
        frame["mode_category"] = frame["TRPTRANS"].map(map_2017_trptrans)
    frame = frame.dropna(subset=[HOUSEHOLD_ID, "mode_category"]).copy()
    frame[HOUSEHOLD_ID] = frame[HOUSEHOLD_ID].astype(str)
    frame[YEAR_COLUMN] = source.year
    return frame[[HOUSEHOLD_ID, YEAR_COLUMN, "mode_category"]]


def aggregate_household_modes(trip_frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        trip_frame.groupby([YEAR_COLUMN, HOUSEHOLD_ID, "mode_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for category in ("private_vehicle", "walk", "bike", "transit", "taxi_ridehail", "other"):
        if category not in counts.columns:
            counts[category] = 0
    mode_count_columns = ["private_vehicle", "walk", "bike", "transit", "taxi_ridehail", "other"]
    counts["valid_trip_count"] = counts[mode_count_columns].sum(axis=1)
    counts = counts[counts["valid_trip_count"] > 0].copy()
    for category in mode_count_columns:
        counts[f"{category}_share"] = counts[category] / counts["valid_trip_count"]
        counts[f"{category}_trips"] = counts[category]
    keep_columns = [YEAR_COLUMN, HOUSEHOLD_ID, "valid_trip_count"] + list(MODE_COLUMNS)
    keep_columns += [f"{category}_trips" for category in mode_count_columns]
    return counts[keep_columns].reset_index(drop=True)


def build_mode_dataset(household_path: Path) -> pd.DataFrame:
    household = pd.read_csv(household_path, low_memory=False)
    household = household[household[YEAR_COLUMN].isin((2017, 2022))].copy()
    household[HOUSEHOLD_ID] = household[HOUSEHOLD_ID].astype(str)
    trip_modes = pd.concat([load_trip_modes(source) for source in TRIP_SOURCES], ignore_index=True)
    targets = aggregate_household_modes(trip_modes)
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


def fit_xgboost_share_models(frame: pd.DataFrame, n_estimators: int, device: str) -> np.ndarray:
    train_frame = frame[frame[YEAR_COLUMN] == 2017].copy()
    test_frame = frame[frame[YEAR_COLUMN] == 2022].copy()
    numeric_features = [column for column in BASE_NUMERIC_FEATURES if column in frame.columns]
    categorical_features = [column for column in BASE_CATEGORICAL_FEATURES if column in frame.columns]
    predictions: list[np.ndarray] = []
    for target in MODE_COLUMNS:
        LOGGER.info("Training mode-share model for %s on %s rows.", target, len(train_frame))
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
    prediction_matrix = np.vstack(predictions).T
    fallback = weighted_mean_prediction(train_frame, len(test_frame))
    return normalize_shares(prediction_matrix, fallback[0])


def weighted_mean_prediction(train_frame: pd.DataFrame, rows: int) -> np.ndarray:
    weights = train_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    means = [
        weighted_average(train_frame[column].to_numpy(dtype=float), weights)
        for column in MODE_COLUMNS
    ]
    mean_array = np.asarray(means, dtype=float)
    mean_array = mean_array / mean_array.sum()
    return np.tile(mean_array, (rows, 1))


def load_llm_pressure(frame_2022: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    llm_frame = load_2022_with_llm_features(
        args.household_path,
        args.cohort_profiles_path,
        args.llm_features_path,
    )
    llm_frame[HOUSEHOLD_ID] = llm_frame[HOUSEHOLD_ID].astype(str)
    merged = frame_2022[[HOUSEHOLD_ID]].merge(
        llm_frame[[HOUSEHOLD_ID, "transit_avoidance_likelihood"]],
        on=HOUSEHOLD_ID,
        how="left",
    )
    pressure = pd.to_numeric(merged["transit_avoidance_likelihood"], errors="coerce").fillna(0.0)
    return np.clip(pressure.to_numpy(dtype=float), 0.0, 1.0)


def apply_transit_avoidance(
    base_predictions: np.ndarray,
    pressure: np.ndarray,
    alpha: float,
    min_transit_factor: float,
) -> np.ndarray:
    adapted = base_predictions.copy()
    transit_index = MODE_COLUMNS.index("transit_share")
    factor = np.clip(1.0 - alpha * pressure, min_transit_factor, 1.0)
    adapted[:, transit_index] = adapted[:, transit_index] * factor
    fallback = np.mean(base_predictions, axis=0)
    fallback = fallback / fallback.sum()
    return normalize_shares(adapted, fallback)


def evaluate_mode_predictions(
    method: str,
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    device: str,
) -> dict[str, float | int | str]:
    truth = test_frame[list(MODE_COLUMNS)].to_numpy(dtype=float)
    weights = test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
    error = predictions - truth
    abs_error = np.abs(error)
    per_household_mae = abs_error.mean(axis=1)
    total_variation = 0.5 * abs_error.sum(axis=1)
    true_mode = np.argmax(truth, axis=1)
    pred_mode = np.argmax(predictions, axis=1)
    correct = (true_mode == pred_mode).astype(float)
    row: dict[str, float | int | str] = {
        "method": method,
        "train_year": 2017,
        "test_year": 2022,
        "test_rows": int(len(test_frame)),
        "device": device,
        "unweighted_mean_share_mae": float(np.mean(per_household_mae)),
        "weighted_mean_share_mae": weighted_average(per_household_mae, weights),
        "unweighted_total_variation": float(np.mean(total_variation)),
        "weighted_total_variation": weighted_average(total_variation, weights),
        "dominant_mode_accuracy": float(np.mean(correct)),
        "weighted_dominant_mode_accuracy": weighted_average(correct, weights),
    }
    for index, column in enumerate(MODE_COLUMNS):
        row[f"{column}_mae"] = float(np.mean(abs_error[:, index]))
        row[f"{column}_weighted_mae"] = weighted_average(abs_error[:, index], weights)
        row[f"{column}_bias"] = float(np.mean(error[:, index]))
        row[f"{column}_weighted_bias"] = weighted_average(error[:, index], weights)
    return row


def mode_distribution_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for year, year_frame in frame.groupby(YEAR_COLUMN):
        weights = year_frame[WEIGHT_COLUMN].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "survey_year": int(year),
            "households_with_trips": int(len(year_frame)),
            "weighted_valid_trip_count": weighted_average(
                year_frame["valid_trip_count"].to_numpy(dtype=float),
                weights,
            ),
        }
        for column in MODE_COLUMNS:
            row[column] = weighted_average(year_frame[column].to_numpy(dtype=float), weights)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("survey_year")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def make_mode_prediction_frame(
    test_frame: pd.DataFrame,
    predictions_by_method: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = test_frame[
        [HOUSEHOLD_ID, WEIGHT_COLUMN, "valid_trip_count", *MODE_COLUMNS]
        + [column.replace("_share", "_trips") for column in MODE_COLUMNS]
    ].copy()
    for method, prediction in predictions_by_method.items():
        for index, column in enumerate(MODE_COLUMNS):
            output[f"{method}_{column}"] = prediction[:, index]
    return output


def load_trip_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trip prediction file: {path}")
    columns = [
        HOUSEHOLD_ID,
        "prediction_historical_xgboost",
        "prediction_gated_trip_suppression_a1_d0p15",
        "prediction_llm_trip_suppression_a1p25",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame[HOUSEHOLD_ID] = frame[HOUSEHOLD_ID].astype(str)
    return frame


def evaluate_mode_trip_counts(
    test_frame: pd.DataFrame,
    mode_predictions: dict[str, np.ndarray],
    trip_predictions: pd.DataFrame,
) -> pd.DataFrame:
    joined = test_frame[[HOUSEHOLD_ID, WEIGHT_COLUMN, "valid_trip_count"]].copy()
    for column in [column.replace("_share", "_trips") for column in MODE_COLUMNS]:
        joined[column] = test_frame[column].to_numpy(dtype=float)
    joined = joined.merge(trip_predictions, on=HOUSEHOLD_ID, how="inner", validate="one_to_one")
    rows: list[dict[str, float | str | int]] = []
    truth = joined[[column.replace("_share", "_trips") for column in MODE_COLUMNS]].to_numpy(dtype=float)
    weights = joined[WEIGHT_COLUMN].to_numpy(dtype=float)
    combos = {
        "traditional_count_x_traditional_mode": (
            "prediction_historical_xgboost",
            "historical_xgboost",
        ),
        "gated_count_x_traditional_mode": (
            "prediction_gated_trip_suppression_a1_d0p15",
            "historical_xgboost",
        ),
        "gated_count_x_llm_mode": (
            "prediction_gated_trip_suppression_a1_d0p15",
            "llm_transit_avoidance_a1",
        ),
        "best_count_x_llm_mode": (
            "prediction_llm_trip_suppression_a1p25",
            "llm_transit_avoidance_a1",
        ),
    }
    for method, (trip_column, mode_method) in combos.items():
        if mode_method not in mode_predictions:
            continue
        total = joined[trip_column].to_numpy(dtype=float).reshape(-1, 1)
        predicted = np.maximum(total * mode_predictions[mode_method], 0.0)
        error = predicted - truth
        abs_error = np.abs(error)
        row: dict[str, float | str | int] = {
            "method": method,
            "rows": int(len(joined)),
            "weighted_total_mode_trip_mae": weighted_average(abs_error.sum(axis=1), weights),
            "weighted_mean_mode_trip_mae": weighted_average(abs_error.mean(axis=1), weights),
            "weighted_total_trip_bias": weighted_average(error.sum(axis=1), weights),
        }
        for index, column in enumerate(MODE_COLUMNS):
            base = column.replace("_share", "")
            row[f"{base}_trip_weighted_mae"] = weighted_average(abs_error[:, index], weights)
            row[f"{base}_trip_weighted_bias"] = weighted_average(error[:, index], weights)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    trip_count_metrics: pd.DataFrame,
    output_dir: Path,
) -> Path:
    path = output_dir / "mode_composition_report.md"
    baseline = metrics[metrics["method"] == "historical_xgboost"].iloc[0]
    mean_row = metrics[metrics["method"] == "historical_mean_2017"].iloc[0]
    best = metrics.sort_values("weighted_total_variation").iloc[0]
    best_tv_gain = 100.0 * (
        baseline.weighted_total_variation - best.weighted_total_variation
    ) / baseline.weighted_total_variation
    best_transit_gain = 100.0 * (
        baseline.transit_share_weighted_mae - best.transit_share_weighted_mae
    ) / baseline.transit_share_weighted_mae
    lines = [
        "# Household Mode Composition Extension",
        "",
        "## Scope",
        "",
        "This extension predicts household-level 2022 mode shares derived from trip records. "
        "It is separate from the main `CNTTDHH` trip-count task and should not be compared with "
        "trip-level `TRPTRANS` classification accuracy.",
        "",
        "## Dataset",
        "",
        "- Training year: `2017` households with at least one valid trip.",
        "- Test year: `2022` households with at least one valid trip.",
        "- Targets: private vehicle, walk, bike, transit, taxi/ridehail, and other shares.",
        "- 2017 and 2022 use year-specific official `TRPTRANS` codebook mappings into common broad categories.",
        "",
        "## Weighted Mode Distribution",
        "",
        "| Year | Households | Trips/HH | Private | Walk | Bike | Transit | Taxi/Ridehail | Other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distribution.itertuples(index=False):
        lines.append(
            f"| {row.survey_year} | {row.households_with_trips} | {row.weighted_valid_trip_count:.2f} | "
            f"{row.private_vehicle_share:.3f} | {row.walk_share:.3f} | {row.bike_share:.3f} | "
            f"{row.transit_share:.3f} | {row.taxi_ridehail_share:.3f} | {row.other_share:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Main Metrics",
            "",
            "| Method | Weighted TV | Weighted Mean Share MAE | Weighted Dominant Accuracy | Transit Weighted MAE | Transit Weighted Bias |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if not trip_count_metrics.empty:
        lines.extend(
            [
                "",
                "## Derived Mode-Specific Trip Volumes",
                "",
                "Mode shares can be combined with trip-count predictions to estimate trips by mode. "
                "This turns the project into a household travel-behavior system rather than a single-output regression task.",
                "",
                "| Method | Total mode-trip MAE | Mean component MAE | Total trip bias | Transit trip MAE | Active trip MAE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in trip_count_metrics.itertuples(index=False):
            active_mae = row.walk_trip_weighted_mae + row.bike_trip_weighted_mae
            lines.append(
                f"| {row.method} | {row.weighted_total_mode_trip_mae:.4f} | "
                f"{row.weighted_mean_mode_trip_mae:.4f} | {row.weighted_total_trip_bias:.4f} | "
                f"{row.transit_trip_weighted_mae:.4f} | {active_mae:.4f} |"
            )
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_total_variation:.4f} | "
            f"{row.weighted_mean_share_mae:.4f} | {row.weighted_dominant_mode_accuracy:.4f} | "
            f"{row.transit_share_weighted_mae:.4f} | {row.transit_share_weighted_bias:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Historical XGBoost weighted total variation: `{baseline.weighted_total_variation:.4f}`.",
            f"- Historical mean weighted total variation: `{mean_row.weighted_total_variation:.4f}`.",
            f"- Best reported row by weighted total variation: `{best.method}`.",
            f"- Best row reduces weighted total variation by `{best_tv_gain:.2f}%` vs historical XGBoost.",
            f"- Best row reduces transit-share weighted MAE by `{best_transit_gain:.2f}%` vs historical XGBoost.",
            "",
            "## Figures",
            "",
            f"- `figures/mode_distribution_shift.png`",
            f"- `figures/mode_metric_comparison.png`",
            "",
            "## Caveat",
            "",
            "Treat this as an exploratory extension. The main course-project story is broader: trip generation, mode composition, and derived mode-specific trip volumes.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_distribution_figure(distribution: pd.DataFrame, output_dir: Path) -> Path:
    figure_dir = output_dir / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "mode_distribution_shift.png"
    labels = [MODE_LABELS[column] for column in MODE_COLUMNS]
    x = np.arange(len(MODE_COLUMNS))
    width = 0.36
    dist = distribution.set_index("survey_year")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [dist.loc[2017, column] for column in MODE_COLUMNS], width, label="2017")
    ax.bar(x + width / 2, [dist.loc[2022, column] for column in MODE_COLUMNS], width, label="2022")
    ax.set_title("Weighted Household Mode Composition")
    ax.set_ylabel("Average household mode share")
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
    path = figure_dir / "mode_metric_comparison.png"
    methods = [
        "historical_mean_2017",
        "historical_xgboost",
        "llm_transit_avoidance_a1",
        "global_transit_avoidance_a1",
    ]
    plot_frame = metrics.set_index("method").loc[methods].reset_index()
    labels = ["2017 mean", "Historical XGBoost", "LLM transit prior", "Global transit prior"]
    x = np.arange(len(plot_frame))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(x, plot_frame["weighted_total_variation"], color="#3B82F6")
    axes[0].set_title("Composition Error")
    axes[0].set_ylabel("Weighted total variation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].bar(x, plot_frame["transit_share_weighted_mae"], color="#F97316")
    axes[1].set_title("Transit Share Error")
    axes[1].set_ylabel("Weighted MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    alphas = parse_float_grid(args.alphas)
    dataset = build_mode_dataset(args.household_path)
    save_csv(dataset, output_dir / "household_mode_composition.csv")
    distribution = mode_distribution_by_year(dataset)
    save_csv(distribution, output_dir / "mode_distribution_by_year.csv")

    train_frame = dataset[dataset[YEAR_COLUMN] == 2017].copy()
    test_frame = dataset[dataset[YEAR_COLUMN] == 2022].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("Mode dataset must include both 2017 training rows and 2022 test rows.")

    LOGGER.info("Training rows: %s; test rows: %s", len(train_frame), len(test_frame))
    mean_prediction = weighted_mean_prediction(train_frame, len(test_frame))
    xgb_prediction = fit_xgboost_share_models(dataset, args.n_estimators, device)

    predictions_by_method: dict[str, np.ndarray] = {
        "historical_mean_2017": mean_prediction,
        "historical_xgboost": xgb_prediction,
    }
    metrics = [
        evaluate_mode_predictions("historical_mean_2017", test_frame, mean_prediction, "none"),
        evaluate_mode_predictions("historical_xgboost", test_frame, xgb_prediction, device),
    ]
    pressure = load_llm_pressure(test_frame, args)
    global_pressure = np.full_like(pressure, weighted_average(pressure, test_frame[WEIGHT_COLUMN].to_numpy(dtype=float)))
    for alpha in alphas:
        token = alpha_token(alpha)
        llm_prediction = apply_transit_avoidance(
            xgb_prediction,
            pressure,
            alpha,
            args.min_transit_factor,
        )
        global_prediction = apply_transit_avoidance(
            xgb_prediction,
            global_pressure,
            alpha,
            args.min_transit_factor,
        )
        metrics.append(evaluate_mode_predictions(f"llm_transit_avoidance_a{token}", test_frame, llm_prediction, device))
        metrics.append(
            evaluate_mode_predictions(f"global_transit_avoidance_a{token}", test_frame, global_prediction, device)
        )
        predictions_by_method[f"llm_transit_avoidance_a{token}"] = llm_prediction
        predictions_by_method[f"global_transit_avoidance_a{token}"] = global_prediction

    metrics_frame = pd.DataFrame(metrics)
    save_csv(metrics_frame, output_dir / "mode_composition_metrics.csv")
    prediction_frame = make_mode_prediction_frame(test_frame, predictions_by_method)
    save_csv(prediction_frame, output_dir / "mode_composition_predictions.csv")
    trip_count_metrics = evaluate_mode_trip_counts(
        test_frame,
        predictions_by_method,
        load_trip_predictions(args.trip_predictions_path),
    )
    save_csv(trip_count_metrics, output_dir / "mode_specific_trip_count_metrics.csv")
    save_distribution_figure(distribution, output_dir)
    save_metrics_figure(metrics_frame, output_dir)
    report_path = write_report(metrics_frame, distribution, trip_count_metrics, output_dir)
    LOGGER.info("Wrote mode-composition report: %s", report_path)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_plot_style()
    run_experiment(args)


if __name__ == "__main__":
    main()
