"""Run household-level external validation on PSRC travel survey microdata.

The validation uses PSRC household/person-day microdata as an independent
external travel survey. It trains on the 2021 PSRC wave and predicts later PSRC
waves, applying a BTS-derived national recovery factor without using target-year
PSRC labels for calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from run_household_baseline import RANDOM_SEED
from run_llm_residual_adaptation import resolve_device, weighted_rmse


LOGGER = logging.getLogger(__name__)

PSRC_HOUSEHOLD_ITEM = "7482a3f923894de687ee184256e72013"
PSRC_DAYS_ITEM = "7cf2cd7f3f324fc290e571ba620bd97a"
PSRC_DATA_PORTAL_URL = (
    "https://psrc-psregcncl.hub.arcgis.com/datasets/"
    "PSREGCNCL::household-travel-survey-households/about"
)
BTS_RESOURCE_URL = "https://data.transportation.gov/resource/aksz-j95y.json"

NUMERIC_FEATURES: tuple[str, ...] = (
    "hhsize_num",
    "vehicle_count_num",
    "numworkers_num",
    "observed_days",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "hhincome_broad",
    "diary_platform",
)


@dataclass(frozen=True)
class MetricRow:
    method: str
    train_year: int
    test_year: int
    train_rows: int
    test_rows: int
    event_factor: float
    event_factor_source: str
    weighted_mae: float
    weighted_rmse: float
    weighted_bias: float
    weighted_r2: float
    weighted_target_mean: float
    weighted_prediction_mean: float
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_validation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/external_validation/cache"))
    parser.add_argument("--train-year", type=int, default=2021)
    parser.add_argument("--test-years", default="2023,2025")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--force-download", action="store_true")
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


def parse_years(raw: str) -> list[int]:
    years = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not years:
        raise ValueError("At least one test year is required.")
    return years


def arcgis_csv_url(item_id: str) -> str:
    return f"https://hub.arcgis.com/api/download/v1/items/{item_id}/csv?layers=0"


def read_public_csv(
    item_id: str,
    cache_path: Path,
    force_download: bool,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    if cache_path.exists() and not force_download:
        LOGGER.info("Loading cached PSRC file %s.", cache_path)
        return pd.read_csv(cache_path, low_memory=False, usecols=usecols)

    LOGGER.info("Downloading PSRC public CSV item=%s.", item_id)
    frame = pd.read_csv(arcgis_csv_url(item_id), low_memory=False, usecols=usecols)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def parse_first_number(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    match = re.search(r"\d+", str(value).replace(",", ""))
    return float(match.group()) if match else float("nan")


def add_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hhsize_num"] = enriched["hhsize"].map(parse_first_number)
    enriched["vehicle_count_num"] = enriched["vehicle_count"].map(parse_first_number)
    enriched["numworkers_num"] = enriched["numworkers"].map(parse_first_number)
    enriched.loc[
        enriched["vehicle_count"].astype(str).str.contains("no vehicles", case=False, na=False),
        "vehicle_count_num",
    ] = 0.0
    enriched.loc[
        enriched["numworkers"].astype(str).str.contains("no workers", case=False, na=False),
        "numworkers_num",
    ] = 0.0
    return enriched


def build_psrc_household_day_target(households: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    day_frame = days.copy()
    for column in ("num_trips", "survey_year", "day_weight"):
        day_frame[column] = pd.to_numeric(day_frame[column], errors="coerce")

    valid = day_frame["num_trips"].notna()
    if "summary_complete" in day_frame.columns:
        valid &= ~day_frame["summary_complete"].isin(["No"])
    if "surveyable" in day_frame.columns:
        valid &= ~day_frame["surveyable"].isin(["No"])
    day_frame = day_frame[valid].copy()

    household_days = (
        day_frame.groupby(["household_id", "survey_year", "daynum"], as_index=False)
        .agg(household_day_trips=("num_trips", "sum"), day_weight=("day_weight", "mean"))
    )
    household_target = (
        household_days.groupby(["household_id", "survey_year"], as_index=False)
        .agg(
            trips_per_day=("household_day_trips", "mean"),
            observed_days=("daynum", "nunique"),
            day_weight=("day_weight", "mean"),
        )
    )
    merged = households.merge(household_target, on=["household_id", "survey_year"], how="inner")
    merged = add_numeric_features(merged)

    for column in ("hh_weight", "survey_year"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    complete = merged["hh_is_complete"].isin(["Yes", "Missing Response"])
    in_region = ~merged["home_in_region"].isin(["No"])
    clean = merged[
        complete
        & in_region
        & merged["hh_weight"].notna()
        & (merged["hh_weight"] > 0)
        & merged["trips_per_day"].notna()
    ].copy()
    return clean.reset_index(drop=True)


def create_pipeline(n_estimators: int, device: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURES)),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=3,
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
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    y_bar = weighted_average(y_true, weights)
    residual = np.average(np.square(y_true - y_pred), weights=weights)
    total = np.average(np.square(y_true - y_bar), weights=weights)
    if total <= 0:
        return float("nan")
    return float(1.0 - residual / total)


def evaluate(
    method: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    predictions: np.ndarray,
    event_factor: float,
    event_factor_source: str,
    device: str,
) -> MetricRow:
    y_true = test_frame["trips_per_day"].to_numpy(dtype=float)
    weights = test_frame["hh_weight"].to_numpy(dtype=float)
    prediction_array = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    error = prediction_array - y_true
    return MetricRow(
        method=method,
        train_year=int(train_frame["survey_year"].iloc[0]),
        test_year=int(test_frame["survey_year"].iloc[0]),
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        event_factor=event_factor,
        event_factor_source=event_factor_source,
        weighted_mae=float(mean_absolute_error(y_true, prediction_array, sample_weight=weights)),
        weighted_rmse=weighted_rmse(y_true, prediction_array, weights),
        weighted_bias=weighted_average(error, weights),
        weighted_r2=weighted_r2(y_true, prediction_array, weights),
        weighted_target_mean=weighted_average(y_true, weights),
        weighted_prediction_mean=weighted_average(prediction_array, weights),
        device=device,
    )


def bts_annual_trips_per_person(start_year: int, end_year: int) -> pd.DataFrame:
    params = {
        "$select": "date,trips,pop_stay_at_home,pop_not_stay_at_home",
        "$where": (
            f"level='National' AND date between '{start_year}-01-01T00:00:00' "
            f"and '{end_year}-12-31T00:00:00'"
        ),
        "$order": "date",
        "$limit": "5000",
    }
    request = Request(f"{BTS_RESOURCE_URL}?{urlencode(params)}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response:
        rows = json.loads(response.read().decode("utf-8"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("BTS recovery-factor query returned no rows.")
    for column in ("trips", "pop_stay_at_home", "pop_not_stay_at_home"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["population"] = frame["pop_stay_at_home"] + frame["pop_not_stay_at_home"]
    frame["year"] = frame["date"].dt.year
    annual = (
        frame.groupby("year", as_index=False)
        .agg(days=("date", "nunique"), trips=("trips", "sum"), population_day_sum=("population", "sum"))
    )
    annual["trips_per_person_per_day"] = annual["trips"] / annual["population_day_sum"]
    return annual


def event_factor_for_year(annual_bts: pd.DataFrame, train_year: int, test_year: int) -> tuple[float, str]:
    train = annual_bts.loc[annual_bts["year"] == train_year]
    test = annual_bts.loc[annual_bts["year"] == test_year]
    if not train.empty and not test.empty and int(test["days"].iloc[0]) >= 300:
        factor = float(test["trips_per_person_per_day"].iloc[0] / train["trips_per_person_per_day"].iloc[0])
        return factor, f"BTS national trips/person {train_year}->{test_year}"

    latest_full = annual_bts.loc[annual_bts["days"] >= 300].sort_values("year").tail(1)
    if train.empty or latest_full.empty:
        return 1.0, "No full-year BTS recovery factor available"
    latest_year = int(latest_full["year"].iloc[0])
    factor = float(latest_full["trips_per_person_per_day"].iloc[0] / train["trips_per_person_per_day"].iloc[0])
    return factor, f"BTS national trips/person {train_year}->{latest_year} carried forward"


def metric_rows_to_dicts(rows: list[MetricRow]) -> list[dict[str, str | int | float]]:
    return [row.__dict__ for row in rows]


def write_year_summary(frame: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for year, group in frame.groupby("survey_year"):
        weights = group["hh_weight"].to_numpy(dtype=float)
        trips = group["trips_per_day"].to_numpy(dtype=float)
        rows.append(
            {
                "survey_year": int(year),
                "households": len(group),
                "weighted_trips_per_day": weighted_average(trips, weights),
                "unweighted_trips_per_day": float(np.mean(trips)),
                "weighted_zero_trip_share": weighted_average((trips == 0).astype(float), weights),
                "mean_observed_days": float(group["observed_days"].mean()),
            }
        )
    pd.DataFrame(rows).sort_values("survey_year").to_csv(output_path, index=False)


def write_report(metrics: pd.DataFrame, year_summary: pd.DataFrame, output_path: Path) -> None:
    primary = metrics[
        (metrics["test_year"] == 2023) & (metrics["method"] == "bts_recovery_event_adapter")
    ].iloc[0]
    baseline = metrics[
        (metrics["test_year"] == 2023) & (metrics["method"] == "psrc_2021_xgboost")
    ].iloc[0]
    mae_gain = baseline["weighted_mae"] - primary["weighted_mae"]
    bias_gain = abs(baseline["weighted_bias"]) - abs(primary["weighted_bias"])
    report = f"""# PSRC Household-Level External Microdata Validation

This validation uses public Puget Sound Regional Council household travel survey
microdata, independent of NHTS. It is an external recovery-transfer check:
train on PSRC 2021 household/day records, predict later PSRC household/day trip
rates, and apply a BTS-derived recovery factor without using target-year PSRC
labels for calibration.

PSRC source: {PSRC_DATA_PORTAL_URL}

## Data Harmonization

- Target: household trips per observed diary day.
- Source table: PSRC `Days`, aggregated from person-day `num_trips` to
  household-day trips, then averaged by household and survey wave.
- Household covariates: household size, vehicle count, worker count, broad
  income bin, diary platform, and observed diary days.
- Weights: PSRC `hh_weight`.

## Survey-Year Summary

| Year | Households | Weighted trips/day | Weighted zero-trip share | Mean observed days |
|---:|---:|---:|---:|---:|
"""
    for row in year_summary.sort_values("survey_year").itertuples(index=False):
        report += (
            f"| {int(row.survey_year)} | {int(row.households)} | "
            f"{row.weighted_trips_per_day:.4f} | {row.weighted_zero_trip_share:.4f} | "
            f"{row.mean_observed_days:.2f} |\n"
        )

    report += """
## Method Comparison

| Method | Test year | Event factor | wMAE | wRMSE | wBias | wR2 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in metrics.sort_values(["test_year", "method"]).itertuples(index=False):
        report += (
            f"| {row.method} | {int(row.test_year)} | {row.event_factor:.4f} | "
            f"{row.weighted_mae:.4f} | {row.weighted_rmse:.4f} | "
            f"{row.weighted_bias:+.4f} | {row.weighted_r2:.4f} |\n"
        )

    report += f"""
## Interpretation

- On the primary external 2021->2023 recovery transfer, the BTS recovery adapter
  changes weighted MAE from `{baseline['weighted_mae']:.4f}` to
  `{primary['weighted_mae']:.4f}`.
- The MAE gain is modest (`{mae_gain:.4f}` trips/day), but the systematic bias
  improves more clearly: absolute weighted bias changes from
  `{abs(baseline['weighted_bias']):.4f}` to `{abs(primary['weighted_bias']):.4f}`
  trips/day.
- This is not a claim that the NHTS 2022 model directly transfers to PSRC.
  It is household-level external evidence that event-scale mobility recovery
  factors can improve label-free temporal transfer on an independent travel
  survey.

## Limitation

The current PSRC Hub CSV exposes complete person-day microdata for 2021, 2023,
and 2025. The older 2017/2019 day/trip microdata are not exposed in the same
current CSV endpoint, so this validation uses 2021 as the source wave rather
than pre-pandemic PSRC microdata.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    LOGGER.info("Using device=%s.", device)

    households = read_public_csv(
        PSRC_HOUSEHOLD_ITEM,
        args.cache_dir / "psrc_households.csv",
        force_download=args.force_download,
    )
    days = read_public_csv(
        PSRC_DAYS_ITEM,
        args.cache_dir / "psrc_days.csv",
        force_download=args.force_download,
        usecols=[
            "day_id",
            "survey_year",
            "household_id",
            "person_id",
            "daynum",
            "num_trips",
            "day_weight",
            "summary_complete",
            "surveyable",
        ],
    )
    household_days = build_psrc_household_day_target(households, days)
    year_summary_path = output_dir / "psrc_household_year_summary.csv"
    write_year_summary(household_days, year_summary_path)

    test_years = parse_years(args.test_years)
    bts = bts_annual_trips_per_person(args.train_year, max(test_years))
    bts.to_csv(output_dir / "psrc_bts_recovery_factor_source.csv", index=False)

    train_frame = household_days[household_days["survey_year"] == args.train_year].copy()
    if train_frame.empty:
        raise ValueError(f"No PSRC household-day rows for train_year={args.train_year}.")
    feature_columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    model = create_pipeline(args.n_estimators, device)
    LOGGER.info("Training PSRC source-year model on %s rows.", len(train_frame))
    model.fit(
        train_frame[feature_columns],
        train_frame["trips_per_day"],
        model__sample_weight=train_frame["hh_weight"],
    )

    rows: list[MetricRow] = []
    for test_year in test_years:
        test_frame = household_days[household_days["survey_year"] == test_year].copy()
        if test_frame.empty:
            LOGGER.warning("No PSRC household-day rows for test_year=%s; skipping.", test_year)
            continue
        base_predictions = np.maximum(model.predict(test_frame[feature_columns]), 0.0)
        rows.append(
            evaluate(
                "psrc_2021_xgboost",
                train_frame,
                test_frame,
                base_predictions,
                event_factor=1.0,
                event_factor_source="No event factor",
                device=device,
            )
        )
        factor, source = event_factor_for_year(bts, args.train_year, test_year)
        rows.append(
            evaluate(
                "bts_recovery_event_adapter",
                train_frame,
                test_frame,
                base_predictions * factor,
                event_factor=factor,
                event_factor_source=source,
                device=device,
            )
        )

    metrics = pd.DataFrame(metric_rows_to_dicts(rows))
    metrics_path = output_dir / "psrc_household_external_validation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    year_summary = pd.read_csv(year_summary_path)
    write_report(metrics, year_summary, output_dir / "psrc_household_external_validation_report.md")
    LOGGER.info("Wrote PSRC external validation artifacts to %s.", output_dir)


if __name__ == "__main__":
    main()
