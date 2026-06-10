"""Validate NHTS 2022 aggregate shifts against an external BTS mobility source.

This script does not use external data for training or calibration. It checks
whether the aggregate direction and magnitude implied by NHTS-based predictions
are consistent with independent BTS/UMD daily mobility statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from run_household_baseline import RANDOM_SEED, TARGET_COLUMN, WEIGHT_COLUMN
from run_llm_residual_adaptation import (
    load_2022_with_llm_features,
    resolve_device,
    train_history_model,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BTS_RESOURCE_URL = "https://data.transportation.gov/resource/aksz-j95y.json"
BTS_LANDING_URL = (
    "https://data.bts.gov/Research-and-Statistics/"
    "Daily-Mobility-Statistics-National-and-State/aksz-j95y"
)
ACS_COMMUTING_BRIEF_URL = "https://www2.census.gov/library/publications/2024/demo/acsbr-018.pdf"
BTS_COVID_BEHAVIOR_URL = (
    "https://www.bts.gov/browse-statistical-products-and-data/"
    "covid-related/effects-covid-19-travel-behavior"
)

PRESSURE_WEIGHTS: dict[str, float] = {
    "trip_suppression_risk": 0.35,
    "remote_work_substitution_likelihood": 0.25,
    "transit_avoidance_likelihood": 0.20,
    "online_delivery_substitution_likelihood": 0.20,
}


@dataclass(frozen=True)
class AggregateShift:
    method: str
    reference_year: int
    target_year: int
    reference_trips_per_person: float
    target_trips_per_person: float
    predicted_shift_pct: float
    bts_shift_pct: float
    absolute_external_error_pct_points: float
    direction_match: bool
    note: str


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_validation"))
    parser.add_argument("--history-n-estimators", type=int, default=200)
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


def clipped(values: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def weighted_total(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return float(np.sum(value_array * weight_array))


def household_trips_per_person(trips: pd.Series | np.ndarray, hsize: pd.Series, weights: pd.Series) -> float:
    trip_total = weighted_total(trips, weights)
    person_total = weighted_total(hsize, weights)
    if person_total <= 0:
        raise ValueError("Weighted household person total must be positive.")
    return trip_total / person_total


def add_event_pressure(frame: pd.DataFrame) -> pd.DataFrame:
    adapted = frame.copy()
    base_pressure = np.zeros(len(adapted), dtype=float)
    for column, weight in PRESSURE_WEIGHTS.items():
        base_pressure += weight * clipped(adapted[column])

    recovery = clipped(adapted["post_pandemic_recovery_sensitivity"])
    adapted["llm_recovery_adjusted_pressure"] = clipped(base_pressure * (1.0 - 0.25 * recovery))
    adapted["llm_trip_suppression_pressure"] = clipped(adapted["trip_suppression_risk"])
    return adapted


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def make_global_pressure(pressure: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_pressure = weighted_average(pressure, weights)
    return np.full_like(np.asarray(pressure, dtype=float), mean_pressure)


def make_delta_gated_pressure(
    pressure: np.ndarray,
    global_pressure: np.ndarray,
    threshold: float,
) -> np.ndarray:
    pressure_array = np.asarray(pressure, dtype=float)
    global_array = np.asarray(global_pressure, dtype=float)
    use_llm = np.abs(pressure_array - global_array) >= threshold
    return np.where(use_llm, pressure_array, global_array)


def apply_multiplicative_pressure(
    base_prediction: np.ndarray,
    pressure: np.ndarray,
    alpha: float,
    min_factor: float,
) -> np.ndarray:
    factor = np.clip(1.0 - alpha * np.asarray(pressure, dtype=float), min_factor, 1.0)
    return np.maximum(np.asarray(base_prediction, dtype=float) * factor, 0.0)


def bts_query_url() -> str:
    params = {
        "$select": "date,trips,pop_stay_at_home,pop_not_stay_at_home",
        "$where": "level='National' AND date between '2019-01-01T00:00:00' and '2022-12-31T00:00:00'",
        "$order": "date",
        "$limit": "5000",
    }
    return f"{BTS_RESOURCE_URL}?{urlencode(params)}"


def download_bts_daily_mobility(cache_path: Path, force_download: bool) -> pd.DataFrame:
    if cache_path.exists() and not force_download:
        LOGGER.info("Loading cached BTS daily mobility data from %s.", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    LOGGER.info("Downloading BTS daily mobility data from Socrata.")
    with urlopen(bts_query_url(), timeout=90) as response:
        rows = json.loads(response.read().decode("utf-8"))
    if not rows:
        raise RuntimeError("BTS daily mobility query returned no rows.")

    frame = pd.DataFrame(rows)
    for column in ("trips", "pop_stay_at_home", "pop_not_stay_at_home"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["population"] = frame["pop_stay_at_home"] + frame["pop_not_stay_at_home"]
    frame["trips_per_person"] = frame["trips"] / frame["population"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def annual_bts_summary(frame: pd.DataFrame) -> pd.DataFrame:
    annual = (
        frame.assign(year=frame["date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            days=("date", "nunique"),
            trips=("trips", "sum"),
            population_day_sum=("population", "sum"),
            stay_home_population_day_sum=("pop_stay_at_home", "sum"),
        )
    )
    annual["trips_per_person_per_day"] = annual["trips"] / annual["population_day_sum"]
    annual["stay_home_share"] = annual["stay_home_population_day_sum"] / annual["population_day_sum"]
    return annual


def build_candidate_dataset_table(output_path: Path) -> None:
    candidates = [
        {
            "dataset": "ACS Commuting in the United States: 2022",
            "provider": "U.S. Census Bureau",
            "url": ACS_COMMUTING_BRIEF_URL,
            "granularity": "national ACS commute-mode aggregate",
            "project_use": "Mechanism validation for remote work substitution and transit avoidance.",
            "status": "implemented",
        },
        {
            "dataset": "BTS Daily Mobility Statistics - National and State",
            "provider": "U.S. Bureau of Transportation Statistics / University of Maryland",
            "url": BTS_LANDING_URL,
            "granularity": "daily national/state aggregate",
            "project_use": "Trip-count compatibility audit; not a direct NHTS household-trip target.",
            "status": "implemented as guardrail",
        },
        {
            "dataset": "BTS Effects of COVID-19 on Travel Behavior / Census Household Pulse Survey",
            "provider": "U.S. BTS / U.S. Census Bureau",
            "url": BTS_COVID_BEHAVIOR_URL,
            "granularity": "national/state/metro survey aggregate",
            "project_use": "Mechanism validation for telework, transit reduction, and online shopping priors.",
            "status": "candidate",
        },
        {
            "dataset": "Regional post-pandemic household travel surveys",
            "provider": "MPO/state travel survey programs",
            "url": "https://www.campo-nc.us/mapsdata/trm-household-travel-survey",
            "granularity": "household/person/trip microdata when released",
            "project_use": "Full external household-level validation if compatible microdata are available.",
            "status": "manual follow-up",
        },
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)


def write_acs_mechanism_validation(output_path: Path) -> pd.DataFrame:
    """Write official ACS mechanism anchors extracted from ACSBR-018.

    The values are from Figure 2 and the accompanying transit text in the U.S.
    Census Bureau brief "Commuting in the United States: 2022". We store only
    aggregate percentages needed for mechanism validation, not Census microdata.
    """

    rows = [
        {
            "mechanism": "remote_work_substitution",
            "external_measure": "ACS worked-from-home commute share",
            "share_2019_pct": 5.7,
            "share_2021_pct": 17.9,
            "share_2022_pct": 15.2,
            "change_2019_2022_pct_points": 15.2 - 5.7,
            "relative_change_2019_2022_pct": (15.2 / 5.7 - 1.0) * 100.0,
            "llm_prior_direction": "increase",
            "external_support": "supports",
            "source": ACS_COMMUTING_BRIEF_URL,
        },
        {
            "mechanism": "transit_avoidance",
            "external_measure": "ACS public-transportation commute share",
            "share_2019_pct": 5.0,
            "share_2021_pct": 2.5,
            "share_2022_pct": 3.1,
            "change_2019_2022_pct_points": 3.1 - 5.0,
            "relative_change_2019_2022_pct": (3.1 / 5.0 - 1.0) * 100.0,
            "llm_prior_direction": "decrease",
            "external_support": "supports",
            "source": ACS_COMMUTING_BRIEF_URL,
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def build_predictions(
    frame: pd.DataFrame,
    household_2022: pd.DataFrame,
    n_estimators: int,
    device: str,
) -> dict[str, np.ndarray]:
    model, feature_columns = train_history_model(frame, n_estimators=n_estimators, device=device)
    base_prediction = np.maximum(model.predict(household_2022[feature_columns]), 0.0)

    enriched = add_event_pressure(household_2022)
    weights = enriched[WEIGHT_COLUMN].to_numpy(dtype=float)
    suppression_pressure = enriched["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    recovery_pressure = enriched["llm_recovery_adjusted_pressure"].to_numpy(dtype=float)
    global_suppression = make_global_pressure(suppression_pressure, weights)
    gated_suppression = make_delta_gated_pressure(
        suppression_pressure,
        global_suppression,
        threshold=0.15,
    )

    return {
        "historical_xgboost": base_prediction,
        "global_event_prior_a1": apply_multiplicative_pressure(
            base_prediction,
            global_suppression,
            alpha=1.0,
            min_factor=0.05,
        ),
        "recovery_event_prior_a1": apply_multiplicative_pressure(
            base_prediction,
            recovery_pressure,
            alpha=1.0,
            min_factor=0.05,
        ),
        "gated_trip_suppression_a1_d0p15": apply_multiplicative_pressure(
            base_prediction,
            gated_suppression,
            alpha=1.0,
            min_factor=0.05,
        ),
    }


def build_shift_rows(
    frame: pd.DataFrame,
    household_2022: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    bts_summary: pd.DataFrame,
) -> list[AggregateShift]:
    bts_2019 = float(
        bts_summary.loc[bts_summary["year"] == 2019, "trips_per_person_per_day"].iloc[0]
    )
    bts_2022 = float(
        bts_summary.loc[bts_summary["year"] == 2022, "trips_per_person_per_day"].iloc[0]
    )
    bts_shift_pct = (bts_2022 / bts_2019) - 1.0

    household_2017 = frame[frame["survey_year"] == 2017].copy()
    reference = household_trips_per_person(
        household_2017[TARGET_COLUMN],
        household_2017["HHSIZE"],
        household_2017[WEIGHT_COLUMN],
    )

    rows: list[AggregateShift] = []
    actual_2022 = household_trips_per_person(
        household_2022[TARGET_COLUMN],
        household_2022["HHSIZE"],
        household_2022[WEIGHT_COLUMN],
    )
    actual_shift = (actual_2022 / reference) - 1.0
    rows.append(
        AggregateShift(
            method="nhts_2022_observed_evaluation_only",
            reference_year=2017,
            target_year=2022,
            reference_trips_per_person=reference,
            target_trips_per_person=actual_2022,
            predicted_shift_pct=actual_shift,
            bts_shift_pct=bts_shift_pct,
            absolute_external_error_pct_points=abs(actual_shift - bts_shift_pct) * 100.0,
            direction_match=np.sign(actual_shift) == np.sign(bts_shift_pct),
            note="Uses 2022 labels only as final evaluation context, never for training.",
        )
    )

    for method, prediction in predictions.items():
        target = household_trips_per_person(
            prediction,
            household_2022["HHSIZE"],
            household_2022[WEIGHT_COLUMN],
        )
        shift_pct = (target / reference) - 1.0
        rows.append(
            AggregateShift(
                method=method,
                reference_year=2017,
                target_year=2022,
                reference_trips_per_person=reference,
                target_trips_per_person=target,
                predicted_shift_pct=shift_pct,
                bts_shift_pct=bts_shift_pct,
                absolute_external_error_pct_points=abs(shift_pct - bts_shift_pct) * 100.0,
                direction_match=np.sign(shift_pct) == np.sign(bts_shift_pct),
                note="External aggregate sanity check; household-level definitions differ from BTS device mobility.",
            )
        )
    return rows


def write_shift_rows(rows: list[AggregateShift], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dict__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_report(
    rows: list[AggregateShift],
    bts_summary: pd.DataFrame,
    acs_mechanisms: pd.DataFrame,
    report_path: Path,
) -> None:
    row_frame = pd.DataFrame([row.__dict__ for row in rows])
    method_rows = row_frame[row_frame["method"] != "nhts_2022_observed_evaluation_only"].copy()
    primary = method_rows[method_rows["method"] == "gated_trip_suppression_a1_d0p15"].iloc[0]

    bts_2019 = bts_summary[bts_summary["year"] == 2019].iloc[0]
    bts_2022 = bts_summary[bts_summary["year"] == 2022].iloc[0]
    report = f"""# External Validation and Compatibility Audit

This audit uses independent non-NHTS sources only for evaluation and
mechanism checking. No external data are used to train or calibrate the NHTS
prediction model.

## Mechanism-Level External Validation

The strongest currently available external evidence is mechanism-level rather
than household-level. The U.S. Census Bureau ACS commuting brief reports that
post-pandemic commuting retained two structural shifts that match the LLM event
priors used in this project:

Source: {ACS_COMMUTING_BRIEF_URL}

| Mechanism | External measure | 2019 | 2021 | 2022 | 2019->2022 change | LLM prior |
|---|---|---:|---:|---:|---:|---|
"""
    for _, row in acs_mechanisms.iterrows():
        report += (
            f"| {row['mechanism']} | {row['external_measure']} | "
            f"{row['share_2019_pct']:.1f}% | {row['share_2021_pct']:.1f}% | "
            f"{row['share_2022_pct']:.1f}% | "
            f"{row['change_2019_2022_pct_points']:+.1f} pp | "
            f"{row['llm_prior_direction']} |\n"
        )

    report += f"""
Interpretation: ACS supports the event semantics used by the LLM adapter:
work-from-home remained far above the 2019 level, while public-transportation
commuting remained below the 2019 level in 2022. This validates the direction
of `remote_work_substitution` and `transit_avoidance` priors, but it is not a
direct household trip-count accuracy test.

## BTS Trip-Count Compatibility Guardrail

BTS / University of Maryland Daily Mobility Statistics provide independent
daily national trip counts from mobile-device-derived mobility statistics:
{BTS_LANDING_URL}

We tested whether BTS trips per person can be used as an external numeric target
for NHTS household travel-day trip counts. The answer is **no**: the two data
products have different trip definitions, observation mechanisms, and baseline
years. The check is retained as a guardrail against overclaiming.

### BTS Reference

| Year | Days | Trips per person per day | Stay-home share |
|---:|---:|---:|---:|
| 2019 | {int(bts_2019['days'])} | {bts_2019['trips_per_person_per_day']:.4f} | {bts_2019['stay_home_share']:.4f} |
| 2022 | {int(bts_2022['days'])} | {bts_2022['trips_per_person_per_day']:.4f} | {bts_2022['stay_home_share']:.4f} |

BTS 2019 to 2022 trip-per-person shift:
`{float(primary['bts_shift_pct']) * 100:+.2f}%`.

### NHTS-to-BTS Compatibility Check

| Method | NHTS 2017->2022 shift | BTS shift | External error | Direction match |
|---|---:|---:|---:|---|
"""
    for _, row in row_frame.sort_values("absolute_external_error_pct_points").iterrows():
        report += (
            f"| {row['method']} | {row['predicted_shift_pct'] * 100:+.2f}% | "
            f"{row['bts_shift_pct'] * 100:+.2f}% | "
            f"{row['absolute_external_error_pct_points']:.2f} pp | "
            f"{'yes' if row['direction_match'] else 'no'} |\n"
        )

    report += f"""
## Interpretation

- ACS mechanism evidence supports the qualitative event priors used by the
  LLM adapter.
- BTS aggregate device trips should **not** be used as a direct external
  numeric label for NHTS household `CNTTDHH`: even the observed 2022 NHTS
  target has a large shift mismatch against BTS.
- Therefore, the defensible external claim is mechanism-level validation plus
  a trip-count compatibility guardrail, not external household-level MAE.

## Scope

This closes a course-project-level external evidence gap but does not close the
stronger paper-submission requirement of household-level external microdata
validation. For a paper submission, this result should be reported as
mechanism-level external validation and paired with a limitation statement.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    set_random_seed(RANDOM_SEED)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    LOGGER.info("Using device=%s.", device)

    bts_cache = output_dir / "bts_daily_mobility_national_2019_2022.csv"
    bts_frame = download_bts_daily_mobility(bts_cache, args.force_download)
    bts_summary = annual_bts_summary(bts_frame)
    bts_summary.to_csv(output_dir / "bts_annual_mobility_summary.csv", index=False)
    build_candidate_dataset_table(output_dir / "external_dataset_candidates.csv")
    acs_mechanisms = write_acs_mechanism_validation(output_dir / "acs_commute_mechanism_validation.csv")

    LOGGER.info("Loading NHTS household data and LLM event priors.")
    full_frame = pd.read_csv(args.dataset_path, low_memory=False)
    household_2022 = load_2022_with_llm_features(
        args.dataset_path,
        args.cohort_profiles_path,
        args.llm_features_path,
    )
    predictions = build_predictions(
        full_frame,
        household_2022,
        n_estimators=args.history_n_estimators,
        device=device,
    )
    shift_rows = build_shift_rows(full_frame, household_2022, predictions, bts_summary)
    write_shift_rows(shift_rows, output_dir / "external_aggregate_validation_metrics.csv")
    write_report(
        shift_rows,
        bts_summary,
        acs_mechanisms,
        output_dir / "external_validation_and_compatibility_report.md",
    )
    LOGGER.info("Wrote external aggregate validation artifacts to %s.", output_dir)


if __name__ == "__main__":
    main()
