"""Build cohort-level household profiles for LLM event-feature generation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "CNTTDHH"
WEIGHT_COLUMN = "WTHHFIN"
ID_COLUMN = "HOUSEID"
PROFILE_YEAR = 2022

SAFE_COLUMNS: tuple[str, ...] = (
    "survey_year",
    "CDIVMSAR",
    "CENSUS_D",
    "CENSUS_R",
    "DRVRCNT",
    "HBHTNRNT",
    "HBHUR",
    "HBPPOPDN",
    "HHFAMINC",
    "HHSIZE",
    "HHVEHCNT",
    "HOMEOWN",
    "HTEEMPDN",
    "HTHTNRNT",
    "HTPPOPDN",
    "LIF_CYC",
    "MSACAT",
    "MSASIZE",
    "NUMADLT",
    "RAIL",
    "RESP_CNT",
    "TRAVDAY",
    "URBAN",
    "URBRUR",
    "WRKCOUNT",
    "travel_month",
)
DEFAULT_COHORT_COLUMNS: tuple[str, ...] = (
    "HHFAMINC",
    "HHVEHCNT_BIN",
    "WRKCOUNT_BIN",
    "URBRUR",
    "RAIL",
    "CENSUS_R",
)
NUMERIC_SUMMARY_COLUMNS: tuple[str, ...] = (
    "HHSIZE",
    "NUMADLT",
    "DRVRCNT",
    "RESP_CNT",
    "HHVEHCNT",
    "WRKCOUNT",
    "travel_month",
)
CATEGORICAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "HHFAMINC",
    "HOMEOWN",
    "URBRUR",
    "RAIL",
    "CENSUS_R",
    "CENSUS_D",
    "TRAVDAY",
    "MSACAT",
    "MSASIZE",
    "URBAN",
)
FORBIDDEN_COLUMNS = {TARGET_COLUMN, WEIGHT_COLUMN, ID_COLUMN}

EVENT_FEATURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "trip_suppression_risk",
        "remote_work_substitution_likelihood",
        "transit_avoidance_likelihood",
        "online_delivery_substitution_likelihood",
        "post_pandemic_recovery_sensitivity",
        "primary_event_mechanism",
        "short_explanation",
        "confidence",
    ],
    "properties": {
        "trip_suppression_risk": {"type": "number", "minimum": 0, "maximum": 1},
        "remote_work_substitution_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
        "transit_avoidance_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
        "online_delivery_substitution_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
        "post_pandemic_recovery_sensitivity": {"type": "number", "minimum": 0, "maximum": 1},
        "primary_event_mechanism": {"type": "string"},
        "short_explanation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "HHFAMINC": "NHTS household income category code.",
    "HHVEHCNT": "Count of household vehicles.",
    "WRKCOUNT": "Number of workers in the household.",
    "URBRUR": "NHTS urban/rural category code.",
    "RAIL": "NHTS rail availability category code.",
    "CENSUS_R": "US Census region code.",
    "CENSUS_D": "US Census division code.",
    "HHSIZE": "Count of household members.",
    "NUMADLT": "Count of adult household members.",
    "DRVRCNT": "Number of drivers in the household.",
    "RESP_CNT": "Count of responding persons in the household.",
    "HOMEOWN": "Home ownership category code.",
    "TRAVDAY": "Travel day of week code.",
    "travel_month": "Travel month derived from TDAYDATE.",
}

SYSTEM_PROMPT = """You are a transportation behavior analyst.
Your task is to infer pandemic-era event-response priors for a household cohort.
Do not predict household trip counts. Do not infer or mention any target label.
Return only valid JSON matching the provided schema."""


@dataclass(frozen=True)
class CohortProfile:
    """Cohort profile prepared for LLM feature generation."""

    cohort_id: str
    cohort_key: dict[str, str]
    cohort_size: int
    profile: dict[str, Any]


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
        default=Path("outputs/llm_event_features"),
        help="Directory for cohort profiles and LLM prompt JSONL.",
    )
    parser.add_argument(
        "--profile-year",
        type=int,
        default=PROFILE_YEAR,
        help="Survey year to profile for event adaptation.",
    )
    parser.add_argument(
        "--min-cohort-size",
        type=int,
        default=1,
        help="Drop cohorts smaller than this threshold.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_safe_columns(columns: tuple[str, ...]) -> None:
    leaked_columns = FORBIDDEN_COLUMNS.intersection(columns)
    if leaked_columns:
        raise ValueError(f"Forbidden columns in profile inputs: {sorted(leaked_columns)}")


def bin_count(value: object, cap: int = 3) -> str:
    if pd.isna(value):
        return "missing"
    numeric_value = int(value)
    if numeric_value >= cap:
        return f"{cap}+"
    return str(numeric_value)


def load_profile_frame(dataset_path: Path, profile_year: int) -> pd.DataFrame:
    validate_safe_columns(SAFE_COLUMNS)
    frame = pd.read_csv(dataset_path, usecols=list(SAFE_COLUMNS), low_memory=False)
    frame = frame[frame["survey_year"] == profile_year].copy()
    if frame.empty:
        raise ValueError(f"No rows found for profile year {profile_year}.")
    frame["HHVEHCNT_BIN"] = frame["HHVEHCNT"].map(bin_count)
    frame["WRKCOUNT_BIN"] = frame["WRKCOUNT"].map(bin_count)
    return frame.reset_index(drop=True)


def value_counts_dict(series: pd.Series, max_values: int = 12) -> dict[str, int]:
    counts = series.astype("string").fillna("missing").value_counts(dropna=False).head(max_values)
    return {str(key): int(value) for key, value in counts.items()}


def numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


def make_profile(
    cohort_id: str,
    cohort_key: dict[str, str],
    frame: pd.DataFrame,
    profile_year: int,
) -> CohortProfile:
    profile = {
        "profile_year": profile_year,
        "cohort_id": cohort_id,
        "cohort_size": int(len(frame)),
        "cohort_key": cohort_key,
        "variable_descriptions": VARIABLE_DESCRIPTIONS,
        "numeric_summaries": {
            column: numeric_summary(frame, column) for column in NUMERIC_SUMMARY_COLUMNS
        },
        "categorical_distributions": {
            column: value_counts_dict(frame[column]) for column in CATEGORICAL_SUMMARY_COLUMNS
        },
        "leakage_note": (
            "This profile excludes target labels, sample weights, identifiers, "
            "and all target-derived or aggregate 2022 outcome statistics."
        ),
    }
    return CohortProfile(
        cohort_id=cohort_id,
        cohort_key=cohort_key,
        cohort_size=int(len(frame)),
        profile=profile,
    )


def build_profiles(
    frame: pd.DataFrame,
    profile_year: int,
    min_cohort_size: int,
) -> list[CohortProfile]:
    profiles: list[CohortProfile] = []
    grouped = frame.groupby(list(DEFAULT_COHORT_COLUMNS), dropna=False, sort=True)
    cohort_index = 1
    for key_values, cohort_frame in grouped:
        if len(cohort_frame) < min_cohort_size:
            continue
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        cohort_key = {
            column: str(value)
            for column, value in zip(DEFAULT_COHORT_COLUMNS, key_values, strict=True)
        }
        cohort_id = f"hhc_{cohort_index:06d}"
        profiles.append(make_profile(cohort_id, cohort_key, cohort_frame, profile_year))
        cohort_index += 1
    return profiles


def build_user_prompt(profile: dict[str, Any]) -> str:
    profile_json = json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True)
    schema_json = json.dumps(EVENT_FEATURE_SCHEMA, ensure_ascii=True, indent=2, sort_keys=True)
    return (
        "Infer pandemic-era event-response priors for this NHTS household cohort.\n"
        "Use only the cohort profile and general transportation behavior knowledge.\n"
        "Do not predict or mention household trip counts.\n\n"
        f"Profile:\n{profile_json}\n\n"
        f"Required JSON schema:\n{schema_json}"
    )


def write_profiles_csv(profiles: list[CohortProfile], output_path: Path) -> None:
    fieldnames = [
        "cohort_id",
        "cohort_size",
        *DEFAULT_COHORT_COLUMNS,
        "profile_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            row = {
                "cohort_id": profile.cohort_id,
                "cohort_size": profile.cohort_size,
                "profile_json": json.dumps(profile.profile, ensure_ascii=True, sort_keys=True),
            }
            row.update(profile.cohort_key)
            writer.writerow(row)


def write_prompt_jsonl(profiles: list[CohortProfile], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            record = {
                "cohort_id": profile.cohort_id,
                "profile_year": profile.profile["profile_year"],
                "cohort_key": profile.cohort_key,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(profile.profile)},
                ],
                "expected_schema": EVENT_FEATURE_SCHEMA,
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_summary(
    profiles: list[CohortProfile],
    source_rows: int,
    min_cohort_size: int,
    output_path: Path,
) -> None:
    covered_rows = sum(profile.cohort_size for profile in profiles)
    lines = [
        "# LLM Household Cohort Profile Summary",
        "",
        f"- Source rows: {source_rows}",
        f"- Cohorts written: {len(profiles)}",
        f"- Rows covered by written cohorts: {covered_rows}",
        f"- Minimum cohort size: {min_cohort_size}",
        f"- Cohort columns: {', '.join(DEFAULT_COHORT_COLUMNS)}",
        "",
        "Leakage controls:",
        "",
        "- Excludes `CNTTDHH`.",
        "- Excludes `WTHHFIN`.",
        "- Excludes `HOUSEID`.",
        "- Excludes aggregate 2022 target statistics.",
        "- Prompts instruct the LLM not to predict or mention household trip counts.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_profile_frame(args.dataset_path, args.profile_year)
    profiles = build_profiles(frame, args.profile_year, args.min_cohort_size)
    if not profiles:
        raise ValueError("No profiles generated. Lower --min-cohort-size.")

    write_profiles_csv(profiles, args.output_dir / "household_cohort_profiles.csv")
    write_prompt_jsonl(profiles, args.output_dir / "household_cohort_prompts.jsonl")
    (args.output_dir / "event_feature_schema.json").write_text(
        json.dumps(EVENT_FEATURE_SCHEMA, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        profiles,
        source_rows=len(frame),
        min_cohort_size=args.min_cohort_size,
        output_path=args.output_dir / "household_cohort_profile_summary.md",
    )
    LOGGER.info("Wrote %s cohort profiles to %s", len(profiles), args.output_dir)


if __name__ == "__main__":
    main()
