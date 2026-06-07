"""Build a harmonized household-level NHTS modeling dataset."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "CNTTDHH"
WEIGHT_COLUMN = "WTHHFIN"
ID_COLUMN = "HOUSEID"
DATE_COLUMN = "TDAYDATE"

COMMON_HOUSEHOLD_COLUMNS: tuple[str, ...] = (
    "CDIVMSAR",
    "CENSUS_D",
    "CENSUS_R",
    "CNTTDHH",
    "DRVRCNT",
    "HBHTNRNT",
    "HBHUR",
    "HBPPOPDN",
    "HHFAMINC",
    "HHSIZE",
    "HHVEHCNT",
    "HOMEOWN",
    "HOUSEID",
    "HTEEMPDN",
    "HTHTNRNT",
    "HTPPOPDN",
    "LIF_CYC",
    "MSACAT",
    "MSASIZE",
    "NUMADLT",
    "RAIL",
    "RESP_CNT",
    "TDAYDATE",
    "TRAVDAY",
    "URBAN",
    "URBRUR",
    "WRKCOUNT",
    "WTHHFIN",
)


@dataclass(frozen=True)
class HouseholdSource:
    """One NHTS household table source."""

    year: int
    path: Path


HOUSEHOLD_SOURCES: tuple[HouseholdSource, ...] = (
    HouseholdSource(year=2001, path=Path("data/raw/nhts_2001/ascii/HHPUB.csv")),
    HouseholdSource(year=2009, path=Path("data/raw/nhts_2009/ascii/Ascii/HHV2PUB.CSV")),
    HouseholdSource(year=2017, path=Path("data/raw/nhts_2017/csv/hhpub.csv")),
    HouseholdSource(year=2022, path=Path("data/raw/nhts_2022/csv/hhv2pub.csv")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/household_harmonized.csv"),
        help="Output CSV path for the harmonized household dataset.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/household_baseline/household_dataset_summary.md"),
        help="Markdown summary path.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def read_household_source(source: HouseholdSource) -> pd.DataFrame:
    if not source.path.exists():
        raise FileNotFoundError(f"Missing household source: {source.path}")

    LOGGER.info("Reading %s household data from %s", source.year, source.path)
    frame = pd.read_csv(source.path, usecols=list(COMMON_HOUSEHOLD_COLUMNS), low_memory=False)
    frame.columns = [str(column).upper() for column in frame.columns]
    missing_columns = set(COMMON_HOUSEHOLD_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{source.year} household data missing columns: {sorted(missing_columns)}")

    frame["survey_year"] = source.year
    frame["source_file"] = source.path.as_posix()
    frame["travel_month"] = pd.to_numeric(frame[DATE_COLUMN], errors="coerce") % 100
    return frame


def clean_household_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned[WEIGHT_COLUMN] = pd.to_numeric(cleaned[WEIGHT_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN, WEIGHT_COLUMN])
    cleaned = cleaned[cleaned[TARGET_COLUMN] >= 0]
    cleaned = cleaned[cleaned[WEIGHT_COLUMN] > 0]
    return cleaned.reset_index(drop=True)


def build_dataset(sources: tuple[HouseholdSource, ...]) -> pd.DataFrame:
    frames = [read_household_source(source) for source in sources]
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return clean_household_frame(combined)


def write_summary(frame: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    by_year = (
        frame.groupby("survey_year")
        .agg(
            rows=(TARGET_COLUMN, "size"),
            target_mean=(TARGET_COLUMN, "mean"),
            target_min=(TARGET_COLUMN, "min"),
            target_max=(TARGET_COLUMN, "max"),
            weight_sum=(WEIGHT_COLUMN, "sum"),
        )
        .reset_index()
    )

    lines = [
        "# Household Harmonized Dataset Summary",
        "",
        f"- Rows: {len(frame)}",
        f"- Columns: {len(frame.columns)}",
        f"- Target: `{TARGET_COLUMN}`",
        f"- Weight: `{WEIGHT_COLUMN}`",
        "",
        "| Year | Rows | Mean Target | Min Target | Max Target | Weight Sum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_year.itertuples(index=False):
        lines.append(
            f"| {row.survey_year} | {row.rows} | {row.target_mean:.3f} | "
            f"{row.target_min:.0f} | {row.target_max:.0f} | {row.weight_sum:.2f} |"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()

    dataset = build_dataset(HOUSEHOLD_SOURCES)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_path, index=False)
    write_summary(dataset, args.summary_path)

    LOGGER.info("Wrote household dataset: %s", args.output_path)
    LOGGER.info("Wrote household summary: %s", args.summary_path)


if __name__ == "__main__":
    main()
