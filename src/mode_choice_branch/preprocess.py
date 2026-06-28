"""Preprocess NHTS 2017 and 2022 tables for the mode-choice branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    CATEGORICAL_FIELDS,
    COMMON_FIELDS,
    DEFAULT_INTERIM_DIR,
    DEFAULT_RAW_ROOT,
    LOGGER,
    NEGATIVE_MISSING_CODES,
    NUMERIC_FIELDS,
    TABLE_SPECS,
    preprocessed_path,
    read_csv_selected,
    source_path,
)


@dataclass(frozen=True)
class PreprocessResult:
    """Summary for one preprocessed table."""

    table_name: str
    year: int
    input_path: Path
    output_path: Path
    rows: int
    columns: int


def standardize_data_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize known numeric and categorical fields."""
    result = frame.copy()
    for field in NUMERIC_FIELDS:
        if field in result.columns:
            result[field] = pd.to_numeric(result[field], errors="coerce")
    for field in CATEGORICAL_FIELDS:
        if field in result.columns:
            result[field] = result[field].astype("string")
    return result


def fill_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace NHTS missing-value codes and impute simple table-level values."""
    result = frame.replace(list(NEGATIVE_MISSING_CODES), np.nan).copy()

    numeric_fields = result.select_dtypes(include=[np.number]).columns
    for field in numeric_fields:
        if result[field].isna().any():
            median_value = result[field].median()
            fill_value = 0.0 if pd.isna(median_value) else float(median_value)
            result[field] = result[field].fillna(fill_value)

    string_fields = result.select_dtypes(include=["object", "string"]).columns
    for field in string_fields:
        if result[field].isna().any():
            mode_values = result[field].mode(dropna=True)
            fill_value = str(mode_values.iloc[0]) if not mode_values.empty else "Unknown"
            result[field] = result[field].fillna(fill_value)

    return result


def preprocess_table(table_name: str, year: int, raw_root: Path, interim_dir: Path) -> PreprocessResult:
    """Preprocess one table-year pair and write it to the interim directory."""
    input_path = source_path(raw_root, table_name, year)
    output_path = preprocessed_path(interim_dir, table_name, year)
    fields = COMMON_FIELDS[table_name]

    LOGGER.info("Reading %s %s from %s", year, table_name, input_path)
    frame = read_csv_selected(input_path, fields, low_memory=False)
    cleaned = fill_missing_values(standardize_data_types(frame))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s rows and %s columns to %s", len(cleaned), len(cleaned.columns), output_path)

    return PreprocessResult(
        table_name=table_name,
        year=year,
        input_path=input_path,
        output_path=output_path,
        rows=len(cleaned),
        columns=len(cleaned.columns),
    )


def write_preprocess_report(results: list[PreprocessResult], report_path: Path) -> None:
    """Write a compact preprocessing report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mode-Choice Preprocessing Report",
        "",
        "This report summarizes the archive-derived preprocessing branch after adapting it to repository paths.",
        "",
        "| Table | Year | Rows | Columns | Output |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.table_name} | {result.year} | {result.rows:,} | "
            f"{result.columns:,} | `{result.output_path.as_posix()}` |"
        )
    lines.append("")
    lines.append("Processing rules:")
    lines.append("- Keep only common 2017/2022 fields defined by the original archive workflow.")
    lines.append("- Convert known numeric fields with `pandas.to_numeric`.")
    lines.append("- Replace NHTS negative missing-value codes with missing values.")
    lines.append("- Fill numeric fields by median and categorical fields by mode within each table-year file.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def preprocess_all(
    raw_root: Path = DEFAULT_RAW_ROOT,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    table_names: tuple[str, ...] = tuple(TABLE_SPECS),
    report_path: Path | None = None,
) -> list[PreprocessResult]:
    """Preprocess selected NHTS table pairs."""
    results: list[PreprocessResult] = []
    for table_name in table_names:
        for year in (2017, 2022):
            results.append(preprocess_table(table_name, year, raw_root, interim_dir))

    target_report_path = report_path or (interim_dir / "preprocess_report.md")
    write_preprocess_report(results, target_report_path)
    LOGGER.info("Wrote preprocessing report: %s", target_report_path)
    return results
