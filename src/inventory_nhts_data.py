"""Create a lightweight inventory of prepared NHTS CSV data files."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".csv"}


@dataclass(frozen=True)
class TableInventory:
    """Inventory metadata for one extracted data table."""

    year: str
    package: str
    table_name: str
    path: Path
    suffix: str
    bytes_size: int
    row_count: int | None
    column_count: int | None
    columns: tuple[str, ...]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing extracted NHTS raw data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inventory"),
        help="Directory where inventory reports will be written.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def infer_year_and_package(path: Path, data_dir: Path) -> tuple[str, str]:
    relative_parts = path.relative_to(data_dir).parts
    year = "unknown"
    package = "unknown"
    if relative_parts:
        year = relative_parts[0].replace("nhts_", "")
    if len(relative_parts) > 1:
        package = relative_parts[1]
    return year, package


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        row_count = sum(1 for _ in reader)
    return max(row_count - 1, 0)


def inspect_csv(path: Path) -> tuple[int, int, tuple[str, ...]]:
    frame = pd.read_csv(path, nrows=0, low_memory=False)
    columns = tuple(str(column) for column in frame.columns)
    return count_csv_rows(path), len(columns), columns


def inspect_table(path: Path, data_dir: Path) -> TableInventory:
    year, package = infer_year_and_package(path, data_dir)
    suffix = path.suffix.lower()
    table_name = path.stem
    bytes_size = path.stat().st_size

    try:
        row_count, column_count, columns = inspect_csv(path)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        LOGGER.warning("Could not inspect %s: %s", path, error)
        return TableInventory(
            year=year,
            package=package,
            table_name=table_name,
            path=path,
            suffix=suffix,
            bytes_size=bytes_size,
            row_count=None,
            column_count=None,
            columns=tuple(),
            error=str(error),
        )

    return TableInventory(
        year=year,
        package=package,
        table_name=table_name,
        path=path,
        suffix=suffix,
        bytes_size=bytes_size,
        row_count=row_count,
        column_count=column_count,
        columns=columns,
    )


def discover_tables(data_dir: Path) -> list[Path]:
    paths = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(paths)


def write_table_summary(inventories: list[TableInventory], output_dir: Path) -> Path:
    output_path = output_dir / "table_inventory.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "year",
                "package",
                "table_name",
                "suffix",
                "bytes_size",
                "row_count",
                "column_count",
                "path",
                "error",
            ]
        )
        for inventory in inventories:
            writer.writerow(
                [
                    inventory.year,
                    inventory.package,
                    inventory.table_name,
                    inventory.suffix,
                    inventory.bytes_size,
                    inventory.row_count,
                    inventory.column_count,
                    inventory.path.as_posix(),
                    inventory.error or "",
                ]
            )
    return output_path


def write_column_summary(inventories: list[TableInventory], output_dir: Path) -> Path:
    output_path = output_dir / "column_inventory.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "package", "table_name", "column_name", "column_index"])
        for inventory in inventories:
            for column_index, column in enumerate(inventory.columns):
                writer.writerow(
                    [
                        inventory.year,
                        inventory.package,
                        inventory.table_name,
                        column,
                        column_index,
                    ]
                )
    return output_path


def write_markdown_summary(inventories: list[TableInventory], output_dir: Path) -> Path:
    output_path = output_dir / "inventory_summary.md"
    by_year: dict[str, list[TableInventory]] = {}
    for inventory in inventories:
        by_year.setdefault(inventory.year, []).append(inventory)

    lines = ["# NHTS Data Inventory", ""]
    for year in sorted(by_year):
        lines.extend([f"## {year}", ""])
        lines.append("| Package | Table | Format | Rows | Columns | Size MB | Error |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for inventory in sorted(by_year[year], key=lambda item: (item.package, item.table_name)):
            size_mb = inventory.bytes_size / (1024 * 1024)
            row_count = "" if inventory.row_count is None else str(inventory.row_count)
            column_count = "" if inventory.column_count is None else str(inventory.column_count)
            error = "" if inventory.error is None else inventory.error.replace("|", "/")
            lines.append(
                f"| {inventory.package} | {inventory.table_name} | {inventory.suffix} | "
                f"{row_count} | {column_count} | {size_mb:.2f} | {error} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_paths = discover_tables(args.data_dir)
    LOGGER.info("Found %s supported data files.", len(table_paths))
    inventories = [inspect_table(path, args.data_dir) for path in table_paths]

    table_path = write_table_summary(inventories, args.output_dir)
    column_path = write_column_summary(inventories, args.output_dir)
    markdown_path = write_markdown_summary(inventories, args.output_dir)

    LOGGER.info("Wrote table inventory: %s", table_path)
    LOGGER.info("Wrote column inventory: %s", column_path)
    LOGGER.info("Wrote markdown summary: %s", markdown_path)


if __name__ == "__main__":
    main()
