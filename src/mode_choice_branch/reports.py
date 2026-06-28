"""Diagnostic reports for the mode-choice processing branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    MODE_GROUP_LABELS,
    MODE_GROUP_ORDER,
    TABLE_SPECS,
    TARGET_COLUMN,
    TRPTRANS_LABELS_BY_YEAR,
    map_trptrans_to_mode_group,
    read_csv_selected,
    source_path,
)


@dataclass(frozen=True)
class FieldDiff:
    """Field-difference summary for one source table."""

    table_name: str
    columns_2017: int
    columns_2022: int
    common_columns: tuple[str, ...]
    only_2017: tuple[str, ...]
    only_2022: tuple[str, ...]


def read_header_columns(path: Path) -> set[str]:
    """Read only the header row and return uppercase column names."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")
    frame = pd.read_csv(path, nrows=0)
    return {str(column).upper() for column in frame.columns}


def analyze_field_diff(raw_root: Path, table_name: str) -> FieldDiff:
    """Compare 2017 and 2022 raw schema fields for one table."""
    fields_2017 = read_header_columns(source_path(raw_root, table_name, 2017))
    fields_2022 = read_header_columns(source_path(raw_root, table_name, 2022))
    return FieldDiff(
        table_name=table_name,
        columns_2017=len(fields_2017),
        columns_2022=len(fields_2022),
        common_columns=tuple(sorted(fields_2017 & fields_2022)),
        only_2017=tuple(sorted(fields_2017 - fields_2022)),
        only_2022=tuple(sorted(fields_2022 - fields_2017)),
    )


def write_field_diff_report(
    raw_root: Path = DEFAULT_RAW_ROOT,
    report_path: Path = DEFAULT_OUTPUT_DIR / "reports" / "field_diff_report.md",
) -> list[FieldDiff]:
    """Write a raw-schema comparison report."""
    diffs = [analyze_field_diff(raw_root, table_name) for table_name in TABLE_SPECS]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# NHTS 2017/2022 Field Difference Report",
        "",
        "This report reproduces the archive field audit against the repository data layout.",
        "",
        "| Table | 2017 Columns | 2022 Columns | Common | Only 2017 | Only 2022 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for diff in diffs:
        lines.append(
            f"| {diff.table_name} | {diff.columns_2017} | {diff.columns_2022} | "
            f"{len(diff.common_columns)} | {len(diff.only_2017)} | {len(diff.only_2022)} |"
        )

    for diff in diffs:
        lines.extend(
            [
                "",
                f"## {diff.table_name}",
                "",
                "Only in 2017:",
                "",
                ", ".join(diff.only_2017) if diff.only_2017 else "None",
                "",
                "Only in 2022:",
                "",
                ", ".join(diff.only_2022) if diff.only_2022 else "None",
                "",
                "Common fields:",
                "",
                ", ".join(diff.common_columns) if diff.common_columns else "None",
            ]
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return diffs


def load_mode_group_counts(raw_root: Path, year: int) -> pd.Series:
    """Load raw TRPTRANS values and count harmonized mode groups."""
    trip_path = source_path(raw_root, "trip", year)
    frame = read_csv_selected(trip_path, [TARGET_COLUMN], low_memory=False)
    groups = map_trptrans_to_mode_group(frame[TARGET_COLUMN], year).dropna()
    return groups.value_counts()


def raw_trptrans_distribution(raw_root: Path, year: int) -> pd.DataFrame:
    """Return a year-specific raw TRPTRANS distribution with year-specific labels."""
    trip_path = source_path(raw_root, "trip", year)
    frame = read_csv_selected(trip_path, [TARGET_COLUMN], low_memory=False)
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").dropna().astype(int)
    counts = values.value_counts().sort_index()
    total = int(counts.sum())
    labels = TRPTRANS_LABELS_BY_YEAR[year]
    rows = []
    for code, count in counts.items():
        rows.append(
            {
                "year": year,
                "code": int(code),
                "label": labels.get(int(code), f"unknown_{int(code)}"),
                "count": int(count),
                "percent": float(count / total * 100) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_trptrans_comparison(raw_root: Path = DEFAULT_RAW_ROOT) -> pd.DataFrame:
    """Build a 2017/2022 harmonized mode-group distribution comparison table."""
    count_2017 = load_mode_group_counts(raw_root, 2017)
    count_2022 = load_mode_group_counts(raw_root, 2022)
    total_2017 = count_2017.sum()
    total_2022 = count_2022.sum()
    percent_2017 = count_2017 / total_2017 * 100
    percent_2022 = count_2022 / total_2022 * 100

    rows = []
    observed_modes = set(count_2017.index) | set(count_2022.index)
    ordered_modes = [mode for mode in MODE_GROUP_ORDER if mode in observed_modes]
    ordered_modes.extend(sorted(observed_modes - set(ordered_modes)))
    for mode in ordered_modes:
        rows.append(
            {
                "mode_group": mode,
                "label": MODE_GROUP_LABELS.get(str(mode), str(mode)),
                "count_2017": int(count_2017.get(mode, 0)),
                "percent_2017": float(percent_2017.get(mode, 0.0)),
                "count_2022": int(count_2022.get(mode, 0)),
                "percent_2022": float(percent_2022.get(mode, 0.0)),
                "change_percentage_points": float(percent_2022.get(mode, 0.0) - percent_2017.get(mode, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def write_trptrans_plot(comparison: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    """Write a compact bar chart for harmonized mode groups."""
    import matplotlib.pyplot as plt
    import numpy as np

    comparison = comparison.copy()
    comparison["combined_percent"] = comparison["percent_2017"] + comparison["percent_2022"]
    top_modes = comparison.sort_values("combined_percent", ascending=False).head(top_n)

    positions = np.arange(len(top_modes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(positions - width / 2, top_modes["percent_2017"], width, label="2017", color="#3b82f6", alpha=0.85)
    ax.bar(positions + width / 2, top_modes["percent_2022"], width, label="2022", color="#ef4444", alpha=0.85)
    ax.set_ylabel("Share (%)")
    ax.set_title("Harmonized Mode Distribution: 2017 vs 2022")
    ax.set_xticks(positions)
    ax.set_xticklabels(top_modes["label"], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_trptrans_report(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR / "reports",
    make_plot: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Write harmonized TRPTRANS comparison CSV, markdown, and optional plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_trptrans_comparison(raw_root)
    raw_distribution = pd.concat(
        [raw_trptrans_distribution(raw_root, 2017), raw_trptrans_distribution(raw_root, 2022)],
        ignore_index=True,
    )
    csv_path = output_dir / "harmonized_mode_distribution.csv"
    raw_csv_path = output_dir / "raw_trptrans_distribution_by_year.csv"
    report_path = output_dir / "harmonized_mode_distribution_report.md"
    plot_path = output_dir / "harmonized_mode_distribution.png" if make_plot else None
    comparison.to_csv(csv_path, index=False)
    raw_distribution.to_csv(raw_csv_path, index=False)
    if plot_path is not None:
        write_trptrans_plot(comparison, plot_path)

    top_increase = comparison.loc[comparison["change_percentage_points"] > 0].sort_values(
        "change_percentage_points",
        ascending=False,
    ).head(5)
    top_decrease = comparison.loc[comparison["change_percentage_points"] < 0].sort_values(
        "change_percentage_points",
        ascending=True,
    ).head(5)

    lines = [
        "# Harmonized Mode Distribution Report",
        "",
        "This report compares trip-level travel-mode distributions after mapping year-specific raw `TRPTRANS` codes into comparable mode groups.",
        "",
        "The raw `TRPTRANS` code meanings differ across the 2017 and 2022 NHTS codebooks; for example, code `01` means walk in 2017 but car in 2022. The raw-code distribution is therefore saved only as a schema diagnostic, not as a direct cross-year comparison.",
        "",
        f"- Harmonized distribution CSV: `{csv_path.as_posix()}`",
        f"- Raw code diagnostic CSV: `{raw_csv_path.as_posix()}`",
    ]
    if plot_path is not None:
        lines.append(f"- Top-mode plot: `{plot_path.as_posix()}`")

    lines.extend(
        [
            "",
            "## Distribution",
            "",
            "| Mode group | Label | 2017 Count | 2017 (%) | 2022 Count | 2022 (%) | Change (pp) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.sort_values("percent_2017", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.mode_group} | {row.label} | {row.count_2017:,} | {row.percent_2017:.2f} | "
            f"{row.count_2022:,} | {row.percent_2022:.2f} | {row.change_percentage_points:+.2f} |"
        )

    lines.extend(["", "## Largest Increases", "", "| Label | Change (pp) |", "|---|---:|"])
    for row in top_increase.itertuples(index=False):
        lines.append(f"| {row.label} | {row.change_percentage_points:+.2f} |")

    lines.extend(["", "## Largest Decreases", "", "| Label | Change (pp) |", "|---|---:|"])
    for row in top_decrease.itertuples(index=False):
        lines.append(f"| {row.label} | {row.change_percentage_points:+.2f} |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path, plot_path
