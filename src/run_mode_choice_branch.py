"""Command entry point for the harmonized mode-choice processing branch."""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_choice_branch.common import (
    DEFAULT_INTERIM_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    LOGGER,
    configure_logging,
    parse_table_names,
)
from mode_choice_branch.datasets import build_mode_choice_datasets
from mode_choice_branch.preprocess import preprocess_all
from mode_choice_branch.reports import write_field_diff_report, write_trptrans_report


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    field_diff = subparsers.add_parser("field-diff", help="Compare raw 2017/2022 source schemas.")
    add_common_paths(field_diff)

    preprocess = subparsers.add_parser("preprocess", help="Preprocess common 2017/2022 NHTS fields.")
    add_common_paths(preprocess)
    preprocess.add_argument("--tables", default="hh,per,trip,veh", help="Comma-separated table names.")

    datasets = subparsers.add_parser("build-datasets", help="Build harmonized mode-group train/validation/test CSV files.")
    add_common_paths(datasets)
    datasets.add_argument("--test-size", type=float, default=0.2)
    datasets.add_argument("--random-state", type=int, default=42)

    trptrans = subparsers.add_parser("trptrans-report", help="Analyze harmonized TRPTRANS-derived mode shifts.")
    add_common_paths(trptrans)
    trptrans.add_argument("--no-plot", action="store_true", help="Skip chart generation.")

    all_steps = subparsers.add_parser("all", help="Run field diff, preprocessing, dataset build, and TRPTRANS report.")
    add_common_paths(all_steps)
    all_steps.add_argument("--tables", default="hh,per,trip,veh", help="Comma-separated table names.")
    all_steps.add_argument("--test-size", type=float, default=0.2)
    all_steps.add_argument("--random-state", type=int, default=42)
    all_steps.add_argument("--no-plot", action="store_true", help="Skip chart generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.command == "field-diff":
        report_path = args.output_dir / "reports" / "field_diff_report.md"
        write_field_diff_report(args.raw_root, report_path)
        LOGGER.info("Wrote field-difference report: %s", report_path)
        return

    if args.command == "preprocess":
        table_names = parse_table_names(args.tables)
        preprocess_all(args.raw_root, args.interim_dir, table_names, args.interim_dir / "preprocess_report.md")
        return

    if args.command == "build-datasets":
        result = build_mode_choice_datasets(
            args.interim_dir,
            args.output_dir / "datasets",
            test_size=args.test_size,
            random_state=args.random_state,
        )
        LOGGER.info("Wrote mode-choice datasets to %s", result.output_dir)
        return

    if args.command == "trptrans-report":
        _, report_path, plot_path = write_trptrans_report(
            args.raw_root,
            args.output_dir / "reports",
            make_plot=not args.no_plot,
        )
        LOGGER.info("Wrote TRPTRANS report: %s", report_path)
        if plot_path is not None:
            LOGGER.info("Wrote TRPTRANS plot: %s", plot_path)
        return

    if args.command == "all":
        table_names = parse_table_names(args.tables)
        field_report = args.output_dir / "reports" / "field_diff_report.md"
        write_field_diff_report(args.raw_root, field_report)
        preprocess_all(args.raw_root, args.interim_dir, table_names, args.interim_dir / "preprocess_report.md")
        build_mode_choice_datasets(
            args.interim_dir,
            args.output_dir / "datasets",
            test_size=args.test_size,
            random_state=args.random_state,
        )
        write_trptrans_report(args.raw_root, args.output_dir / "reports", make_plot=not args.no_plot)
        LOGGER.info("Completed full mode-choice branch pipeline.")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
