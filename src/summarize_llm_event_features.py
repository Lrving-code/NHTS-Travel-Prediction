"""Summarize generated LLM event features for manual audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_FEATURES: tuple[str, ...] = (
    "trip_suppression_risk",
    "remote_work_substitution_likelihood",
    "transit_avoidance_likelihood",
    "online_delivery_substitution_likelihood",
    "post_pandemic_recovery_sensitivity",
    "confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    return parser.parse_args()


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = frame.loc[:, list(NUMERIC_FEATURES)].apply(pd.to_numeric, errors="coerce")
    summary = numeric_frame.agg(["count", "mean", "std", "min", "median", "max"]).transpose()
    return summary.round(4)


def format_markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown()


def write_summary(frame: pd.DataFrame, output_md: Path, max_examples: int) -> None:
    mechanisms = frame["primary_event_mechanism"].value_counts().head(12)
    examples = frame.head(max_examples)
    lines = [
        "# LLM Event Feature Audit Summary",
        "",
        f"- Records: {len(frame)}",
        "",
        "## Numeric Feature Distribution",
        "",
        format_markdown_table(numeric_summary(frame)),
        "",
        "## Primary Event Mechanisms",
        "",
        mechanisms.to_frame("count").to_markdown(),
        "",
        "## Example Explanations",
        "",
    ]
    for _, row in examples.iterrows():
        lines.extend(
            [
                f"### {row['cohort_id']}",
                "",
                f"- Mechanism: {row['primary_event_mechanism']}",
                f"- Confidence: {row['confidence']}",
                f"- Explanation: {row['short_explanation']}",
                "",
            ]
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_summary(frame, args.output_md, args.max_examples)


if __name__ == "__main__":
    main()
