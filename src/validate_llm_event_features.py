"""Validate and normalize LLM-generated event features."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
NUMERIC_FEATURES: tuple[str, ...] = (
    "trip_suppression_risk",
    "remote_work_substitution_likelihood",
    "transit_avoidance_likelihood",
    "online_delivery_substitution_likelihood",
    "post_pandemic_recovery_sensitivity",
    "confidence",
)
TEXT_FEATURES: tuple[str, ...] = (
    "primary_event_mechanism",
    "short_explanation",
)
REQUIRED_FIELDS: tuple[str, ...] = (*NUMERIC_FEATURES, *TEXT_FEATURES)


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for one LLM feature record."""

    cohort_id: str
    features: dict[str, float | str]


@dataclass(frozen=True)
class InvalidRecord:
    """Invalid LLM output with error details."""

    line_number: int
    error: str
    raw_record: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="JSONL file containing LLM outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_event_features"),
        help="Directory for normalized features and invalid records.",
    )
    parser.add_argument(
        "--allow-extra-fields",
        action="store_true",
        help="Allow fields beyond the strict event feature schema.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def unwrap_features(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cohort_id = record.get("cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("Missing non-empty cohort_id.")

    features = record.get("features", record)
    if not isinstance(features, dict):
        raise ValueError("Feature payload must be a JSON object.")
    return cohort_id, features


def validate_numeric(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric.") from error
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1], found {numeric_value}.")
    return numeric_value


def validate_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be non-empty.")
    return stripped


def validate_feature_record(
    record: dict[str, Any],
    allow_extra_fields: bool,
) -> ValidationResult:
    cohort_id, features = unwrap_features(record)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in features]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}.")

    if not allow_extra_fields:
        allowed_fields = set(REQUIRED_FIELDS)
        extra_fields = sorted(set(features) - allowed_fields - {"cohort_id", "features"})
        if extra_fields:
            raise ValueError(f"Unexpected extra fields: {extra_fields}.")

    normalized: dict[str, float | str] = {}
    for field in NUMERIC_FEATURES:
        normalized[field] = validate_numeric(features[field], field)
    for field in TEXT_FEATURES:
        normalized[field] = validate_text(features[field], field)
    return ValidationResult(cohort_id=cohort_id, features=normalized)


def load_and_validate(
    input_jsonl: Path,
    allow_extra_fields: bool,
) -> tuple[list[ValidationResult], list[InvalidRecord]]:
    valid_records: list[ValidationResult] = []
    invalid_records: list[InvalidRecord] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise ValueError("Each JSONL line must be an object.")
                valid_records.append(validate_feature_record(record, allow_extra_fields))
            except (json.JSONDecodeError, ValueError) as error:
                invalid_records.append(
                    InvalidRecord(
                        line_number=line_number,
                        error=str(error),
                        raw_record=stripped,
                    )
                )
    return valid_records, invalid_records


def write_valid_records(records: list[ValidationResult], output_path: Path) -> None:
    fieldnames = ["cohort_id", *NUMERIC_FEATURES, *TEXT_FEATURES]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, str | float] = {"cohort_id": record.cohort_id}
            row.update(record.features)
            writer.writerow(row)


def write_invalid_records(records: list[InvalidRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "line_number": record.line_number,
                        "error": record.error,
                        "raw_record": record.raw_record,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )


def write_summary(
    valid_count: int,
    invalid_count: int,
    output_path: Path,
) -> None:
    lines = [
        "# LLM Event Feature Validation Summary",
        "",
        f"- Valid records: {valid_count}",
        f"- Invalid records: {invalid_count}",
        f"- Required numeric features: {', '.join(NUMERIC_FEATURES)}",
        f"- Required text fields: {', '.join(TEXT_FEATURES)}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    valid_records, invalid_records = load_and_validate(
        args.input_jsonl,
        allow_extra_fields=args.allow_extra_fields,
    )
    write_valid_records(valid_records, args.output_dir / "llm_event_features_normalized.csv")
    write_invalid_records(invalid_records, args.output_dir / "llm_event_features_invalid.jsonl")
    write_summary(
        valid_count=len(valid_records),
        invalid_count=len(invalid_records),
        output_path=args.output_dir / "llm_event_feature_validation_summary.md",
    )
    LOGGER.info("Valid records: %s", len(valid_records))
    LOGGER.info("Invalid records: %s", len(invalid_records))


if __name__ == "__main__":
    main()
