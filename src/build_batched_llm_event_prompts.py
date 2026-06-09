"""Build batched LLM prompts for cohort-level event-prior generation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_PROFILE_PATH = Path("outputs/llm_event_features/household_cohort_profiles.csv")
DEFAULT_SCHEMA_PATH = Path("outputs/llm_event_features/event_feature_schema.json")
DEFAULT_OUTPUT_JSONL = Path("outputs/llm_event_features/household_cohort_batch_prompts.jsonl")
DEFAULT_SUMMARY_PATH = Path("outputs/llm_event_features/household_cohort_batch_prompt_summary.md")


@dataclass(frozen=True)
class CohortProfile:
    """Compact cohort profile for one LLM item."""

    cohort_id: str
    cohort_size: int
    profile: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-path", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-chars", type=int, default=32000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prospective-context-path", type=Path, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def load_profiles(path: Path, limit: int | None) -> list[CohortProfile]:
    profiles: list[CohortProfile] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            profile = json.loads(row["profile_json"])
            profiles.append(
                CohortProfile(
                    cohort_id=row["cohort_id"],
                    cohort_size=int(row["cohort_size"]),
                    profile=profile,
                )
            )
            if limit is not None and len(profiles) >= limit:
                break
    if not profiles:
        raise ValueError(f"No profiles loaded from {path}")
    return profiles


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(path: Path | None) -> str:
    if path is None:
        return (
            "Use general pandemic-era transportation knowledge only. Do not use 2022 NHTS outcomes, "
            "aggregate target statistics, sample weights, or household identifiers."
        )
    return path.read_text(encoding="utf-8").strip()


def make_profile_item(profile: CohortProfile) -> dict[str, Any]:
    return {
        "cohort_id": profile.cohort_id,
        "cohort_size": profile.cohort_size,
        "profile": profile.profile,
    }


def make_batches(
    profiles: list[CohortProfile],
    batch_size: int,
    max_batch_chars: int,
) -> list[list[CohortProfile]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    batches: list[list[CohortProfile]] = []
    current: list[CohortProfile] = []
    current_chars = 0
    for profile in profiles:
        item_chars = len(compact_json(make_profile_item(profile)))
        would_exceed_size = len(current) >= batch_size
        would_exceed_chars = current and current_chars + item_chars > max_batch_chars
        if would_exceed_size or would_exceed_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(profile)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def make_messages(batch: list[CohortProfile], schema: dict[str, Any], context: str) -> list[dict[str, str]]:
    cohort_items = [make_profile_item(profile) for profile in batch]
    system = (
        "You are a transportation behavior analyst. Infer pandemic-era event-response priors "
        "for multiple NHTS household cohorts in one batch. Do not predict household trip counts. "
        "Do not infer or mention any target label. Return JSON only."
    )
    user_payload = {
        "task": "Infer event-response priors for each cohort.",
        "prospective_event_context": context,
        "leakage_guardrails": [
            "Do not use CNTTDHH or any trip-count label.",
            "Do not use WTHHFIN sample weights.",
            "Do not use HOUSEID or household identifiers.",
            "Do not use aggregate 2022 NHTS target outcomes.",
        ],
        "output_contract": {
            "type": "object",
            "required_key": "records",
            "records_item": {
                "cohort_id": "same cohort_id as input",
                "features": schema,
            },
        },
        "cohorts": cohort_items,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": compact_json(user_payload)},
    ]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(compact_json(record) + "\n")


def write_summary(
    path: Path,
    profiles: list[CohortProfile],
    batches: list[list[CohortProfile]],
    batch_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    profile_count = len(profiles)
    batch_count = len(batches)
    request_reduction = 1.0 - batch_count / profile_count
    token_estimate = round(sum(len(compact_json(record["messages"])) for record in batch_records) / 4)
    max_cohorts = max(len(batch) for batch in batches)
    max_chars = max(len(compact_json(record["messages"])) for record in batch_records)
    lines = [
        "# Batched LLM Prompt Summary",
        "",
        f"- Source cohort profiles: `{args.profiles_path}`",
        f"- Cohorts: `{profile_count}`",
        f"- Batches: `{batch_count}`",
        f"- Batch size target: `{args.batch_size}`",
        f"- Max cohorts in a batch: `{max_cohorts}`",
        f"- Max message chars: `{max_chars}`",
        f"- Rough total input token estimate: `{token_estimate}`",
        f"- Request reduction vs one-cohort prompts: `{request_reduction:.2%}`",
        f"- Output JSONL: `{args.output_jsonl}`",
        "",
        "## Recommended Use",
        "",
        "Use this when the LLM endpoint supports long context and reliable JSON output. "
        "Run a small pilot first, validate every returned record, then scale batch size upward. "
        "For the current 1327 cohorts, batch size 8 reduces requests to about 166; batch size 15 reduces them to about 89 but increases validation risk.",
        "",
        "## Why This Is Better",
        "",
        "- Fewer API calls and less scheduling overhead.",
        "- Cohorts are still auditable and label-free.",
        "- The LLM can compare related cohorts inside a batch, which may improve ranking consistency.",
        "- The output remains a structured event-prior table, not direct trip-count prediction.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    profiles = load_profiles(args.profiles_path, args.limit)
    schema = load_schema(args.schema_path)
    context = load_context(args.prospective_context_path)
    batches = make_batches(profiles, args.batch_size, args.max_batch_chars)
    records: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        records.append(
            {
                "batch_id": f"batch_{index:04d}",
                "cohort_ids": [profile.cohort_id for profile in batch],
                "messages": make_messages(batch, schema, context),
            }
        )
    write_jsonl(records, args.output_jsonl)
    write_summary(args.summary_path, profiles, batches, records, args)
    LOGGER.info("Wrote %s batched prompts for %s cohorts.", len(records), len(profiles))


if __name__ == "__main__":
    main()
