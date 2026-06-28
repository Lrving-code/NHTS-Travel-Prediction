"""Build single-cohort frozen-context prompts for local LLM replay."""

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
DEFAULT_CONTEXT_PATH = Path("outputs/event_context_corpus/frozen_event_context_prompt.md")
DEFAULT_OUTPUT_JSONL = Path("outputs/event_context_corpus/frozen_context_single_prompts.jsonl")
DEFAULT_SUMMARY_PATH = Path("outputs/event_context_corpus/frozen_context_single_prompt_summary.md")


@dataclass(frozen=True)
class CohortProfile:
    """Cohort profile for one prompt."""

    cohort_id: str
    cohort_size: int
    cohort_key: dict[str, str]
    profile: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-path", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--context-path", type=Path, default=DEFAULT_CONTEXT_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_profiles(path: Path, limit: int | None) -> list[CohortProfile]:
    profiles: list[CohortProfile] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cohort_columns = [column for column in reader.fieldnames or [] if column not in {"cohort_id", "cohort_size", "profile_json"}]
        for row in reader:
            profile = json.loads(row["profile_json"])
            profiles.append(
                CohortProfile(
                    cohort_id=row["cohort_id"],
                    cohort_size=int(row["cohort_size"]),
                    cohort_key={column: row[column] for column in cohort_columns},
                    profile=profile,
                )
            )
            if limit is not None and len(profiles) >= limit:
                break
    if not profiles:
        raise ValueError(f"No profiles loaded from {path}")
    return profiles


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def make_messages(profile: CohortProfile, schema: dict[str, Any], context: str) -> list[dict[str, str]]:
    system = (
        "You are a transportation behavior analyst. Infer pandemic-era event-response priors "
        "for one NHTS household cohort. Do not predict household trip counts. Do not infer or mention "
        "any target label. Return one JSON object only."
    )
    user_payload = {
        "task": "Infer event-response priors for this cohort.",
        "cohort_id": profile.cohort_id,
        "cohort_size": profile.cohort_size,
        "cohort_key": profile.cohort_key,
        "profile": profile.profile,
        "prospective_event_context": context,
        "leakage_guardrails": [
            "Use only the frozen event context and cohort covariates.",
            "Do not use CNTTDHH or any trip-count label.",
            "Do not use WTHHFIN sample weights.",
            "Do not use HOUSEID or household identifiers.",
            "Do not use aggregate 2022 NHTS target outcomes.",
            "Do not use PSRC validation-only evidence as prompt context.",
        ],
        "output_contract": {
            "type": "object",
            "cohort_id": "same cohort_id as input may be omitted",
            "features": schema,
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": compact_json(user_payload)},
    ]


def write_prompt_jsonl(profiles: list[CohortProfile], schema: dict[str, Any], context: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            record = {
                "cohort_id": profile.cohort_id,
                "profile_year": profile.profile.get("profile_year"),
                "cohort_key": profile.cohort_key,
                "messages": make_messages(profile, schema, context),
                "expected_schema": schema,
                "context_source": "outputs/event_context_corpus/frozen_event_context_prompt.md",
            }
            handle.write(compact_json(record) + "\n")


def write_summary(
    profiles: list[CohortProfile],
    context: str,
    args: argparse.Namespace,
) -> None:
    max_chars = 0
    total_chars = 0
    schema = load_json(args.schema_path)
    for profile in profiles:
        chars = len(compact_json(make_messages(profile, schema, context)))
        max_chars = max(max_chars, chars)
        total_chars += chars
    lines = [
        "# Frozen-Context Single-Cohort Prompt Summary",
        "",
        f"- Source cohort profiles: `{args.profiles_path}`",
        f"- Frozen context: `{args.context_path}`",
        f"- Output JSONL: `{args.output_jsonl}`",
        f"- Cohorts: `{len(profiles)}`",
        f"- Max message chars: `{max_chars}`",
        f"- Rough total input token estimate: `{round(total_chars / 4)}`",
        "",
        "## Purpose",
        "",
        "This prompt file supports local/open-source LLM replay under the same frozen event-context constraints "
        "used by the batched API prompts. It is compatible with `src/run_local_llm_prior_replication.py`.",
        "The JSONL file can be regenerated from committed cohort profiles and frozen context, and may be ignored "
        "by git to avoid committing another large prompt artifact.",
        "",
        "## Leakage Boundary",
        "",
        "- Prompts include frozen ACS/BTS mechanism facts.",
        "- Prompts exclude NHTS 2022 labels, weights, identifiers, and aggregate target outcomes.",
        "- PSRC is explicitly excluded from prompt context and kept as validation-only evidence.",
    ]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    profiles = load_profiles(args.profiles_path, args.limit)
    schema = load_json(args.schema_path)
    context = args.context_path.read_text(encoding="utf-8").strip()
    write_prompt_jsonl(profiles, schema, context, args.output_jsonl)
    write_summary(profiles, context, args)
    LOGGER.info("Wrote %s frozen-context single prompts to %s", len(profiles), args.output_jsonl)


if __name__ == "__main__":
    main()
