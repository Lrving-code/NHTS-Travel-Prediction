"""Audit LLM cohort inputs for target leakage fields."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
FORBIDDEN_KEYS = {
    "CNTTDHH",
    "WTHHFIN",
    "HOUSEID",
    "PERSONID",
    "TDTRPNUM",
    "TRPTRANS",
    "TRIPPURP",
    "weighted_target_mean",
    "weighted_prediction_mean",
    "target",
    "label",
    "trip_count_label",
}
REQUIRED_GUARDRAIL_TERMS = ("CNTTDHH", "WTHHFIN", "HOUSEID", "aggregate 2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path("outputs/llm_event_features/household_cohort_profiles.csv"),
    )
    parser.add_argument(
        "--batch-prompt-path",
        type=Path,
        default=Path("outputs/llm_event_features/household_cohort_batch_prompts_b15.jsonl"),
    )
    parser.add_argument(
        "--llm-features-path",
        type=Path,
        default=Path(
            "outputs/llm_event_features/"
            "cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/leakage_audit"))
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def collect_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            keys.add(path)
            keys.add(key_text)
            keys.update(collect_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.update(collect_keys(child, f"{prefix}[{index}]"))
    return keys


def forbidden_matches(keys: set[str]) -> list[str]:
    matches: list[str] = []
    lower_forbidden = {key.lower() for key in FORBIDDEN_KEYS}
    for key in sorted(keys):
        key_lower = key.lower()
        if key_lower in lower_forbidden or any(part.lower() in lower_forbidden for part in key.split(".")):
            matches.append(key)
    return matches


def audit_profiles(profile_path: Path) -> dict[str, int | str]:
    profiles = pd.read_csv(profile_path)
    violations: list[str] = []
    for row in profiles.itertuples(index=False):
        profile = json.loads(row.profile_json)
        keys = collect_keys(profile)
        matches = forbidden_matches(keys)
        if matches:
            violations.append(f"{row.cohort_id}: {', '.join(matches[:8])}")
    return {
        "artifact": str(profile_path),
        "records_checked": len(profiles),
        "violations": len(violations),
        "examples": " | ".join(violations[:5]),
    }


def extract_user_payload(batch_record: dict[str, Any]) -> dict[str, Any]:
    for message in batch_record.get("messages", []):
        if message.get("role") == "user":
            return json.loads(message["content"])
    raise ValueError(f"No user payload found for batch {batch_record.get('batch_id')}")


def audit_batch_prompts(batch_prompt_path: Path) -> tuple[dict[str, int | str], dict[str, bool | str]]:
    batch_count = 0
    cohort_count = 0
    violations: list[str] = []
    guardrail_text_parts: list[str] = []
    with batch_prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch_count += 1
            record = json.loads(line)
            payload = extract_user_payload(record)
            guardrail_text_parts.append(json.dumps(payload.get("leakage_guardrails", []), ensure_ascii=False))
            guardrail_text_parts.append(str(payload.get("prospective_event_context", "")))
            for cohort in payload.get("cohorts", []):
                cohort_count += 1
                cohort_payload = cohort.get("profile", {})
                keys = collect_keys(cohort_payload)
                matches = forbidden_matches(keys)
                if matches:
                    violations.append(f"{record.get('batch_id')}:{cohort.get('cohort_id')}: {', '.join(matches[:8])}")
    guardrail_text = "\n".join(guardrail_text_parts)
    guardrails = {term: term in guardrail_text for term in REQUIRED_GUARDRAIL_TERMS}
    return (
        {
            "artifact": str(batch_prompt_path),
            "records_checked": cohort_count,
            "batches_checked": batch_count,
            "violations": len(violations),
            "examples": " | ".join(violations[:5]),
        },
        {
            "artifact": str(batch_prompt_path),
            "all_required_guardrails_present": all(guardrails.values()),
            "missing_terms": ", ".join(term for term, present in guardrails.items() if not present),
        },
    )


def audit_llm_features(features_path: Path) -> dict[str, int | str]:
    features = pd.read_csv(features_path, nrows=1)
    matches = forbidden_matches(set(features.columns))
    return {
        "artifact": str(features_path),
        "records_checked": 1,
        "violations": len(matches),
        "examples": ", ".join(matches[:8]),
    }


def write_report(rows: list[dict[str, int | str]], guardrail_row: dict[str, bool | str], output_dir: Path) -> Path:
    path = output_dir / "llm_input_leakage_audit_report.md"
    lines = [
        "# LLM Input Leakage Audit",
        "",
        "## Scope",
        "",
        "This audit checks whether LLM cohort payloads contain target labels, survey weights, household IDs, trip-level labels, or target-derived aggregate outcomes. It treats forbidden terms inside explicit guardrail instructions as allowed and necessary warnings.",
        "",
        "## Payload Audit",
        "",
        "| Artifact | Records checked | Violations | Examples |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['artifact']} | {row['records_checked']} | {row['violations']} | {row.get('examples', '')} |"
        )
    lines.extend(
        [
            "",
            "## Guardrail Instruction Check",
            "",
            f"- Required guardrail terms present: `{guardrail_row['all_required_guardrails_present']}`",
            f"- Missing terms: `{guardrail_row['missing_terms']}`",
            "",
            "## Interpretation",
            "",
            "A passing audit supports the label-free claim at the input-schema level: the LLM branch receives cohort covariates and event context, not 2022 target labels or evaluation outcomes. This does not eliminate retrospective world-knowledge risk; that remains a separate RAG/frozen-context requirement for a paper-grade version.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_row = audit_profiles(args.profile_path)
    batch_row, guardrail_row = audit_batch_prompts(args.batch_prompt_path)
    feature_row = audit_llm_features(args.llm_features_path)
    rows = [profile_row, batch_row, feature_row]
    pd.DataFrame(rows).to_csv(args.output_dir / "llm_input_leakage_audit.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame([guardrail_row]).to_csv(args.output_dir / "llm_guardrail_instruction_check.csv", index=False)
    report_path = write_report(rows, guardrail_row, args.output_dir)
    LOGGER.info("Wrote leakage audit report: %s", report_path)


if __name__ == "__main__":
    main()
