"""Generate and validate batched LLM event-prior records."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_event_generation import (
    GenerationConfig,
    augment_messages,
    extract_response_content,
    load_env_file,
    post_json,
    stable_json_hash,
)
from validate_llm_event_features import NUMERIC_FEATURES, TEXT_FEATURES, validate_feature_record


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_JSONL = Path("outputs/event_context_corpus/frozen_context_batch_prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/llm_event_features/frozen_context_cursor_api_pilot")
DEFAULT_REFERENCE_FEATURES = Path(
    "outputs/llm_event_features/cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv"
)
DEFAULT_CURSOR_API_URL = "http://127.0.0.1:3008/v1/chat/completions"
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
@dataclass(frozen=True)
class BatchPromptRecord:
    """One batched prompt request."""

    line_number: int
    batch_id: str
    cohort_ids: list[str]
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class BatchGenerationResult:
    """Validated output for one batch request."""

    batch_id: str
    parsed_records: list[dict[str, Any]]
    raw_record: dict[str, Any]
    error_record: dict[str, Any] | None
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-features", type=Path, default=DEFAULT_REFERENCE_FEATURES)
    parser.add_argument("--provider", default="cursor_api")
    parser.add_argument("--base-url", default=os.getenv("CURSOR_API_BASE_URL", DEFAULT_CURSOR_API_URL))
    parser.add_argument("--auth-token-env", default="CURSOR_API_AUTH_TOKEN")
    parser.add_argument("--model", default=os.getenv("CURSOR_API_MODEL", "gpt-5.5-low"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--response-format",
        choices=("none", "json_object"),
        default="json_object",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_batch_prompt_records(path: Path) -> list[BatchPromptRecord]:
    records: list[BatchPromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            batch_id = payload.get("batch_id")
            cohort_ids = payload.get("cohort_ids")
            messages = payload.get("messages")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError(f"Line {line_number} is missing batch_id.")
            if not isinstance(cohort_ids, list) or not all(isinstance(value, str) for value in cohort_ids):
                raise ValueError(f"Line {line_number} is missing cohort_ids.")
            if not isinstance(messages, list):
                raise ValueError(f"Line {line_number} is missing messages.")
            records.append(
                BatchPromptRecord(
                    line_number=line_number,
                    batch_id=batch_id,
                    cohort_ids=list(cohort_ids),
                    messages=messages,
                )
            )
    if not records:
        raise ValueError(f"No batch prompt records loaded from {path}")
    return records


def select_records(records: list[BatchPromptRecord], offset: int, limit: int | None) -> list[BatchPromptRecord]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    available = records[offset:]
    if limit is None:
        return available
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    return available[:limit]


def make_request_payload(record: BatchPromptRecord, config: GenerationConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": augment_messages(record.messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def parse_batch_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = JSON_OBJECT_PATTERN.search(stripped)
        if match is None:
            raise ValueError("Response content does not contain a JSON object.")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON is not an object.")
    records = parsed.get("records")
    if not isinstance(records, list):
        raise ValueError("Response JSON must contain a records list.")
    return parsed


def validate_batch_payload(payload: dict[str, Any], expected_cohort_ids: list[str]) -> list[dict[str, Any]]:
    expected = set(expected_cohort_ids)
    parsed_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload["records"]:
        if not isinstance(item, dict):
            raise ValueError("Each records item must be an object.")
        validation = validate_feature_record(item, allow_extra_fields=False)
        if validation.cohort_id not in expected:
            raise ValueError(f"Unexpected cohort_id in response: {validation.cohort_id}")
        if validation.cohort_id in seen:
            raise ValueError(f"Duplicate cohort_id in response: {validation.cohort_id}")
        seen.add(validation.cohort_id)
        parsed_records.append({"cohort_id": validation.cohort_id, "features": validation.features})
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"Missing cohort_ids in response: {missing}")
    return parsed_records


def generate_one_batch(record: BatchPromptRecord, config: GenerationConfig) -> BatchGenerationResult:
    request_payload = make_request_payload(record, config)
    request_hash = stable_json_hash(request_payload)
    raw_attempts: list[dict[str, Any]] = []
    error = ""
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = post_json(config, request_payload)
            content = extract_response_content(response)
            payload = parse_batch_content(content)
            parsed_records = validate_batch_payload(payload, record.cohort_ids)
            return BatchGenerationResult(
                batch_id=record.batch_id,
                parsed_records=parsed_records,
                raw_record={
                    "batch_id": record.batch_id,
                    "cohort_ids": record.cohort_ids,
                    "provider": config.provider,
                    "model": config.model,
                    "attempts": [*raw_attempts, {"attempt": attempt, "request_hash": request_hash, "raw_response": response}],
                },
                error_record=None,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            raw_attempts.append({"attempt": attempt, "request_hash": request_hash, "error": error})
    return BatchGenerationResult(
        batch_id=record.batch_id,
        parsed_records=[],
        raw_record={
            "batch_id": record.batch_id,
            "cohort_ids": record.cohort_ids,
            "provider": config.provider,
            "model": config.model,
            "attempts": raw_attempts,
        },
        error_record={"batch_id": record.batch_id, "cohort_ids": record.cohort_ids, "error": error},
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_normalized_csv(path: Path, parsed_records: list[dict[str, Any]]) -> None:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record in parsed_records:
        row = {"cohort_id": record["cohort_id"]}
        row.update(record["features"])
        rows.append(row)
    fieldnames = ["cohort_id", *NUMERIC_FEATURES, *TEXT_FEATURES]
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)


def compare_with_reference(parsed_records: list[dict[str, Any]], reference_path: Path, output_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    if not parsed_records or not reference_path.exists():
        return pd.DataFrame()
    rows = [{"cohort_id": record["cohort_id"], **record["features"]} for record in parsed_records]
    generated = pd.DataFrame(rows)
    reference = pd.read_csv(reference_path)
    merged = generated.merge(reference, on="cohort_id", suffixes=("_frozen", "_reference"))
    comparison_rows: list[dict[str, float | str | int]] = []
    for column in NUMERIC_FEATURES:
        left = merged[f"{column}_frozen"].astype(float)
        right = merged[f"{column}_reference"].astype(float)
        comparison_rows.append(
            {
                "feature": column,
                "n": int(len(merged)),
                "mean_abs_delta": float(np.mean(np.abs(left - right))) if len(merged) else np.nan,
                "pearson": float(left.corr(right, method="pearson")) if len(merged) >= 2 else np.nan,
                "spearman": float(left.corr(right, method="spearman")) if len(merged) >= 2 else np.nan,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "frozen_context_vs_reference_prior_comparison.csv", index=False)
    return comparison


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    selected_records: list[BatchPromptRecord],
    parsed_records: list[dict[str, Any]],
    error_records: list[dict[str, Any]],
    comparison: Any,
) -> None:
    selected_cohorts = sum(len(record.cohort_ids) for record in selected_records)
    lines = [
        "# Frozen-Context Batched LLM Prior Generation Report",
        "",
        f"- Provider: `{args.provider}`",
        f"- Model: `{args.model}`",
        f"- Input prompt file: `{args.input_jsonl}`",
        f"- Selected batches: `{len(selected_records)}`",
        f"- Selected cohorts: `{selected_cohorts}`",
        f"- Valid generated cohort priors: `{len(parsed_records)}`",
        f"- Failed batches: `{len(error_records)}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Interpretation",
        "",
    ]
    if parsed_records:
        lines.append(
            "These priors were generated from the frozen event-context prompt package, so the request context is "
            "auditable and constrained to the committed source facts rather than open-ended model memory."
        )
    else:
        lines.append(
            "No validated priors were generated. Treat this artifact as a runnable frozen-context replay protocol, "
            "not as completed frozen-context prior evidence."
        )
    if error_records:
        lines.extend(["", "## Runtime Errors", "", "| Batch | Error |", "|---|---|"])
        for record in error_records[:10]:
            error_text = " ".join(str(record.get("error", "unknown")).split())[:240]
            lines.append(f"| {record.get('batch_id', 'runtime')} | {error_text} |")
        if len(error_records) > 10:
            lines.append(f"| ... | {len(error_records) - 10} additional errors omitted from report. |")
    if comparison is not None and not comparison.empty:
        lines.extend(["", "## Frozen Context vs GPT-Reference Prior Agreement", "", "| Feature | n | MAE | Pearson | Spearman |", "|---|---:|---:|---:|---:|"])
        for row in comparison.itertuples(index=False):
            lines.append(
                f"| {row.feature} | {row.n} | {row.mean_abs_delta:.4f} | {row.pearson:.4f} | {row.spearman:.4f} |"
            )
        lines.extend(
            [
                "",
                f"Mean numeric feature MAE: `{comparison['mean_abs_delta'].mean():.4f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paper Use",
            "",
            "Use this as the retrieval/frozen-context replay evidence for the LLM event-prior branch. "
            "If only a pilot subset is generated, report it as provenance validation rather than a full replacement "
            "for the existing GPT-reference priors.",
        ]
    )
    (output_dir / "frozen_context_batched_generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_generation_config(args: argparse.Namespace, auth_token: str) -> GenerationConfig:
    return GenerationConfig(
        provider=args.provider,
        base_url=args.base_url,
        auth_token=auth_token,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        request_sleep_seconds=0.0,
        response_format=args.response_format,
    )


def main() -> None:
    load_env_file()
    args = parse_args()
    configure_logging()
    records = select_records(load_batch_prompt_records(args.input_jsonl), args.offset, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in args.output_dir.glob("frozen_context_*"):
            if path.is_file():
                path.unlink()
        for path in [args.output_dir / "llm_event_features.jsonl", args.output_dir / "llm_event_features_raw.jsonl"]:
            if path.exists():
                path.unlink()
    selected_cohorts = sum(len(record.cohort_ids) for record in records)
    LOGGER.info("Selected batches: %s", len(records))
    LOGGER.info("Selected cohorts: %s", selected_cohorts)
    if args.dry_run:
        write_summary(args.output_dir, args, records, [], [], None)
        return
    auth_token = os.getenv(args.auth_token_env, "")
    if not auth_token:
        error_records = [
            {
                "batch_id": "runtime_auth",
                "cohort_ids": [],
                "error": f"Missing auth token env var: {args.auth_token_env}",
            }
        ]
        write_jsonl(args.output_dir / "llm_event_feature_generation_errors.jsonl", error_records)
        write_summary(args.output_dir, args, records, [], error_records, None)
        LOGGER.error("Missing auth token env var: %s", args.auth_token_env)
        return
    config = build_generation_config(args, auth_token)
    parsed_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(generate_one_batch, record, config): record for record in records}
        for future in as_completed(futures):
            result = future.result()
            raw_records.append(result.raw_record)
            parsed_records.extend(result.parsed_records)
            if result.error_record is not None:
                error_records.append(result.error_record)
                LOGGER.warning("Batch failed: %s", result.batch_id)
            else:
                LOGGER.info("Batch succeeded: %s (%s cohorts)", result.batch_id, len(result.parsed_records))
    write_jsonl(args.output_dir / "llm_event_features.jsonl", parsed_records)
    write_jsonl(args.output_dir / "llm_event_features_raw.jsonl", raw_records)
    write_jsonl(args.output_dir / "llm_event_feature_generation_errors.jsonl", error_records)
    write_normalized_csv(args.output_dir / "llm_event_features_normalized.csv", parsed_records)
    comparison = compare_with_reference(parsed_records, args.reference_features, args.output_dir)
    write_summary(args.output_dir, args, records, parsed_records, error_records, comparison)


if __name__ == "__main__":
    main()
