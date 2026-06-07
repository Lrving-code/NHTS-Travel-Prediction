"""Generate LLM event features from cohort prompts."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_event_generation import (
    GenerationConfig,
    append_jsonl,
    generate_one,
    load_completed_cohort_ids,
    load_env_file,
    load_prompt_records,
    prepare_output_paths,
    rough_token_count,
    select_records,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_JSONL = Path("outputs/llm_event_features/household_cohort_prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/llm_event_features/cursor_api_pilot")
DEFAULT_CURSOR_API_URL = "http://127.0.0.1:3008/v1/chat/completions"
DEFAULT_PROVIDER = "cursor_api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--base-url", default=os.getenv("CURSOR_API_BASE_URL", DEFAULT_CURSOR_API_URL))
    parser.add_argument("--auth-token-env", default="CURSOR_API_AUTH_TOKEN")
    parser.add_argument("--model", default=os.getenv("CURSOR_API_MODEL", "gpt-5.5-low"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--selection", choices=("first", "even"), default="even")
    parser.add_argument("--cohort-id", action="append", dest="cohort_ids", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--response-format",
        choices=("none", "json_object"),
        default="none",
        help="Provider hint only. Local Cursor API did not enforce this in smoke tests.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def write_run_summary(
    output_path: Path,
    args: argparse.Namespace,
    selected_count: int,
    skipped_count: int,
    success_count: int,
    error_count: int,
    input_token_estimate: int,
) -> None:
    lines = [
        "# LLM Event Feature Generation Summary",
        "",
        f"- Provider: `{args.provider}`",
        f"- Model: `{args.model}`",
        f"- Input prompt file: `{args.input_jsonl}`",
        f"- Selected records: {selected_count}",
        f"- Skipped completed records: {skipped_count}",
        f"- New successful records: {success_count}",
        f"- New error records: {error_count}",
        f"- Rough input token estimate: {input_token_estimate}",
        f"- Max concurrency: {args.max_concurrency}",
        f"- Max attempts: {args.max_attempts}",
        f"- Response format hint: `{args.response_format}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
        request_sleep_seconds=args.request_sleep_seconds,
        response_format=args.response_format,
    )


def validate_runtime_args(args: argparse.Namespace, auth_token: str) -> None:
    if not args.dry_run and not auth_token:
        raise ValueError(f"Missing auth token env var: {args.auth_token_env}")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")


def main() -> None:
    load_env_file()
    args = parse_args()
    configure_logging()
    auth_token = os.getenv(args.auth_token_env, "")
    validate_runtime_args(args, auth_token)

    records = load_prompt_records(args.input_jsonl)
    selected_records = select_records(records, args.limit, args.offset, args.selection, args.cohort_ids)
    paths = prepare_output_paths(args.output_dir, args.overwrite)
    completed_ids = set() if args.no_resume else load_completed_cohort_ids(paths.parsed)
    records_to_run = [record for record in selected_records if record.cohort_id not in completed_ids]
    input_token_estimate = rough_token_count(records_to_run)

    skipped_count = len(selected_records) - len(records_to_run)
    LOGGER.info("Loaded records: %s", len(records))
    LOGGER.info("Selected records: %s", len(selected_records))
    LOGGER.info("Skipped completed records: %s", skipped_count)
    LOGGER.info("Records to run: %s", len(records_to_run))
    LOGGER.info("Rough input token estimate: %s", input_token_estimate)

    if args.dry_run:
        write_run_summary(paths.summary, args, len(selected_records), skipped_count, 0, 0, input_token_estimate)
        return

    config = build_generation_config(args, auth_token)
    success_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_to_record = {executor.submit(generate_one, record, config): record for record in records_to_run}
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            result = future.result()
            append_jsonl(paths.raw, result.raw_records)
            if result.parsed_record is not None:
                append_jsonl(paths.parsed, [result.parsed_record])
                success_count += 1
                LOGGER.info("Generated valid features for %s", record.cohort_id)
            if result.error_record is not None:
                append_jsonl(paths.errors, [result.error_record])
                error_count += 1
                LOGGER.warning("Generation failed for %s", record.cohort_id)

    write_run_summary(
        paths.summary,
        args,
        len(selected_records),
        skipped_count,
        success_count,
        error_count,
        input_token_estimate,
    )
    LOGGER.info("Successful records: %s", success_count)
    LOGGER.info("Error records: %s", error_count)


if __name__ == "__main__":
    main()
