"""Utilities for generating LLM event features."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from validate_llm_event_features import ValidationResult, validate_feature_record


LOGGER = logging.getLogger(__name__)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
STRICT_JSON_INSTRUCTION = (
    "Return exactly one compact JSON object and no prose, markdown, code fence, "
    "or explanatory text. Use all required schema fields. Numeric values must be "
    "between 0 and 1."
)


@dataclass(frozen=True)
class PromptRecord:
    """Input prompt for one cohort."""

    line_number: int
    cohort_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration for one generation run."""

    provider: str
    base_url: str
    auth_token: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_attempts: int
    request_sleep_seconds: float
    response_format: str


@dataclass(frozen=True)
class GenerationResult:
    """Result for one cohort generation attempt sequence."""

    cohort_id: str
    ok: bool
    parsed_record: dict[str, Any] | None
    raw_records: list[dict[str, Any]]
    error_record: dict[str, Any] | None


@dataclass(frozen=True)
class OutputPaths:
    """Output paths for generated features and logs."""

    parsed: Path
    raw: Path
    errors: Path
    summary: Path


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stable_json_hash(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_prompt_records(input_jsonl: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            cohort_id = payload.get("cohort_id")
            if not isinstance(cohort_id, str) or not cohort_id:
                raise ValueError(f"Line {line_number} is missing cohort_id.")
            records.append(PromptRecord(line_number=line_number, cohort_id=cohort_id, payload=payload))
    return records


def select_records(
    records: list[PromptRecord],
    limit: int | None,
    offset: int,
    selection: str,
    cohort_ids: list[str] | None,
) -> list[PromptRecord]:
    if cohort_ids:
        wanted = set(cohort_ids)
        selected = [record for record in records if record.cohort_id in wanted]
        missing = sorted(wanted - {record.cohort_id for record in selected})
        if missing:
            raise ValueError(f"Requested cohort IDs not found: {missing}")
        return selected

    available = records[offset:]
    if limit is None or limit >= len(available):
        return available
    if selection == "first":
        return available[:limit]
    if limit <= 0:
        return []
    if limit == 1:
        return [available[0]]
    step = (len(available) - 1) / (limit - 1)
    return [available[round(index * step)] for index in range(limit)]


def load_completed_cohort_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    completed: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            cohort_id = record.get("cohort_id")
            if isinstance(cohort_id, str):
                completed.add(cohort_id)
    return completed


def rough_token_count(records: list[PromptRecord]) -> int:
    total_chars = 0
    for record in records:
        total_chars += len(json.dumps(record.payload.get("messages", []), ensure_ascii=True))
    return round(total_chars / 4)


def prepare_output_paths(output_dir: Path, overwrite: bool) -> OutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = OutputPaths(
        parsed=output_dir / "llm_event_features.jsonl",
        raw=output_dir / "llm_event_features_raw.jsonl",
        errors=output_dir / "llm_event_feature_generation_errors.jsonl",
        summary=output_dir / "llm_event_feature_generation_summary.md",
    )
    if overwrite:
        for path in (paths.parsed, paths.raw, paths.errors, paths.summary):
            if path.exists():
                path.unlink()
    return paths


def append_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def augment_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message must contain string role and content.")
        normalized_messages.append({"role": role, "content": content})
    normalized_messages.append({"role": "system", "content": STRICT_JSON_INSTRUCTION})
    return normalized_messages


def make_request_payload(record: PromptRecord, config: GenerationConfig) -> dict[str, Any]:
    messages = record.payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{record.cohort_id} is missing messages list.")
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": augment_messages(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def post_json(config: GenerationConfig, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        config.base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.auth_token}",
        },
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Request failed: {error.reason}") from error

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Provider returned non-JSON envelope: {response_body[:500]}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("Provider response envelope is not a JSON object.")
    return decoded


def extract_response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Provider response has no choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Provider choice is not an object.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Provider choice has no message object.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider message content is empty.")
    return content.strip()


def parse_feature_content(cohort_id: str, content: str) -> ValidationResult:
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
    return validate_feature_record({"cohort_id": cohort_id, "features": parsed}, allow_extra_fields=False)


def make_repair_payload(
    record: PromptRecord,
    config: GenerationConfig,
    previous_content: str,
    validation_error: str,
) -> dict[str, Any]:
    payload = make_request_payload(record, config)
    payload["messages"].append({"role": "assistant", "content": previous_content[:4000]})
    payload["messages"].append(
        {
            "role": "user",
            "content": (
                "The previous answer failed validation with this error: "
                f"{validation_error}\nReturn a corrected JSON object only."
            ),
        }
    )
    return payload


def parsed_output_record(
    record: PromptRecord,
    config: GenerationConfig,
    validation_result: ValidationResult,
    request_hash: str,
    attempt_count: int,
) -> dict[str, Any]:
    return {
        "cohort_id": record.cohort_id,
        "provider": config.provider,
        "model": config.model,
        "request_hash": request_hash,
        "attempt_count": attempt_count,
        "features": validation_result.features,
    }


def generate_one(record: PromptRecord, config: GenerationConfig) -> GenerationResult:
    raw_records: list[dict[str, Any]] = []
    request_payload = make_request_payload(record, config)
    previous_content = ""
    validation_error = ""

    for attempt in range(1, config.max_attempts + 1):
        if attempt > 1:
            request_payload = make_repair_payload(record, config, previous_content, validation_error)
        request_hash = stable_json_hash(request_payload)
        started_at = time.time()
        try:
            response = post_json(config, request_payload)
            content = extract_response_content(response)
            previous_content = content
            raw_records.append(
                {
                    "cohort_id": record.cohort_id,
                    "provider": config.provider,
                    "model": config.model,
                    "attempt": attempt,
                    "request_hash": request_hash,
                    "latency_seconds": round(time.time() - started_at, 3),
                    "raw_response": response,
                }
            )
            validation_result = parse_feature_content(record.cohort_id, content)
            return GenerationResult(
                cohort_id=record.cohort_id,
                ok=True,
                parsed_record=parsed_output_record(record, config, validation_result, request_hash, attempt),
                raw_records=raw_records,
                error_record=None,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            validation_error = str(error)
            raw_records.append(
                {
                    "cohort_id": record.cohort_id,
                    "provider": config.provider,
                    "model": config.model,
                    "attempt": attempt,
                    "request_hash": request_hash,
                    "error": validation_error,
                }
            )
            if config.request_sleep_seconds > 0:
                time.sleep(config.request_sleep_seconds)

    return GenerationResult(
        cohort_id=record.cohort_id,
        ok=False,
        parsed_record=None,
        raw_records=raw_records,
        error_record={
            "cohort_id": record.cohort_id,
            "provider": config.provider,
            "model": config.model,
            "error": validation_error,
            "attempt_count": config.max_attempts,
        },
    )
