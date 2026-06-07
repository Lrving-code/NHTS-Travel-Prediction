"""Probe local Cursor API concurrency with tiny requests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_event_generation import load_env_file


DEFAULT_BASE_URL = "http://127.0.0.1:3008/v1/chat/completions"
DEFAULT_OUTPUT_DIR = Path("outputs/llm_event_features/cursor_api_concurrency")


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for one concurrency probe run."""

    base_url: str
    auth_token: str
    model: str
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class ProbeResult:
    """Result for one tiny request."""

    request_id: int
    ok: bool
    latency_seconds: float
    content: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("CURSOR_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--auth-token-env", default="CURSOR_API_AUTH_TOKEN")
    parser.add_argument("--model", default=os.getenv("CURSOR_API_MODEL", "gpt-5.5-low"))
    parser.add_argument("--levels", default="1,2,3,5,8,10,15,20,25,30")
    parser.add_argument("--requests-per-level", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error-rate", type=float, default=0.2)
    return parser.parse_args()


def parse_levels(raw_levels: str) -> list[int]:
    levels = [int(value.strip()) for value in raw_levels.split(",") if value.strip()]
    if not levels:
        raise ValueError("--levels cannot be empty.")
    if max(levels) > 30:
        raise ValueError("Refusing to test concurrency above 30.")
    if min(levels) < 1:
        raise ValueError("All concurrency levels must be >= 1.")
    return levels


def post_probe(config: ProbeConfig, request_id: int) -> ProbeResult:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly this JSON and nothing else: {"ok":true}',
            }
        ],
        "temperature": 0,
        "max_tokens": config.max_tokens,
    }
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
    start = time.time()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=config.timeout_seconds) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        content = extract_content(envelope)
        ok = '"ok"' in content and "true" in content.lower()
        error = "" if ok else f"Unexpected content: {content[:120]}"
        return ProbeResult(request_id, ok, round(time.time() - start, 3), content, error)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
        return ProbeResult(request_id, False, round(time.time() - start, 3), "", str(error))


def extract_content(envelope: dict[str, Any]) -> str:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing message.")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Missing content.")
    return content.strip()


def summarize_results(level: int, results: list[ProbeResult], elapsed_seconds: float) -> dict[str, Any]:
    latencies = [result.latency_seconds for result in results if result.ok]
    error_count = sum(1 for result in results if not result.ok)
    row: dict[str, Any] = {
        "concurrency": level,
        "requests": len(results),
        "success": len(results) - error_count,
        "errors": error_count,
        "error_rate": round(error_count / len(results), 4) if results else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_rps": round(len(results) / elapsed_seconds, 4) if elapsed_seconds > 0 else 0.0,
        "latency_mean": "",
        "latency_p50": "",
        "latency_p95": "",
        "latency_max": "",
    }
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_index = min(len(sorted_latencies) - 1, round((len(sorted_latencies) - 1) * 0.95))
        row.update(
            {
                "latency_mean": round(statistics.mean(latencies), 3),
                "latency_p50": round(statistics.median(latencies), 3),
                "latency_p95": round(sorted_latencies[p95_index], 3),
                "latency_max": round(max(latencies), 3),
            }
        )
    return row


def run_level(level: int, request_count: int, config: ProbeConfig) -> tuple[dict[str, Any], list[ProbeResult]]:
    start = time.time()
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=level) as executor:
        futures = [executor.submit(post_probe, config, request_id) for request_id in range(1, request_count + 1)]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_seconds = time.time() - start
    return summarize_results(level, results, elapsed_seconds), results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_details(path: Path, level: int, results: list[ProbeResult]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item.request_id):
            handle.write(
                json.dumps(
                    {
                        "concurrency": level,
                        "request_id": result.request_id,
                        "ok": result.ok,
                        "latency_seconds": result.latency_seconds,
                        "content": result.content,
                        "error": result.error,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )


def write_markdown(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    header = "| concurrency | requests | success | errors | error_rate | p50 | p95 | max | rps |"
    divider = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Cursor API Concurrency Probe",
        "",
        f"- Model: `{args.model}`",
        f"- Max tokens: {args.max_tokens}",
        f"- Requests per level: {args.requests_per_level}",
        "",
        header,
        divider,
    ]
    for row in rows:
        lines.append(
            "| {concurrency} | {requests} | {success} | {errors} | {error_rate} | "
            "{latency_p50} | {latency_p95} | {latency_max} | {throughput_rps} |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_env_file()
    args = parse_args()
    auth_token = os.getenv(args.auth_token_env, "")
    if not auth_token:
        raise ValueError(f"Missing auth token env var: {args.auth_token_env}")
    levels = parse_levels(args.levels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "concurrency_summary.csv"
    details_path = args.output_dir / "concurrency_details.jsonl"
    markdown_path = args.output_dir / "concurrency_summary.md"
    if args.overwrite:
        for path in (summary_path, details_path, markdown_path):
            if path.exists():
                path.unlink()

    config = ProbeConfig(
        base_url=args.base_url,
        auth_token=auth_token,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    rows: list[dict[str, Any]] = []
    for level in levels:
        row, results = run_level(level, args.requests_per_level, config)
        rows.append(row)
        write_details(details_path, level, results)
        write_csv(summary_path, rows)
        write_markdown(markdown_path, args, rows)
        print(
            f"concurrency={level} success={row['success']}/{row['requests']} "
            f"errors={row['errors']} p95={row['latency_p95']} rps={row['throughput_rps']}"
        )
        if row["error_rate"] > args.stop_on_error_rate:
            print(f"Stopping because error_rate={row['error_rate']} exceeded threshold.")
            break


if __name__ == "__main__":
    main()
