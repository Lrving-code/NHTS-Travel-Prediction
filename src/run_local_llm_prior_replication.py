"""Run or audit a local open-source LLM prior-replication experiment."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_event_generation import (
    PromptRecord,
    augment_messages,
    load_prompt_records,
    select_records,
    stable_json_hash,
)
from validate_llm_event_features import NUMERIC_FEATURES, TEXT_FEATURES, validate_feature_record


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_JSONL = Path("outputs/llm_event_features/household_cohort_prompts.jsonl")
DEFAULT_REFERENCE_FEATURES = Path(
    "outputs/llm_event_features/cursor_api_full_gpt55_low_c15/validated/llm_event_features_normalized.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/local_llm_prior_replication")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class EnvironmentAudit:
    nvidia_smi_available: bool
    gpu_name: str
    torch_installed: bool
    torch_version: str
    torch_cuda_available: bool
    torch_cuda_version: str
    transformers_installed: bool
    transformers_version: str
    accelerate_installed: bool
    bitsandbytes_installed: bool
    recommended_status: str
    recommendation: str


@dataclass(frozen=True)
class LocalGenerationRecord:
    cohort_id: str
    provider: str
    model: str
    request_hash: str
    attempt_count: int
    latency_seconds: float
    features: dict[str, float | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--reference-features", type=Path, default=DEFAULT_REFERENCE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--selection", choices=("first", "even"), default="even")
    parser.add_argument("--cohort-id", action="append", dest="cohort_ids", default=None)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def query_gpu_name() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        return False, message
    gpu_name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return bool(gpu_name), gpu_name


def inspect_environment() -> EnvironmentAudit:
    nvidia_ok, gpu_name = query_gpu_name()
    torch_installed = importlib.util.find_spec("torch") is not None
    transformers_installed = importlib.util.find_spec("transformers") is not None
    accelerate_installed = importlib.util.find_spec("accelerate") is not None
    bitsandbytes_installed = importlib.util.find_spec("bitsandbytes") is not None
    torch_version = package_version("torch") if torch_installed else "not-installed"
    transformers_version = package_version("transformers") if transformers_installed else "not-installed"
    torch_cuda_available = False
    torch_cuda_version = "unavailable"
    if torch_installed:
        import torch
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_version = str(torch.version.cuda)
    if nvidia_ok and torch_cuda_available and transformers_installed:
        status = "READY"
        recommendation = "Run local LLM generation with a small cohort subset, then scale if validation passes."
    elif nvidia_ok and torch_installed and not torch_cuda_available:
        status = "BLOCKED_TORCH_CPU"
        recommendation = (
            "Install a CUDA-enabled PyTorch build in the project environment before claiming GPU local-LLM "
            "replication, for example the official PyTorch CUDA wheel matching the installed driver."
        )
    else:
        status = "BLOCKED_ENVIRONMENT"
        recommendation = "Install/verify GPU drivers, CUDA-enabled PyTorch, and transformers before running replication."
    return EnvironmentAudit(
        nvidia_smi_available=nvidia_ok,
        gpu_name=gpu_name,
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        transformers_installed=transformers_installed,
        transformers_version=transformers_version,
        accelerate_installed=accelerate_installed,
        bitsandbytes_installed=bitsandbytes_installed,
        recommended_status=status,
        recommendation=recommendation,
    )


def render_prompt(tokenizer: Any, record: PromptRecord) -> str:
    messages = augment_messages(record.payload.get("messages", []))
    if hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def load_local_model(args: argparse.Namespace, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=local_files_only, trust_remote_code=args.trust_remote_code)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_one_local(
    record: PromptRecord,
    tokenizer: Any,
    model: Any,
    device: str,
    args: argparse.Namespace,
) -> tuple[LocalGenerationRecord | None, dict[str, Any]]:
    import torch

    prompt = render_prompt(tokenizer, record)
    request_hash = stable_json_hash({"model": args.model_id, "prompt": prompt, "temperature": args.temperature})
    started_at = time.time()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature > 0.0:
        generation_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    raw_record = {
        "cohort_id": record.cohort_id,
        "provider": "local_transformers",
        "model": args.model_id,
        "request_hash": request_hash,
        "latency_seconds": round(time.time() - started_at, 3),
        "raw_text": content,
    }
    try:
        validated = validate_feature_record(
            {"cohort_id": record.cohort_id, "features": json.loads(content)},
            allow_extra_fields=False,
        )
    except (json.JSONDecodeError, ValueError):
        from llm_event_generation import parse_feature_content
        validated = parse_feature_content(record.cohort_id, content)
    parsed = LocalGenerationRecord(record.cohort_id, "local_transformers", args.model_id, request_hash, 1, raw_record["latency_seconds"], validated.features)
    return parsed, raw_record


def write_environment_audit(audit: EnvironmentAudit, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "local_llm_environment_audit.json").write_text(json.dumps(asdict(audit), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Local/Open-Source LLM Prior Replication Environment Audit",
        "",
        f"- Status: `{audit.recommended_status}`",
        f"- GPU from `nvidia-smi`: `{audit.gpu_name}`",
        f"- Torch installed: `{audit.torch_installed}`",
        f"- Torch version: `{audit.torch_version}`",
        f"- Torch CUDA available: `{audit.torch_cuda_available}`",
        f"- Torch CUDA version: `{audit.torch_cuda_version}`",
        f"- Transformers installed: `{audit.transformers_installed}`",
        f"- Transformers version: `{audit.transformers_version}`",
        f"- Accelerate installed: `{audit.accelerate_installed}`",
        f"- BitsAndBytes installed: `{audit.bitsandbytes_installed}`",
        f"- Default model id: `{args.model_id}`",
        f"- Recommendation: {audit.recommendation}",
        "",
        "## Interpretation",
        "",
        "This audit is a reproducibility guardrail for the open-source LLM control. A paper claim that local "
        "LLM priors were generated on GPU should be made only when this status is `READY` and a non-empty "
        "`local_llm_event_features_normalized.csv` is present.",
    ]
    (output_dir / "local_llm_environment_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_records(records: list[LocalGenerationRecord], output_dir: Path) -> None:
    jsonl_path = output_dir / "local_llm_event_features.jsonl"
    csv_path = output_dir / "local_llm_event_features_normalized.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) + "\n")
    fieldnames = ["cohort_id", *NUMERIC_FEATURES, *TEXT_FEATURES]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, str | float] = {"cohort_id": record.cohort_id}
            row.update(record.features)
            writer.writerow(row)


def write_invalid(records: list[dict[str, Any]], output_dir: Path) -> None:
    path = output_dir / "local_llm_event_feature_errors.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def compare_with_reference(records: list[LocalGenerationRecord], reference_path: Path, output_dir: Path) -> pd.DataFrame:
    if not records or not reference_path.exists():
        return pd.DataFrame()
    local_rows = [{"cohort_id": record.cohort_id, **record.features} for record in records]
    local = pd.DataFrame(local_rows)
    reference = pd.read_csv(reference_path)
    merged = local.merge(reference, on="cohort_id", suffixes=("_local", "_reference"))
    rows: list[dict[str, float | str | int]] = []
    for column in NUMERIC_FEATURES:
        left = merged[f"{column}_local"].astype(float)
        right = merged[f"{column}_reference"].astype(float)
        rows.append(
            {
                "feature": column,
                "n": int(len(merged)),
                "mean_abs_delta": float(np.mean(np.abs(left - right))),
                "pearson": float(left.corr(right, method="pearson")) if len(merged) >= 2 else np.nan,
                "spearman": float(left.corr(right, method="spearman")) if len(merged) >= 2 else np.nan,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "local_vs_reference_prior_comparison.csv", index=False)
    return comparison


def write_run_summary(
    audit: EnvironmentAudit,
    selected_count: int,
    success_count: int,
    error_count: int,
    comparison: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    error_records: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# Local/Open-Source LLM Prior Replication Report",
        "",
        f"- Environment status: `{audit.recommended_status}`",
        f"- Model id: `{args.model_id}`",
        f"- Input prompt file: `{args.input_jsonl}`",
        f"- Selected cohorts: `{selected_count}`",
        f"- Successful local priors: `{success_count}`",
        f"- Invalid/error records: `{error_count}`",
        f"- Output directory: `{output_dir}`",
        f"- Error log: `{output_dir / 'local_llm_event_feature_errors.jsonl'}`",
    ]
    if error_records:
        lines.extend(["", "## Runtime Errors", "", "| Cohort | Error |", "|---|---|"])
        for record in error_records[:10]:
            error_text = " ".join(str(record.get("error", "unknown")).split())[:240]
            lines.append(f"| {record.get('cohort_id', 'runtime')} | {error_text} |")
        if len(error_records) > 10:
            lines.append(f"| ... | {len(error_records) - 10} additional errors omitted from report. |")
    if comparison.empty:
        lines.extend(
            [
                "",
                "No local-vs-reference comparison was generated. This is expected for `--audit-only` runs or when "
                "the local model environment is not ready.",
            ]
        )
    else:
        lines.extend(["", "## Local vs GPT-Reference Prior Agreement", "", "| Feature | n | MAE | Pearson | Spearman |", "|---|---:|---:|---:|---:|"])
        for row in comparison.itertuples(index=False):
            lines.append(
                f"| {row.feature} | {row.n} | {row.mean_abs_delta:.4f} | {row.pearson:.4f} | {row.spearman:.4f} |"
            )
        mean_abs_delta = float(comparison["mean_abs_delta"].mean())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"The local open-source model produced valid structured priors for `{success_count}` cohorts "
                f"with mean feature MAE `{mean_abs_delta:.4f}` versus the GPT-reference priors. Treat this as "
                "a reproducibility and sensitivity control rather than a drop-in replacement for the main prior "
                "source unless a larger cohort run shows stable agreement.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paper Use",
            "",
            (
                "This artifact supports a completed small-sample open-source GPU LLM control."
                if success_count > 0
                else "This artifact should be cited as an implemented replication protocol and environment audit, "
                "not as completed generation evidence."
            ),
        ]
    )
    (output_dir / "local_llm_prior_replication_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in args.output_dir.glob("local_llm_*"):
            if path.is_file():
                path.unlink()
    audit = inspect_environment()
    write_environment_audit(audit, args.output_dir, args)
    records = select_records(
        load_prompt_records(args.input_jsonl),
        limit=args.limit,
        offset=args.offset,
        selection=args.selection,
        cohort_ids=args.cohort_ids,
    )
    if args.audit_only:
        write_run_summary(audit, len(records), 0, 0, pd.DataFrame(), args.output_dir, args)
        LOGGER.info("Audit-only run complete: %s", audit.recommended_status)
        return
    if audit.recommended_status != "READY" and not args.allow_cpu:
        write_run_summary(audit, len(records), 0, 0, pd.DataFrame(), args.output_dir, args)
        raise RuntimeError(audit.recommendation)
    device = "cuda" if audit.torch_cuda_available else "cpu"
    try:
        tokenizer, model = load_local_model(args, device)
    except (OSError, RuntimeError, ValueError) as error:
        error_records = [{"cohort_id": "runtime_model_load", "error": str(error)}]
        write_invalid(error_records, args.output_dir)
        write_run_summary(
            audit,
            len(records),
            0,
            len(error_records),
            pd.DataFrame(),
            args.output_dir,
            args,
            error_records,
        )
        LOGGER.error("Local model load failed: %s", error)
        return
    parsed_records: list[LocalGenerationRecord] = []
    error_records: list[dict[str, Any]] = []
    for record in records:
        try:
            parsed, raw = generate_one_local(record, tokenizer, model, device, args)
            parsed_records.append(parsed)
            LOGGER.info("Generated local prior for %s in %.3fs", record.cohort_id, raw["latency_seconds"])
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            error_records.append({"cohort_id": record.cohort_id, "error": str(error)})
            LOGGER.warning("Local prior failed for %s: %s", record.cohort_id, error)
    write_records(parsed_records, args.output_dir)
    write_invalid(error_records, args.output_dir)
    comparison = compare_with_reference(parsed_records, args.reference_features, args.output_dir)
    write_run_summary(
        audit,
        len(records),
        len(parsed_records),
        len(error_records),
        comparison,
        args.output_dir,
        args,
        error_records,
    )


if __name__ == "__main__":
    main()
