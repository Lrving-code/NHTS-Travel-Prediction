"""Build SHA256 integrity manifests for the frozen event-context package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "event_context_corpus"

TRACKED_ARTIFACTS = [
    ("source_input", "outputs/external_validation/acs_commute_mechanism_validation.csv"),
    ("source_input", "outputs/external_validation/bts_annual_mobility_summary.csv"),
    ("source_input", "outputs/external_validation/psrc_household_year_summary.csv"),
    ("source_input", "outputs/external_validation/psrc_household_external_validation_metrics.csv"),
    ("source_input", "outputs/external_validation/external_dataset_candidates.csv"),
    ("prompt_input", "outputs/llm_event_features/household_cohort_profiles.csv"),
    ("prompt_input", "outputs/llm_event_features/event_feature_schema.json"),
    ("frozen_corpus", "outputs/event_context_corpus/frozen_event_context_sources.csv"),
    ("frozen_corpus", "outputs/event_context_corpus/frozen_event_context_facts.json"),
    ("frozen_corpus", "outputs/event_context_corpus/frozen_event_context_prompt.md"),
    ("frozen_corpus", "outputs/event_context_corpus/frozen_event_context_audit.md"),
    ("frozen_prompt", "outputs/event_context_corpus/frozen_context_batch_prompt_summary.md"),
    ("frozen_prompt", "outputs/event_context_corpus/frozen_context_batch_prompts.jsonl"),
    ("frozen_prompt_leakage_audit", "outputs/event_context_corpus/leakage_audit/llm_input_leakage_audit.csv"),
    ("frozen_prompt_leakage_audit", "outputs/event_context_corpus/leakage_audit/llm_guardrail_instruction_check.csv"),
    ("frozen_prompt_leakage_audit", "outputs/event_context_corpus/leakage_audit/llm_input_leakage_audit_report.md"),
]


@dataclass(frozen=True)
class IntegrityRow:
    """One file-level integrity record."""

    role: str
    relative_path: str
    exists: bool
    size_bytes: int
    line_count: int
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def build_rows() -> list[IntegrityRow]:
    rows: list[IntegrityRow] = []
    for role, relative_path in TRACKED_ARTIFACTS:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            rows.append(
                IntegrityRow(
                    role=role,
                    relative_path=relative_path,
                    exists=True,
                    size_bytes=path.stat().st_size,
                    line_count=count_lines(path),
                    sha256=sha256_file(path),
                )
            )
        else:
            rows.append(
                IntegrityRow(
                    role=role,
                    relative_path=relative_path,
                    exists=False,
                    size_bytes=0,
                    line_count=0,
                    sha256="",
                )
            )
    return rows


def write_manifest(rows: list[IntegrityRow]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "frozen_context_integrity_manifest.csv"
    json_path = OUTPUT_DIR / "frozen_context_integrity_manifest.json"
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(csv_path, index=False)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(rows),
        "missing_count": int((~frame["exists"]).sum()),
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def write_report(rows: list[IntegrityRow], csv_path: Path, json_path: Path) -> Path:
    report_path = OUTPUT_DIR / "frozen_context_integrity_report.md"
    frame = pd.DataFrame([asdict(row) for row in rows])
    missing = frame.loc[~frame["exists"]]
    total_bytes = int(frame["size_bytes"].sum())
    lines = [
        "# Frozen Event-Context Integrity Manifest",
        "",
        "## Summary",
        "",
        f"- Artifacts tracked: `{len(frame)}`",
        f"- Missing artifacts: `{len(missing)}`",
        f"- Total bytes hashed: `{total_bytes}`",
        f"- CSV manifest: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- JSON manifest: `{json_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## Interpretation",
        "",
        "This manifest gives the frozen event-context corpus and prompt package stable file-level SHA256 hashes. "
        "It supports provenance review by making later context or prompt edits detectable.",
        "",
        "## Artifact Hashes",
        "",
        "| Role | Artifact | Exists | Size bytes | Lines | SHA256 prefix |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        prefix = row.sha256[:16] if row.sha256 else ""
        lines.append(
            f"| {row.role} | `{row.relative_path}` | {row.exists} | {row.size_bytes} | {row.line_count} | `{prefix}` |"
        )
    if not missing.empty:
        lines.extend(["", "## Missing Artifacts", ""])
        for row in missing.itertuples(index=False):
            lines.append(f"- `{row.relative_path}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    rows = build_rows()
    csv_path, json_path = write_manifest(rows)
    report_path = write_report(rows, csv_path, json_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
