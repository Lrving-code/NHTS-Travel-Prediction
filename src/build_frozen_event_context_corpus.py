"""Build a frozen event-context corpus for label-free LLM prior generation.

The corpus is a provenance artifact: it constrains what an LLM may use when
producing event-response priors. It does not calibrate the NHTS 2022 target and
does not use NHTS 2022 trip-count outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "event_context_corpus"
EXTERNAL_DIR = PROJECT_ROOT / "outputs" / "external_validation"
PROSPECTIVE_CONTEXT_PATH = PROJECT_ROOT / "plan" / "prospective_event_context_2022.md"

ACS_PATH = EXTERNAL_DIR / "acs_commute_mechanism_validation.csv"
BTS_PATH = EXTERNAL_DIR / "bts_annual_mobility_summary.csv"
PSRC_YEAR_PATH = EXTERNAL_DIR / "psrc_household_year_summary.csv"
PSRC_METRICS_PATH = EXTERNAL_DIR / "psrc_household_external_validation_metrics.csv"
DATASET_CANDIDATES_PATH = EXTERNAL_DIR / "external_dataset_candidates.csv"

FREEZE_DATE = "2026-06-28"
FORBIDDEN_TERMS = [
    "CNTTDHH",
    "WTHHFIN",
    "HOUSEID",
    "2022 NHTS target",
    "2022 NHTS outcome",
    "NHTS 2022 observed",
]


@dataclass(frozen=True)
class ContextFact:
    """One frozen external fact available to the LLM context."""

    source_id: str
    mechanism: str
    provider: str
    url: str
    source_granularity: str
    allowed_use: str
    forbidden_use: str
    evidence_summary: str
    numeric_fact: str
    status: str


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source table: {path}")
    return pd.read_csv(path)


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def acs_context_facts() -> list[ContextFact]:
    frame = read_csv(ACS_PATH)
    facts: list[ContextFact] = []
    for row in frame.itertuples(index=False):
        source_id = "ACS_REMOTE_WORK" if row.mechanism == "remote_work_substitution" else "ACS_TRANSIT_COMMUTE"
        facts.append(
            ContextFact(
                source_id=source_id,
                mechanism=str(row.mechanism),
                provider="U.S. Census Bureau",
                url=str(row.source),
                source_granularity="National ACS commute-mode aggregate",
                allowed_use=(
                    "Use as direction-only mechanism evidence for event priors; map to cohort exposure through "
                    "worker count, income, urban context, rail exposure, and vehicle access."
                ),
                forbidden_use=(
                    "Do not use this aggregate to infer exact NHTS household trip counts, exact mode shares, "
                    "or target-year calibration constants."
                ),
                evidence_summary=(
                    f"{row.external_measure} changed from {format_pct(float(row.share_2019_pct))} in 2019 "
                    f"to {format_pct(float(row.share_2022_pct))} in 2022."
                ),
                numeric_fact=(
                    f"2019={format_pct(float(row.share_2019_pct))}; "
                    f"2021={format_pct(float(row.share_2021_pct))}; "
                    f"2022={format_pct(float(row.share_2022_pct))}; "
                    f"2019_to_2022_change={float(row.change_2019_2022_pct_points):+.1f} percentage points"
                ),
                status="frozen_allowed_context",
            )
        )
    return facts


def bts_context_fact() -> ContextFact:
    frame = read_csv(BTS_PATH)
    rows = frame.set_index("year")
    trips_2019 = float(rows.loc[2019, "trips_per_person_per_day"])
    trips_2022 = float(rows.loc[2022, "trips_per_person_per_day"])
    stay_2019 = float(rows.loc[2019, "stay_home_share"])
    stay_2022 = float(rows.loc[2022, "stay_home_share"])
    trip_change = (trips_2022 / trips_2019 - 1.0) * 100.0
    stay_change = (stay_2022 - stay_2019) * 100.0
    return ContextFact(
        source_id="BTS_DEVICE_MOBILITY_COMPATIBILITY",
        mechanism="aggregate_mobility_recovery_guardrail",
        provider="U.S. Bureau of Transportation Statistics / University of Maryland",
        url="https://data.bts.gov/Research-and-Statistics/Daily-Mobility-Statistics-National-and-State/aksz-j95y",
        source_granularity="Daily national mobile-device aggregate",
        allowed_use=(
            "Use as a compatibility guardrail showing that external device mobility has different measurement "
            "semantics from NHTS travel diaries."
        ),
        forbidden_use=(
            "Do not use BTS trips per person as a direct numeric label or calibration target for NHTS household CNTTDHH."
        ),
        evidence_summary=(
            "BTS device trips per person recovered near the 2019 aggregate level by 2022, while stay-home share "
            "remained above 2019."
        ),
        numeric_fact=(
            f"trips_per_person_per_day: 2019={trips_2019:.4f}, 2022={trips_2022:.4f}, "
            f"relative_change={trip_change:+.2f}%; stay_home_share_change={stay_change:+.2f} percentage points"
        ),
        status="frozen_guardrail_context",
    )


def psrc_context_fact() -> ContextFact:
    year_summary = read_csv(PSRC_YEAR_PATH)
    metrics = read_csv(PSRC_METRICS_PATH)
    psrc_source = "https://psrc-psregcncl.hub.arcgis.com/datasets/PSREGCNCL::household-travel-survey-households/about"
    pre = metrics.loc[
        (metrics["method"] == "psrc_pre_pandemic_xgboost") & (metrics["test_year"].astype(str) == "2023")
    ].iloc[0]
    acs = metrics.loc[
        (metrics["method"] == "acs_remote_work_suppression_adapter") & (metrics["test_year"].astype(str) == "2023")
    ].iloc[0]
    year_column = "survey_year" if "survey_year" in year_summary.columns else "year"
    summary = year_summary.set_index(year_column)
    trips_2019 = float(summary.loc[2019, "weighted_trips_per_day"])
    trips_2023 = float(summary.loc[2023, "weighted_trips_per_day"])
    return ContextFact(
        source_id="PSRC_HOUSEHOLD_EXTERNAL_REPLICATION",
        mechanism="household_level_external_pre_post_replication",
        provider="Puget Sound Regional Council",
        url=psrc_source,
        source_granularity="Regional household travel survey microdata",
        allowed_use=(
            "Use only as external replication evidence after model design; do not expose target-year PSRC labels "
            "to NHTS prior generation."
        ),
        forbidden_use=(
            "Do not use PSRC target-year labels to tune NHTS event-prior scores, thresholds, or leaf values."
        ),
        evidence_summary=(
            "A fixed ACS-derived remote-work suppression factor improves PSRC 2017+2019 to 2023 transfer, "
            "supporting the event-adaptation principle on independent household microdata."
        ),
        numeric_fact=(
            f"PSRC weighted trips/day: 2019={trips_2019:.4f}, 2023={trips_2023:.4f}; "
            f"2023 pre-pandemic XGBoost wMAE={float(pre.weighted_mae):.4f}, wBias={float(pre.weighted_bias):+.4f}; "
            f"ACS adapter wMAE={float(acs.weighted_mae):.4f}, wBias={float(acs.weighted_bias):+.4f}"
        ),
        status="external_validation_only_not_prompt_context",
    )


def candidate_context_facts() -> list[ContextFact]:
    if not DATASET_CANDIDATES_PATH.exists():
        return []
    frame = pd.read_csv(DATASET_CANDIDATES_PATH)
    rows = frame.loc[frame["status"].astype(str).str.contains("candidate", case=False, na=False)]
    facts = []
    for row in rows.itertuples(index=False):
        facts.append(
            ContextFact(
                source_id="CANDIDATE_" + str(row.dataset).upper().replace(" ", "_").replace("/", "_")[:48],
                mechanism="candidate_future_context",
                provider=str(row.provider),
                url=str(row.url),
                source_granularity=str(row.granularity),
                allowed_use="Do not use in committed priors until the source is frozen and summarized.",
                forbidden_use="Do not treat candidate sources as evidence for current numeric claims.",
                evidence_summary=str(row.project_use),
                numeric_fact="not_frozen",
                status="candidate_not_used",
            )
        )
    return facts


def build_facts() -> list[ContextFact]:
    return acs_context_facts() + [bts_context_fact(), psrc_context_fact()] + candidate_context_facts()


def fact_to_prompt_block(fact: ContextFact) -> str:
    return (
        f"[{fact.source_id}] {fact.mechanism}\n"
        f"- Provider: {fact.provider}\n"
        f"- URL: {fact.url}\n"
        f"- Status: {fact.status}\n"
        f"- Evidence: {fact.evidence_summary}\n"
        f"- Numeric fact: {fact.numeric_fact}\n"
        f"- Allowed use: {fact.allowed_use}\n"
        f"- Forbidden use: {fact.forbidden_use}\n"
    )


def write_sources(facts: list[ContextFact]) -> Path:
    path = OUTPUT_DIR / "frozen_event_context_sources.csv"
    frame = pd.DataFrame([asdict(fact) for fact in facts])
    frame.insert(0, "freeze_date", FREEZE_DATE)
    frame.to_csv(path, index=False)
    return path


def write_json(facts: list[ContextFact]) -> Path:
    path = OUTPUT_DIR / "frozen_event_context_facts.json"
    payload = {
        "freeze_date": FREEZE_DATE,
        "label_policy": "No NHTS 2022 target labels, sample weights, household IDs, or aggregate target outcomes.",
        "facts": [asdict(fact) for fact in facts],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_prompt_context(facts: list[ContextFact]) -> Path:
    path = OUTPUT_DIR / "frozen_event_context_prompt.md"
    allowed_facts = [fact for fact in facts if fact.status in {"frozen_allowed_context", "frozen_guardrail_context"}]
    lines = [
        "# Frozen Event Context for LLM Event-Prior Generation",
        "",
        f"Freeze date: `{FREEZE_DATE}`",
        "",
        "## Role",
        "",
        "Use only the frozen context facts below plus the input household-cohort covariates to generate structured "
        "event-response priors. Do not use general model memory about NHTS 2022 outcomes.",
        "",
        "## Label Leakage Rules",
        "",
        "- Do not use NHTS 2022 trip-count labels, sample weights, household IDs, or aggregate target outcomes.",
        "- Do not infer exact household trip counts, exact mode shares, or exact purpose shares.",
        "- Produce mechanism priors only: trip suppression, remote-work substitution, transit avoidance, delivery substitution, and recovery sensitivity.",
        "- Treat PSRC as external validation evidence, not as prompt context for NHTS prior generation.",
        "",
        "## Frozen Context Facts",
        "",
    ]
    for fact in allowed_facts:
        lines.extend([fact_to_prompt_block(fact), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_audit(facts: list[ContextFact], sources_path: Path, json_path: Path, prompt_path: Path) -> Path:
    path = OUTPUT_DIR / "frozen_event_context_audit.md"
    facts_payload = "\n".join(f"{fact.evidence_summary}\n{fact.numeric_fact}" for fact in facts)
    forbidden_hits = [term for term in FORBIDDEN_TERMS if term.lower() in facts_payload.lower()]
    status_counts = pd.Series([fact.status for fact in facts]).value_counts().to_dict()
    prospective_context_exists = PROSPECTIVE_CONTEXT_PATH.exists()
    allowed_count = sum(fact.status == "frozen_allowed_context" for fact in facts)
    guardrail_count = sum(fact.status == "frozen_guardrail_context" for fact in facts)
    validation_only_count = sum(fact.status == "external_validation_only_not_prompt_context" for fact in facts)
    candidate_count = sum(fact.status == "candidate_not_used" for fact in facts)
    lines = [
        "# Frozen Event-Context Corpus Audit",
        "",
        f"Freeze date: `{FREEZE_DATE}`",
        "",
        "## Summary",
        "",
        f"- Frozen allowed prompt-context facts: `{allowed_count}`",
        f"- Frozen guardrail context facts: `{guardrail_count}`",
        f"- External validation-only facts: `{validation_only_count}`",
        f"- Candidate sources not used in current priors: `{candidate_count}`",
        f"- Prospective event context file exists: `{prospective_context_exists}`",
        f"- Forbidden target-field hits in allowed fact summaries: `{len(forbidden_hits)}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Source manifest: `{sources_path.relative_to(PROJECT_ROOT)}`",
            f"- JSON facts: `{json_path.relative_to(PROJECT_ROOT)}`",
            f"- Prompt context: `{prompt_path.relative_to(PROJECT_ROOT)}`",
            "",
            "## Interpretation",
            "",
            "This corpus turns the earlier prospective-context plan into an auditable source package. "
            "The LLM can be instructed to use the frozen ACS and BTS facts for mechanism direction and compatibility "
            "guardrails, while PSRC remains external validation-only. The corpus still does not prove full prospective "
            "deployment, but it materially reduces the risk that the priors are an unconstrained memory-based prompt artifact.",
            "",
        ]
    )
    if forbidden_hits:
        lines.extend(["## Forbidden Hits", "", *[f"- `{term}`" for term in forbidden_hits], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    facts = build_facts()
    sources_path = write_sources(facts)
    json_path = write_json(facts)
    prompt_path = write_prompt_context(facts)
    audit_path = write_audit(facts, sources_path, json_path, prompt_path)
    print(f"Wrote {sources_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
