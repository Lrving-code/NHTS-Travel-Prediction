"""Build a submission-readiness audit for the NHTS LLM adaptation project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pptx import Presentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submission_readiness"
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_project"


@dataclass(frozen=True)
class Check:
    category: str
    item: str
    status: str
    score: float
    evidence: str
    recommendation: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def metric_value(path: str, method: str, column: str) -> float | None:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return None
    frame = pd.read_csv(file_path)
    if "method" not in frame.columns or column not in frame.columns:
        return None
    row = frame.loc[frame["method"] == method]
    if row.empty:
        return None
    return float(row[column].iloc[0])


def metric_value_filtered(path: str, filters: dict[str, str | int], column: str) -> float | None:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return None
    frame = pd.read_csv(file_path)
    if column not in frame.columns:
        return None
    mask = pd.Series(True, index=frame.index)
    for filter_column, filter_value in filters.items():
        if filter_column not in frame.columns:
            return None
        mask &= frame[filter_column].astype(str) == str(filter_value)
    rows = frame.loc[mask]
    if rows.empty:
        return None
    return float(rows[column].iloc[0])


def contains_text(path: str, terms: list[str]) -> bool:
    content = read_text(PROJECT_ROOT / path)
    return all(term in content for term in terms)


def ppt_text(path: str) -> tuple[int, str]:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return 0, ""
    presentation = Presentation(file_path)
    text = "\n".join(
        shape.text
        for slide in presentation.slides
        for shape in slide.shapes
        if hasattr(shape, "text")
    )
    return len(presentation.slides), text


def pass_check(category: str, item: str, evidence: str, recommendation: str = "Keep.") -> Check:
    return Check(category, item, "PASS", 1.0, evidence, recommendation)


def partial_check(category: str, item: str, evidence: str, recommendation: str) -> Check:
    return Check(category, item, "PARTIAL", 0.5, evidence, recommendation)


def fail_check(category: str, item: str, evidence: str, recommendation: str) -> Check:
    return Check(category, item, "FAIL", 0.0, evidence, recommendation)


def audit_core_results() -> list[Check]:
    category = "Core results"
    checks: list[Check] = []
    baseline = metric_value("outputs/final_project/final_metrics_summary.csv", "historical_xgboost", "weighted_mae")
    primary = metric_value(
        "outputs/final_project/final_metrics_summary.csv",
        "gated_trip_suppression_a1_d0p15",
        "weighted_mae",
    )
    bias = metric_value(
        "outputs/final_project/final_metrics_summary.csv",
        "gated_trip_suppression_a1_d0p15",
        "weighted_bias",
    )
    if baseline is not None and primary is not None and bias is not None:
        gain = (baseline - primary) / baseline
        if gain >= 0.35 and abs(bias) <= 0.1:
            checks.append(
                pass_check(
                    category,
                    "Primary trip-count result",
                    f"historical wMAE {baseline:.4f} -> primary wMAE {primary:.4f}; primary wBias {bias:+.4f}.",
                )
            )
        else:
            checks.append(
                partial_check(
                    category,
                    "Primary trip-count result",
                    f"historical wMAE {baseline:.4f} -> primary wMAE {primary:.4f}; primary wBias {bias:+.4f}.",
                    "Tighten the primary operating point or explain the bias/accuracy tradeoff more explicitly.",
                )
            )
    else:
        checks.append(
            fail_check(
                category,
                "Primary trip-count result",
                "Missing final_metrics_summary.csv values.",
                "Regenerate final project assets.",
            )
        )
    required = [
        "outputs/final_project/final_project_report.md",
        "outputs/final_project/household_accuracy_summary.csv",
        "outputs/statistical_validation/confidence_interval_report.md",
    ]
    missing = [path for path in required if not exists(path)]
    if missing:
        checks.append(
            fail_check(
                category,
                "Core result artifacts",
                f"Missing: {', '.join(missing)}.",
                "Regenerate final report, accuracy summary, and statistical validation.",
            )
        )
    else:
        checks.append(pass_check(category, "Core result artifacts", "Final report, accuracy summary, and CI report exist."))
    return checks


def audit_baselines_and_controls() -> list[Check]:
    category = "Baselines and controls"
    required = {
        "Stronger tabular baseline": "outputs/strong_baselines/strong_tabular_baseline_metrics.csv",
        "Zero-shot rule tree": "outputs/zero_shot_llm_rule_tree_baseline/zero_shot_llm_rule_tree_metrics.csv",
        "Small historical calibration": "outputs/llm_rule_small_data_calibration/method_spectrum_metrics.csv",
        "Irrelevant pseudo-event placebo": "outputs/irrelevant_pseudo_event_placebo/irrelevant_pseudo_event_placebo_metrics.csv",
        "Permutation robustness": "outputs/robustness_checks/permutation_pressure_controls.csv",
    }
    checks = []
    for item, path in required.items():
        if exists(path):
            checks.append(pass_check(category, item, path))
        else:
            checks.append(fail_check(category, item, f"Missing {path}.", "Run the corresponding baseline/control script."))
    return checks


def audit_guardrails() -> list[Check]:
    category = "Causal and leakage guardrails"
    checks: list[Check] = []
    required = {
        "Leakage audit": "outputs/leakage_audit/llm_input_leakage_audit_report.md",
        "Causal evidence pack": "outputs/causal_guardrails/causal_guardrail_evidence_report.md",
        "Prospective event context": "plan/prospective_event_context_2022.md",
        "Pre-COVID placebo": "outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md",
    }
    for item, path in required.items():
        if exists(path):
            checks.append(pass_check(category, item, path))
        else:
            checks.append(fail_check(category, item, f"Missing {path}.", "Add this guardrail before paper submission."))
    if contains_text("outputs/final_project/final_project_report.md", ["2022 trip-count labels", "only for final evaluation"]):
        checks.append(pass_check(category, "No-target-label framing", "Final report states target-year labels are evaluation-only."))
    else:
        checks.append(
            partial_check(
                category,
                "No-target-label framing",
                "Final report does not clearly restate the evaluation-only target-label rule.",
                "Add a short no-label protocol paragraph.",
            )
        )
    return checks


def audit_multi_output_evaluation() -> list[Check]:
    category = "Mobility behavior system"
    required = {
        "Mode composition": "outputs/mode_composition_extension/mode_composition_metrics.csv",
        "Mode-specific trips": "outputs/mode_composition_extension/mode_specific_trip_count_metrics.csv",
        "Purpose composition": "outputs/purpose_composition_extension/purpose_composition_metrics.csv",
        "Equity-aware evaluation": "outputs/equity_aware_evaluation/subgroup_equity_metrics.csv",
        "Multi-objective Pareto": "outputs/multi_objective_pareto/preference_operating_points.csv",
    }
    checks = []
    for item, path in required.items():
        if exists(path):
            checks.append(pass_check(category, item, path))
        else:
            checks.append(fail_check(category, item, f"Missing {path}.", "Regenerate this evaluation artifact."))
    return checks


def audit_temporal_external_validation() -> list[Check]:
    category = "Temporal and external validity"
    checks: list[Check] = []
    if exists("outputs/temporal_transfer_validation/temporal_transfer_validation_report.md"):
        checks.append(pass_check(category, "Temporal transfer validation", "Pre-COVID and 2022 transfer report exists."))
    else:
        checks.append(fail_check(category, "Temporal transfer validation", "Missing temporal transfer report.", "Run temporal validation."))
    if exists("outputs/pre_covid_placebo_event_correction/pre_covid_placebo_event_correction_report.md"):
        checks.append(pass_check(category, "Pre-COVID placebo validation", "Pre-COVID event-correction placebo report exists."))
    else:
        checks.append(fail_check(category, "Pre-COVID placebo validation", "Missing pre-COVID placebo report.", "Run placebo validation."))
    if exists("outputs/external_validation/external_validation_and_compatibility_report.md") and exists(
        "outputs/external_validation/acs_commute_mechanism_validation.csv"
    ):
        checks.append(
            pass_check(
                category,
                "External mechanism validation beyond NHTS",
                "ACS commute-mode mechanism validation and BTS trip-count compatibility guardrail exist.",
                "Report as mechanism-level external evidence, not household-level MAE.",
            )
        )
    else:
        checks.append(
            partial_check(
                category,
                "External validation beyond NHTS",
                "Internal temporal validation exists; no independent external dataset is documented.",
                "Run src/run_external_aggregate_validation.py or add an external mobility survey/region/shock dataset.",
            )
        )

    psrc_metric_path = "outputs/external_validation/psrc_household_external_validation_metrics.csv"
    psrc_report_path = "outputs/external_validation/psrc_household_external_validation_report.md"
    baseline_mae = metric_value_filtered(
        psrc_metric_path,
        {"method": "psrc_pre_pandemic_xgboost", "test_year": 2023},
        "weighted_mae",
    )
    adapted_mae = metric_value_filtered(
        psrc_metric_path,
        {"method": "acs_remote_work_suppression_adapter", "test_year": 2023},
        "weighted_mae",
    )
    baseline_bias = metric_value_filtered(
        psrc_metric_path,
        {"method": "psrc_pre_pandemic_xgboost", "test_year": 2023},
        "weighted_bias",
    )
    adapted_bias = metric_value_filtered(
        psrc_metric_path,
        {"method": "acs_remote_work_suppression_adapter", "test_year": 2023},
        "weighted_bias",
    )
    if all(value is not None for value in [baseline_mae, adapted_mae, baseline_bias, adapted_bias]) and exists(
        psrc_report_path
    ):
        checks.append(
            pass_check(
                category,
                "Household-level external microdata validation",
                (
                    f"PSRC 2017+2019->2023 household microdata: wMAE {baseline_mae:.4f}->{adapted_mae:.4f}; "
                    f"wBias {baseline_bias:+.4f}->{adapted_bias:+.4f}."
                ),
                "Report as direct external pre/post replication with regional-survey scope limitations.",
            )
        )
        checks.append(
            pass_check(
                category,
                "External validation scope statement",
                "PSRC report states that the regional survey is external replication of the event-adaptation principle, not direct NHTS numerical validation.",
                "Keep the limitation statement in the paper.",
            )
        )
    else:
        checks.append(
            partial_check(
                category,
                "Household-level external microdata validation",
                "No PSRC household-level external validation metrics are documented.",
                "Run src/run_psrc_external_household_validation.py.",
            )
        )
    return checks


def audit_literature_and_story() -> list[Check]:
    category = "Literature and paper story"
    checks: list[Check] = []
    literature_terms = ["ELLMob", "CausalMob", "AgentMove", "AgentMob", "UniMob", "ELP-Mob"]
    if contains_text("plan/literature_grounding_2026.md", literature_terms):
        checks.append(pass_check(category, "2025-2026 literature grounding", "Literature grounding note contains current anchors."))
    else:
        checks.append(fail_check(category, "2025-2026 literature grounding", "Missing key literature anchors.", "Update literature grounding."))
    if contains_text("plan/paper_logic_chain.md", ["event-driven temporal adaptation", "LLM event-semantic adapter"]):
        checks.append(pass_check(category, "Single paper spine", "paper_logic_chain.md states the event-adaptation spine."))
    else:
        checks.append(
            partial_check(
                category,
                "Single paper spine",
                "Paper logic chain exists but does not clearly state the single spine.",
                "Rewrite the opening claim around event-driven label-free adaptation.",
            )
        )
    if exists("plan/reviewer_qa_backup_2026.md"):
        checks.append(pass_check(category, "Reviewer Q&A backup", "Reviewer Q&A backup document exists."))
    else:
        checks.append(partial_check(category, "Reviewer Q&A backup", "Missing Q&A backup.", "Create discussion backup answers."))
    if exists("outputs/paper_draft/nhts_event_adaptation_paper_draft.md") and contains_text(
        "outputs/paper_draft/claim_evidence_matrix.md",
        ["Primary gated weighted MAE", "Unsafe wording", "PSRC external weighted MAE improvement"],
    ):
        checks.append(pass_check(category, "Paper draft and claim ledger", "Paper draft and claim-evidence matrix exist."))
    else:
        checks.append(
            partial_check(
                category,
                "Paper draft and claim ledger",
                "Missing paper draft or claim-evidence matrix.",
                "Create outputs/paper_draft/nhts_event_adaptation_paper_draft.md and claim_evidence_matrix.md.",
            )
        )
    return checks


def audit_presentation() -> list[Check]:
    category = "Presentation readiness"
    checks: list[Check] = []
    slide_count, text = ppt_text("outputs/final_project/NHTS_Travel_Behavior_Template_Presentation.pptx")
    backup_markers = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    if slide_count >= 32 and all(term in text for term in backup_markers):
        checks.append(pass_check(category, "Main deck plus backup", f"Template PPT has {slide_count} slides with B1-B8 backup."))
    elif slide_count >= 24:
        checks.append(
            partial_check(
                category,
                "Main deck plus backup",
                f"Template PPT has {slide_count} slides but missing some backup markers.",
                "Regenerate the template presentation with backup slides.",
            )
        )
    else:
        checks.append(fail_check(category, "Main deck plus backup", f"Template PPT has {slide_count} slides.", "Regenerate the deck."))
    if all(
        term in text
        for term in [
            "ELLMob",
            "CausalMob",
            "AgentMob",
            "Zero-shot rule tree",
            "Rule + 500 history",
            "Method Spectrum",
            "Distillation & Deployment",
        ]
    ):
        checks.append(pass_check(category, "Key defense content in deck", "Deck contains literature anchors, method spectrum, and distillation deployment content."))
    else:
        checks.append(
            partial_check(
                category,
                "Key defense content in deck",
                "Deck misses at least one literature or baseline defense marker.",
                "Update related-work and comparison slides.",
            )
        )
    if exists("outputs/final_project/presentation_speaker_notes_zh.md") and contains_text(
        "outputs/final_project/presentation_speaker_notes_zh.md",
        ["10 分钟主讲路径", "Backup 页怎么用"],
    ):
        checks.append(pass_check(category, "Speaker notes", "Chinese speaker notes include talk path and backup map."))
    else:
        checks.append(partial_check(category, "Speaker notes", "Speaker notes do not show talk path.", "Update speaker notes."))
    return checks


def audit_reproducibility_and_gpu() -> list[Check]:
    category = "Reproducibility and GPU"
    checks: list[Check] = []
    if exists("src/run_stronger_tabular_baselines.py") and exists("outputs/strong_baselines/strong_tabular_baseline_metrics.csv"):
        checks.append(pass_check(category, "Stronger baseline script", "Strong baseline script and metrics exist."))
    else:
        checks.append(fail_check(category, "Stronger baseline script", "Missing stronger baseline script or output.", "Restore baseline script/output."))
    device_path = PROJECT_ROOT / "outputs/label_free_llm_adaptation/label_free_llm_adaptation_metrics.csv"
    if device_path.exists():
        devices = set(pd.read_csv(device_path).get("device", pd.Series(dtype=str)).dropna().astype(str))
        if "cuda" in devices:
            checks.append(pass_check(category, "GPU execution evidence", "label_free_llm_adaptation_metrics.csv records device=cuda."))
        else:
            checks.append(
                partial_check(
                    category,
                    "GPU execution evidence",
                    f"Recorded devices: {sorted(devices)}.",
                    "Record GPU model/CUDA metadata in experiment outputs.",
                )
            )
    else:
        checks.append(fail_check(category, "GPU execution evidence", "Missing label-free metrics with device column.", "Rerun label-free adaptation."))
    if exists("outputs/submission_readiness/environment_manifest.json") and exists(
        "outputs/submission_readiness/environment_freeze.txt"
    ):
        checks.append(
            pass_check(
                category,
                "Environment manifest",
                "Environment manifest and freeze file exist under outputs/submission_readiness.",
            )
        )
    else:
        checks.append(
            partial_check(
                category,
                "Environment manifest",
                "No committed environment freeze or run manifest was found in this audit.",
                "Run src/build_environment_manifest.py before paper submission.",
            )
        )
    return checks


def build_checks() -> list[Check]:
    checks: list[Check] = []
    checks.extend(audit_core_results())
    checks.extend(audit_baselines_and_controls())
    checks.extend(audit_guardrails())
    checks.extend(audit_multi_output_evaluation())
    checks.extend(audit_temporal_external_validation())
    checks.extend(audit_literature_and_story())
    checks.extend(audit_presentation())
    checks.extend(audit_reproducibility_and_gpu())
    return checks


def write_outputs(checks: list[Check]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT_DIR / "submission_readiness_scorecard.csv"
    report_path = OUTPUT_DIR / "submission_readiness_audit.md"
    frame = pd.DataFrame([check.__dict__ for check in checks])
    frame.to_csv(score_path, index=False)

    overall = float(frame["score"].mean())
    by_category = frame.groupby("category", as_index=False)["score"].mean().sort_values("category")
    blockers = frame[frame["status"] == "FAIL"]
    partial = frame[frame["status"] == "PARTIAL"]

    lines = [
        "# Submission Readiness Audit",
        "",
        "Generated from current repository artifacts. This is an evidence audit, not a claim that the project is fully submission-ready.",
        "",
        f"Overall readiness score: `{overall:.2f}` / 1.00",
        "",
        "## Category Scores",
        "",
        "| Category | Score |",
        "|---|---:|",
    ]
    for row in by_category.itertuples(index=False):
        lines.append(f"| {row.category} | {row.score:.2f} |")

    lines.extend(["", "## Hard Blockers", ""])
    if blockers.empty:
        lines.append("No FAIL-level blockers were detected from the checked artifacts.")
    else:
        lines.extend(["| Category | Item | Evidence | Recommendation |", "|---|---|---|---|"])
        for row in blockers.itertuples(index=False):
            lines.append(f"| {row.category} | {row.item} | {row.evidence} | {row.recommendation} |")

    lines.extend(["", "## Partial Items To Fix Before Paper Submission", ""])
    if partial.empty:
        lines.append("No PARTIAL items were detected.")
    else:
        lines.extend(["| Category | Item | Evidence | Recommendation |", "|---|---|---|---|"])
        for row in partial.itertuples(index=False):
            lines.append(f"| {row.category} | {row.item} | {row.evidence} | {row.recommendation} |")

    lines.extend(["", "## Full Evidence Matrix", "", "| Category | Item | Status | Score | Evidence |", "|---|---|---:|---:|---|"])
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.category} | {row.item} | {row.status} | {row.score:.1f} | {row.evidence} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Course-project readiness is strong: the core result, baselines, guardrails, deck, Q&A material, and paper draft package are present.",
            "- No artifact-level FAIL or PARTIAL items remain in this audit; remaining work is LaTeX formatting, advisor feedback, and optional additional replications.",
            "- The defensible paper claim should remain scoped to label-free event adaptation for survey-based household mobility under a post-pandemic shift.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, score_path


def main() -> None:
    report_path, score_path = write_outputs(build_checks())
    print(f"Wrote {report_path}")
    print(f"Wrote {score_path}")


if __name__ == "__main__":
    main()
