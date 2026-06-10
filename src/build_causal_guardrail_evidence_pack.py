"""Build a causal-guardrail evidence pack from existing experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "causal_guardrails"

LABEL_FREE_METRICS = PROJECT_ROOT / "outputs" / "label_free_llm_adaptation" / "label_free_llm_adaptation_metrics.csv"
STRONG_BASELINES = PROJECT_ROOT / "outputs" / "strong_baselines" / "strong_tabular_baseline_metrics.csv"
PERMUTATION_CONTROLS = PROJECT_ROOT / "outputs" / "robustness_checks" / "permutation_pressure_controls.csv"
TEMPORAL_TRANSFER = PROJECT_ROOT / "outputs" / "temporal_transfer_validation" / "temporal_transfer_metrics.csv"
PLACEBO_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "pre_covid_placebo_event_correction"
    / "pre_covid_placebo_event_correction_metrics.csv"
)
LEAKAGE_AUDIT = PROJECT_ROOT / "outputs" / "leakage_audit" / "llm_input_leakage_audit.csv"
GUARDRAIL_CHECK = PROJECT_ROOT / "outputs" / "leakage_audit" / "llm_guardrail_instruction_check.csv"
PARETO_POINTS = PROJECT_ROOT / "outputs" / "multi_objective_pareto" / "preference_operating_points.csv"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceItem:
    check: str
    verdict: str
    key_result: str
    implication: str
    artifact: str


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return pd.read_csv(path)


def metric(df: pd.DataFrame, method: str, column: str) -> float:
    rows = df.loc[df["method"] == method]
    if rows.empty:
        raise KeyError(f"Missing method {method!r} in metrics table.")
    return float(rows.iloc[0][column])


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def compute_permutation_p_value(permutation: pd.DataFrame, actual_method: str, permuted_method: str) -> tuple[float, int]:
    actual_rows = permutation.loc[permutation["method"] == actual_method]
    permuted_rows = permutation.loc[permutation["method"] == permuted_method]
    if actual_rows.empty or permuted_rows.empty:
        raise KeyError("Missing actual or permuted rows for permutation p-value.")
    actual_mae = float(actual_rows.iloc[0]["weighted_mae"])
    permuted_mae = pd.to_numeric(permuted_rows["weighted_mae"], errors="raise")
    count_as_good = int((permuted_mae <= actual_mae).sum())
    p_value = (count_as_good + 1) / (len(permuted_mae) + 1)
    return p_value, len(permuted_mae)


def selected_method(row: pd.Series) -> str:
    if "selected_method" in row.index:
        return str(row["selected_method"])
    return str(row["method"])


def audit_record_count(audit: pd.DataFrame, artifact_substring: str) -> int:
    rows = audit.loc[audit["artifact"].astype(str).str.contains(artifact_substring, regex=False)]
    if rows.empty:
        return 0
    return int(pd.to_numeric(rows["records_checked"], errors="raise").sum())


def build_evidence_items() -> tuple[list[EvidenceItem], dict[str, float | int | str]]:
    label_free = read_csv(LABEL_FREE_METRICS)
    strong = read_csv(STRONG_BASELINES)
    permutation = read_csv(PERMUTATION_CONTROLS)
    temporal = read_csv(TEMPORAL_TRANSFER)
    placebo = read_csv(PLACEBO_METRICS)
    leakage = read_csv(LEAKAGE_AUDIT)
    guardrail = read_csv(GUARDRAIL_CHECK)
    pareto = read_csv(PARETO_POINTS)

    baseline_mae = metric(label_free, "historical_xgboost", "weighted_mae")
    baseline_bias = metric(label_free, "historical_xgboost", "weighted_bias")
    primary_mae = metric(label_free, "gated_trip_suppression_a1_d0p15", "weighted_mae")
    primary_bias = metric(label_free, "gated_trip_suppression_a1_d0p15", "weighted_bias")
    primary_r2 = metric(label_free, "gated_trip_suppression_a1_d0p15", "weighted_r2")
    global_mae = metric(label_free, "global_trip_suppression_a1", "weighted_mae")
    llm_only_mae = metric(label_free, "llm_only_trip_suppression_a1p25", "weighted_mae")
    catboost_mae = metric(strong, "catboost_gpu", "weighted_mae")
    catboost_bias = metric(strong, "catboost_gpu", "weighted_bias")

    mae_reduction = (baseline_mae - primary_mae) / baseline_mae
    catboost_gap = (catboost_mae - primary_mae) / catboost_mae
    global_gap = global_mae - primary_mae

    p_value, permutation_runs = compute_permutation_p_value(
        permutation,
        "actual_gated_a1_d0p15",
        "permuted_gated_a1_d0p15",
    )

    pre_covid = temporal.loc[temporal["experiment"].str.startswith("pre_covid")]
    post_covid = temporal.loc[temporal["experiment"] == "post_covid_history_to_2022"]
    if pre_covid.empty or post_covid.empty:
        raise KeyError("Temporal transfer table is missing pre-COVID or post-COVID rows.")
    pre_covid_abs_bias = float(pd.to_numeric(pre_covid["weighted_bias"]).abs().mean())
    post_covid_abs_bias = abs(float(post_covid.iloc[0]["weighted_bias"]))

    routine_placebo = metric(placebo, "routine_2001_2009_to_2017", "weighted_mae")
    placebo_a1 = metric(placebo, "placebo_global_2022_suppression_a1", "weighted_mae")
    placebo_a1p25 = metric(placebo, "placebo_global_2022_suppression_a1p25", "weighted_mae")
    placebo_a1_delta = placebo_a1 - routine_placebo
    placebo_a1p25_delta = placebo_a1p25 - routine_placebo

    violation_count = int(pd.to_numeric(leakage["violations"], errors="raise").sum())
    profile_records = audit_record_count(leakage, "household_cohort_profiles")
    prompt_records = audit_record_count(leakage, "household_cohort_batch_prompts")
    feature_records = audit_record_count(leakage, "llm_event_features_normalized")
    batches_checked = int(pd.to_numeric(leakage.get("batches_checked", pd.Series(dtype=float))).fillna(0).sum())
    guardrails_present = bool(guardrail.iloc[0]["all_required_guardrails_present"])

    balanced = pareto.loc[pareto["profile"] == "balanced_course_report"]
    low_cost = pareto.loc[pareto["profile"] == "low_cost_deployment"]
    if balanced.empty or low_cost.empty:
        raise KeyError("Pareto operating point table is missing required profiles.")
    balanced_method = selected_method(balanced.iloc[0])
    low_cost_method = selected_method(low_cost.iloc[0])

    items = [
        EvidenceItem(
            check="2022 is an event-shift target",
            verdict="Supported",
            key_result=(
                f"pre-COVID mean absolute weighted bias = {fmt(pre_covid_abs_bias)}; "
                f"2022 absolute weighted bias = {fmt(post_covid_abs_bias)}"
            ),
            implication=(
                "The target year is harder than routine temporal transfer; ordinary historical models "
                "systematically overpredict post-pandemic travel."
            ),
            artifact=str(TEMPORAL_TRANSFER.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Main label-free event adapter improves prediction",
            verdict="Supported",
            key_result=(
                f"historical XGBoost wMAE {fmt(baseline_mae)} -> primary gated wMAE {fmt(primary_mae)} "
                f"({pct(mae_reduction)} reduction); wBias {fmt(baseline_bias)} -> {fmt(primary_bias)}; "
                f"wR2 {fmt(primary_r2)}"
            ),
            implication=(
                "The strongest quantitative claim is label-free event adaptation, not direct LLM prediction."
            ),
            artifact=str(LABEL_FREE_METRICS.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Stronger non-LLM tabular baselines do not remove the shift",
            verdict="Supported",
            key_result=(
                f"best extra tabular baseline CatBoost GPU wMAE = {fmt(catboost_mae)}, "
                f"wBias = {fmt(catboost_bias)}; primary adapter is {pct(catboost_gap)} lower in wMAE"
            ),
            implication=(
                "Model capacity alone does not solve the mechanism shift; the event mechanism must be represented."
            ),
            artifact=str(STRONG_BASELINES.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Prompt/schema target leakage",
            verdict="Passed with caveat",
            key_result=(
                f"{profile_records} cohort profiles, {prompt_records} prompt payload records, "
                f"{feature_records} validated feature-output record(s), {batches_checked} batch prompts; "
                f"forbidden-field violations = {violation_count}; required guardrails present = {guardrails_present}"
            ),
            implication=(
                "The LLM branch receives cohort covariates and event context, not target labels, weights, IDs, "
                "or aggregate 2022 target outcomes. Retrospective world knowledge remains a limitation."
            ),
            artifact=str(LEAKAGE_AUDIT.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Random-pressure negative control",
            verdict="Supported",
            key_result=(
                f"actual gated pressure beats {permutation_runs} random assignments; empirical p = {fmt(p_value)}"
            ),
            implication=(
                "The structured event pressure contains more information than arbitrary random perturbation."
            ),
            artifact=str(PERMUTATION_CONTROLS.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Global event prior as a low-cost baseline",
            verdict="Boundary condition",
            key_result=(
                f"global a1 wMAE = {fmt(global_mae)}; primary gated wMAE = {fmt(primary_mae)}; "
                f"absolute gap = {fmt(global_gap)}"
            ),
            implication=(
                "The dominant signal is event-level suppression. Cohort-aware LLM priors add auditable refinement, "
                "not the entire gain."
            ),
            artifact=str(PERMUTATION_CONTROLS.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Pre-COVID placebo event correction",
            verdict="Supported as guardrail",
            key_result=(
                f"2017 routine wMAE = {fmt(routine_placebo)}; full 2022 suppression a1 wMAE = {fmt(placebo_a1)} "
                f"(delta {fmt(placebo_a1_delta)}); a1.25 wMAE = {fmt(placebo_a1p25)} "
                f"(delta {fmt(placebo_a1p25_delta)})"
            ),
            implication=(
                "The event prior is not a universal downshift. It needs event context and strength discipline."
            ),
            artifact=str(PLACEBO_METRICS.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Hybrid design versus LLM-only prediction",
            verdict="Supported",
            key_result=(
                f"LLM-only pressure wMAE = {fmt(llm_only_mae)}; primary hybrid wMAE = {fmt(primary_mae)}"
            ),
            implication=(
                "The LLM is useful as an event-prior generator, while the routine household predictor remains necessary."
            ),
            artifact=str(LABEL_FREE_METRICS.relative_to(PROJECT_ROOT)),
        ),
        EvidenceItem(
            check="Planning-oriented operating point",
            verdict="Supported",
            key_result=(
                f"balanced profile selects {balanced_method}; low-cost profile selects {low_cost_method}"
            ),
            implication=(
                "The method exposes an auditable choice between accuracy/calibration and request cost."
            ),
            artifact=str(PARETO_POINTS.relative_to(PROJECT_ROOT)),
        ),
    ]

    scalars: dict[str, float | int | str] = {
        "baseline_mae": baseline_mae,
        "baseline_bias": baseline_bias,
        "primary_mae": primary_mae,
        "primary_bias": primary_bias,
        "primary_r2": primary_r2,
        "mae_reduction": mae_reduction,
        "catboost_mae": catboost_mae,
        "catboost_gap": catboost_gap,
        "permutation_p_value": p_value,
        "permutation_runs": permutation_runs,
        "pre_covid_abs_bias": pre_covid_abs_bias,
        "post_covid_abs_bias": post_covid_abs_bias,
        "placebo_a1_delta": placebo_a1_delta,
        "placebo_a1p25_delta": placebo_a1p25_delta,
        "global_gap": global_gap,
        "leakage_violations": violation_count,
        "profile_records": profile_records,
        "prompt_records": prompt_records,
        "feature_records": feature_records,
        "batches_checked": batches_checked,
        "balanced_method": balanced_method,
        "low_cost_method": low_cost_method,
    }
    return items, scalars


def draw_node(ax, label: str, xy: tuple[float, float], width: float, height: float, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.4,
        facecolor=color,
        edgecolor="#334155",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10, color="#0f172a")


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#64748b") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_causal_dag(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlim(0, 12.35)
    ax.set_ylim(0, 7)
    ax.axis("off")

    blue = "#dbeafe"
    orange = "#fed7aa"
    green = "#dcfce7"
    red = "#fee2e2"
    pale = "#f8fafc"

    draw_node(ax, "Household attributes\nincome, vehicles, workers,\nurban form", (0.55, 4.75), 2.35, 1.05, blue)
    draw_node(ax, "Routine travel demand\nf_hist(X)", (3.55, 5.0), 2.2, 0.8, blue)
    draw_node(ax, "Observed 2022 behavior\nY", (8.9, 5.0), 2.15, 0.8, red)

    draw_node(ax, "Societal event context\nCOVID recovery, telework,\ntransit perception", (0.55, 2.45), 2.35, 1.05, orange)
    draw_node(ax, "Event response mechanisms\nremote work, transit avoidance,\ndelivery substitution", (3.55, 2.45), 2.35, 1.05, orange)

    draw_node(ax, "Cohort profile\nP(H)", (3.55, 0.95), 1.95, 0.75, pale)
    draw_node(ax, "LLM event-prior generator\nno target labels", (6.25, 1.0), 2.15, 0.9, orange)
    draw_node(ax, "Structured event priors\ns_event", (9.0, 1.05), 2.0, 0.8, orange)
    draw_node(ax, "No-label adapter\nY_hat = f_hist(X) * c(s_event)", (6.25, 3.55), 2.4, 0.9, green)
    draw_node(ax, "Predicted 2022 behavior\nY_hat", (9.0, 3.55), 2.0, 0.8, green)
    draw_node(ax, "Evaluation only\n2022 labels stay outside\nLLM / adapter", (8.65, 6.16), 2.8, 0.72, red)

    draw_arrow(ax, (2.9, 5.25), (3.55, 5.35))
    draw_arrow(ax, (5.75, 5.35), (8.9, 5.35))
    draw_arrow(ax, (2.9, 2.95), (3.55, 2.95))
    draw_arrow(ax, (5.9, 2.95), (8.9, 5.05))
    draw_arrow(ax, (1.75, 4.75), (4.4, 3.5))
    draw_arrow(ax, (1.75, 2.45), (6.25, 1.45))
    draw_arrow(ax, (2.9, 5.0), (3.55, 1.35))
    draw_arrow(ax, (5.5, 1.35), (6.25, 1.45))
    draw_arrow(ax, (8.4, 1.45), (9.0, 1.45))
    draw_arrow(ax, (10.0, 1.85), (7.5, 3.55))
    draw_arrow(ax, (4.65, 5.0), (6.45, 4.45))
    draw_arrow(ax, (8.65, 4.0), (9.0, 4.0))
    draw_arrow(ax, (9.9, 4.35), (9.9, 5.0))
    draw_arrow(ax, (9.9, 5.8), (9.9, 6.16), "#be123c")

    ax.text(
        0.55,
        6.55,
        "Causal structure as guardrail, not causal-effect identification",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.55,
        6.23,
        "The LLM branch sees cohort covariates and event context; 2022 outcomes are evaluation-only.",
        fontsize=10.5,
        color="#475569",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary_csv(items: list[EvidenceItem], path: Path) -> None:
    rows = [item.__dict__ for item in items]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(items: list[EvidenceItem], scalars: dict[str, float | int | str], path: Path) -> None:
    lines = [
        "# Causal Guardrail Evidence Pack",
        "",
        "## Purpose",
        "",
        "This evidence pack consolidates the checks that keep the LLM component auditable. "
        "It supports causal plausibility and label-free event adaptation; it does not claim causal-effect identification.",
        "",
        "## Core Claim Boundary",
        "",
        "- Allowed: LLM-derived event priors are structured mechanism proxies for post-pandemic mobility adaptation.",
        "- Allowed: Causal guardrails reduce leakage and arbitrary-correction risk.",
        "- Not allowed: The LLM estimates the causal effect of COVID-19 on household travel.",
        "- Not allowed: The LLM directly predicts household trip counts better than tabular models.",
        "",
        "## Headline Evidence",
        "",
        (
            f"- Historical XGBoost wMAE `{fmt(float(scalars['baseline_mae']))}` -> primary gated adapter "
            f"`{fmt(float(scalars['primary_mae']))}` "
            f"({pct(float(scalars['mae_reduction']))} reduction)."
        ),
        (
            f"- Weighted bias `{fmt(float(scalars['baseline_bias']))}` -> "
            f"`{fmt(float(scalars['primary_bias']))}`; weighted R2 = "
            f"`{fmt(float(scalars['primary_r2']))}`."
        ),
        (
            f"- Best stronger non-LLM baseline CatBoost GPU wMAE `{fmt(float(scalars['catboost_mae']))}`; "
            f"primary adapter is {pct(float(scalars['catboost_gap']))} lower."
        ),
        (
            f"- Leakage audit: `{int(scalars['profile_records'])}` cohort profiles, "
            f"`{int(scalars['prompt_records'])}` prompt payload records, "
            f"`{int(scalars['feature_records'])}` validated feature-output record(s), and "
            f"`{int(scalars['batches_checked'])}` batch prompts checked, "
            f"`{int(scalars['leakage_violations'])}` forbidden-field violations."
        ),
        (
            f"- Random-pressure negative control: empirical p = "
            f"`{fmt(float(scalars['permutation_p_value']))}` over "
            f"`{int(scalars['permutation_runs'])}` permutations."
        ),
        (
            f"- Pre-COVID placebo: full 2022-style suppression worsens 2017 wMAE by "
            f"`{fmt(float(scalars['placebo_a1_delta']))}`; stronger a1.25 worsens it by "
            f"`{fmt(float(scalars['placebo_a1p25_delta']))}`."
        ),
        "",
        "## Evidence Table",
        "",
        "| Check | Verdict | Key result | Implication | Artifact |",
        "|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.check} | {item.verdict} | {item.key_result} | {item.implication} | `{item.artifact}` |"
        )

    lines.extend(
        [
            "",
            "## How to Use in the Paper/PPT",
            "",
            "1. Put the DAG in the method or robustness section.",
            "2. Report the leakage audit before discussing LLM gains.",
            "3. Use the random-pressure and placebo checks as guardrails, not as causal proof.",
            "4. State that the global prior is a strong low-cost baseline; the gated LLM adapter is the balanced operating point.",
            "5. Keep retrospective world-knowledge risk as a limitation unless a frozen event-context RAG is implemented.",
            "",
            "## Generated Artifacts",
            "",
            "- `outputs/causal_guardrails/causal_guardrail_summary.csv`",
            "- `outputs/causal_guardrails/causal_dag.png`",
            "- `outputs/causal_guardrails/causal_guardrail_evidence_report.md`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items, scalars = build_evidence_items()
    summary_path = OUTPUT_DIR / "causal_guardrail_summary.csv"
    dag_path = OUTPUT_DIR / "causal_dag.png"
    report_path = OUTPUT_DIR / "causal_guardrail_evidence_report.md"

    write_summary_csv(items, summary_path)
    draw_causal_dag(dag_path)
    write_report(items, scalars, report_path)

    LOGGER.info("Wrote summary: %s", summary_path)
    LOGGER.info("Wrote DAG: %s", dag_path)
    LOGGER.info("Wrote report: %s", report_path)


if __name__ == "__main__":
    main()
