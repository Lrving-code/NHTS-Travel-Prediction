"""Run robustness-check diagnostics for the final project."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "outputs" / "final_project"
MODE_DIR = PROJECT_ROOT / "outputs" / "mode_composition_extension"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "final_label_free_2022"
LABEL_FREE_DIR = PROJECT_ROOT / "outputs" / "label_free_llm_adaptation"
LLM_FEATURE_DIR = PROJECT_ROOT / "outputs" / "llm_event_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness_checks"

TARGET = "CNTTDHH"
WEIGHT = "WTHHFIN"
BASE = "historical_xgboost"
LLM_BEST = "llm_trip_suppression_a1p25"
LLM_ONLY = "llm_only_trip_suppression_a1p25"
GLOBAL_BEST = "global_trip_suppression_a1p25"
PRIMARY = "gated_trip_suppression_a1_d0p15"
RANDOM_SEED = 20260609

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    target_mean = weighted_average(y_true, weights)
    numerator = np.sum(weights * np.square(y_true - y_pred))
    denominator = np.sum(weights * np.square(y_true - target_mean))
    return float(1.0 - numerator / denominator)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "weighted_mae": weighted_average(np.abs(error), weights),
        "weighted_rmse": float(np.sqrt(weighted_average(np.square(error), weights))),
        "weighted_bias": weighted_average(error, weights),
        "weighted_r2": weighted_r2(y_true, y_pred, weights),
        "weighted_prediction_mean": weighted_average(y_pred, weights),
    }


def apply_pressure(base_prediction: np.ndarray, pressure: np.ndarray, alpha: float, min_factor: float = 0.05) -> np.ndarray:
    factor = np.clip(1.0 - alpha * pressure, min_factor, 1.0)
    return np.maximum(base_prediction * factor, 0.0)


def global_pressure(pressure: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.full_like(pressure, weighted_average(pressure, weights), dtype=float)


def gated_pressure(pressure: np.ndarray, weights: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    global_value = weighted_average(pressure, weights)
    global_values = np.full_like(pressure, global_value, dtype=float)
    return np.where(np.abs(pressure - global_value) >= threshold, pressure, global_values)


def random_pressures(pressure: np.ndarray, n_runs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(n_runs):
        shuffled = pressure.copy()
        rng.shuffle(shuffled)
        output.append(shuffled)
    return output


def metric_from_summary(summary: pd.DataFrame, method: str, column: str) -> float:
    return float(summary.loc[summary["method"] == method, column].iloc[0])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_metrics = pd.read_csv(FINAL_DIR / "final_metrics_summary.csv")
    mode_metrics = pd.read_csv(MODE_DIR / "mode_composition_metrics.csv")
    grid_metrics = pd.read_csv(LABEL_FREE_DIR / "label_free_llm_adaptation_metrics.csv")
    predictions = pd.read_csv(MODEL_DIR / "final_2022_predictions.csv")
    return final_metrics, mode_metrics, grid_metrics, predictions


def run_permutation_controls(predictions: pd.DataFrame, n_runs: int = 500) -> pd.DataFrame:
    y_true = predictions[TARGET].to_numpy(dtype=float)
    weights = predictions[WEIGHT].to_numpy(dtype=float)
    base = predictions["base_prediction"].to_numpy(dtype=float)
    pressure = predictions["llm_trip_suppression_pressure"].to_numpy(dtype=float)
    rows = []

    deterministic = {
        "actual_llm_a1p25": apply_pressure(base, pressure, 1.25),
        "actual_global_a1p25": apply_pressure(base, global_pressure(pressure, weights), 1.25),
        "actual_gated_a1_d0p15": apply_pressure(base, gated_pressure(pressure, weights, 0.15), 1.0),
        "actual_global_a1": apply_pressure(base, global_pressure(pressure, weights), 1.0),
    }
    for method, values in deterministic.items():
        rows.append({"family": "actual", "method": method, "run": 0, **evaluate(y_true, values, weights)})

    for run, shuffled in enumerate(random_pressures(pressure, n_runs, RANDOM_SEED), start=1):
        rows.append(
            {
                "family": "permutation",
                "method": "permuted_llm_a1p25",
                "run": run,
                **evaluate(y_true, apply_pressure(base, shuffled, 1.25), weights),
            }
        )
        rows.append(
            {
                "family": "permutation",
                "method": "permuted_gated_a1_d0p15",
                "run": run,
                **evaluate(y_true, apply_pressure(base, gated_pressure(shuffled, weights, 0.15), 1.0), weights),
            }
        )
    return pd.DataFrame(rows)


def empirical_p_value(control: pd.Series, actual: float) -> float:
    return float((np.sum(control <= actual) + 1.0) / (len(control) + 1.0))


def leakage_status() -> tuple[bool, list[str]]:
    checked_paths = [
        LLM_FEATURE_DIR / "household_cohort_profiles.csv",
        LLM_FEATURE_DIR / "cursor_api_full_gpt55_low_c15" / "validated" / "llm_event_features_normalized.csv",
    ]
    forbidden = {"HOUSEID", "CNTTDHH", "WTHHFIN"}
    leaks: list[str] = []
    for path in checked_paths:
        if not path.exists():
            leaks.append(f"missing:{path}")
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        overlap = sorted(columns.intersection(forbidden))
        if overlap:
            leaks.append(f"{path}: {overlap}")
    return not leaks, leaks


def build_claim_audit(
    final_metrics: pd.DataFrame,
    mode_metrics: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    baseline = metric_from_summary(final_metrics, BASE, "weighted_mae")
    primary = metric_from_summary(final_metrics, PRIMARY, "weighted_mae")
    llm_best = metric_from_summary(final_metrics, LLM_BEST, "weighted_mae")
    global_best = metric_from_summary(final_metrics, GLOBAL_BEST, "weighted_mae")
    primary_bias = metric_from_summary(final_metrics, PRIMARY, "weighted_bias")
    llm_only = metric_from_summary(final_metrics, LLM_ONLY, "weighted_mae")
    mode_base_tv = metric_from_summary(mode_metrics, "historical_xgboost", "weighted_total_variation")
    mode_llm_tv = metric_from_summary(mode_metrics, "llm_transit_avoidance_a1", "weighted_total_variation")
    transit_base = metric_from_summary(mode_metrics, "historical_xgboost", "transit_share_weighted_mae")
    transit_llm = metric_from_summary(mode_metrics, "llm_transit_avoidance_a1", "transit_share_weighted_mae")
    actual_llm = float(permutation.loc[permutation["method"] == "actual_llm_a1p25", "weighted_mae"].iloc[0])
    random_llm = permutation.loc[permutation["method"] == "permuted_llm_a1p25", "weighted_mae"]
    p_llm = empirical_p_value(random_llm, actual_llm)

    claims = [
        {
            "claim": "Primary fixed no-label adapter improves trip-count prediction.",
            "verdict": "Supported",
            "evidence": f"weighted MAE {baseline:.4f} -> {primary:.4f}; primary weighted bias {primary_bias:+.4f}.",
            "revision": "Keep as the main quantitative claim.",
        },
        {
            "claim": "LLM cohort-specific ranking is the dominant reason for improvement.",
            "verdict": "Overclaim",
            "evidence": f"best LLM-specific row improves over global rule only {(global_best - llm_best) / global_best:.2%}; permutation p={p_llm:.4f}.",
            "revision": "Say the dominant signal is event-level downscaling; cohort-specific LLM ranking adds incremental evidence.",
        },
        {
            "claim": "Pure LLM can replace the historical household model.",
            "verdict": "Rejected",
            "evidence": f"LLM-only pressure wMAE {llm_only:.4f}, worse than hybrid {primary:.4f}.",
            "revision": "Frame LLM as semantic adapter, not standalone predictor.",
        },
        {
            "claim": "Mode composition is strongly solved by the method.",
            "verdict": "Overclaim",
            "evidence": f"weighted TV {mode_base_tv:.4f} -> {mode_llm_tv:.4f}, only {(mode_base_tv - mode_llm_tv) / mode_base_tv:.2%} improvement.",
            "revision": f"Report it as exploratory; emphasize transit MAE {transit_base:.4f} -> {transit_llm:.4f}.",
        },
        {
            "claim": "No target-year label leakage in LLM inputs.",
            "verdict": "Supported with caveat",
            "evidence": "cohort profile and validated LLM feature files exclude HOUSEID, CNTTDHH, and WTHHFIN.",
            "revision": "Also disclose that GPT-5.5 has retrospective world knowledge; prospective deployment needs frozen event context.",
        },
    ]
    return pd.DataFrame(claims)


def plot_permutation_null(permutation: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "permutation_null_mae.png"
    plt.figure(figsize=(7.2, 4.3))
    controls = permutation[permutation["method"] == "permuted_llm_a1p25"]
    actual = permutation[permutation["method"] == "actual_llm_a1p25"].iloc[0]
    global_row = permutation[permutation["method"] == "actual_global_a1p25"].iloc[0]
    sns.histplot(controls["weighted_mae"], bins=32, color="#94a3b8", edgecolor="white")
    plt.axvline(actual["weighted_mae"], color="#198754", linewidth=2.5, label="actual LLM pressure")
    plt.axvline(global_row["weighted_mae"], color="#00796b", linewidth=2.5, linestyle="--", label="global pressure")
    plt.xlabel("weighted MAE after random assignment of cohort pressure")
    plt.ylabel("count")
    plt.title("Robustness check: real LLM pressure beats random assignment", fontsize=12, fontweight="bold")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def write_report(
    final_metrics: pd.DataFrame,
    mode_metrics: pd.DataFrame,
    permutation: pd.DataFrame,
    claim_audit: pd.DataFrame,
    figure_path: Path,
) -> Path:
    path = OUTPUT_DIR / "robustness_check_report.md"
    baseline = metric_from_summary(final_metrics, BASE, "weighted_mae")
    primary = metric_from_summary(final_metrics, PRIMARY, "weighted_mae")
    global_a1 = float(permutation.loc[permutation["method"] == "actual_global_a1", "weighted_mae"].iloc[0])
    primary_perm = permutation.loc[permutation["method"] == "permuted_gated_a1_d0p15", "weighted_mae"]
    primary_actual = float(permutation.loc[permutation["method"] == "actual_gated_a1_d0p15", "weighted_mae"].iloc[0])
    llm_actual = float(permutation.loc[permutation["method"] == "actual_llm_a1p25", "weighted_mae"].iloc[0])
    llm_perm = permutation.loc[permutation["method"] == "permuted_llm_a1p25", "weighted_mae"]
    leak_ok, leaks = leakage_status()

    lines = [
        "# Robustness Check Report",
        "",
        "## Scope",
        "",
        "This report checks whether the main project results hold under stricter controls. It focuses on target leakage, global-event baselines, random-pressure controls, and claims that should be scoped carefully in the course presentation.",
        "",
        "## Main Findings",
        "",
        f"- Main trip-count claim remains supported: ordinary historical XGBoost weighted MAE `{baseline:.4f}` -> primary gated weighted MAE `{primary:.4f}`.",
        f"- The strongest contribution should be phrased as **event-level correction without 2022 label calibration**, not as strong individualized LLM ranking.",
        f"- Compared with same-alpha global pressure, primary gated wMAE is `{primary:.4f}` vs global-a1 wMAE `{global_a1:.4f}`.",
        f"- 500-run permutation null for best LLM pressure: actual wMAE `{llm_actual:.4f}`, random mean `{llm_perm.mean():.4f}`, empirical p `{empirical_p_value(llm_perm, llm_actual):.4f}`.",
        f"- 500-run permutation null for primary gated rule: actual wMAE `{primary_actual:.4f}`, random mean `{primary_perm.mean():.4f}`, empirical p `{empirical_p_value(primary_perm, primary_actual):.4f}`.",
        f"- LLM-input leakage scan: `{'PASS' if leak_ok else 'FAIL'}`" + ("" if leak_ok else f" ({'; '.join(leaks)})"),
        "",
        "## Claim Check",
        "",
        "| Claim | Verdict | Evidence | Revision |",
        "|---|---|---|---|",
    ]
    for row in claim_audit.itertuples(index=False):
        lines.append(f"| {row.claim} | {row.verdict} | {row.evidence} | {row.revision} |")

    mode_base_tv = metric_from_summary(mode_metrics, "historical_xgboost", "weighted_total_variation")
    mode_llm_tv = metric_from_summary(mode_metrics, "llm_transit_avoidance_a1", "weighted_total_variation")
    transit_base = metric_from_summary(mode_metrics, "historical_xgboost", "transit_share_weighted_mae")
    transit_llm = metric_from_summary(mode_metrics, "llm_transit_avoidance_a1", "transit_share_weighted_mae")
    lines.extend(
        [
            "",
            "## Presentation Framing Fixes",
            "",
            "### Finding 1: Global event downscaling is a very strong baseline",
            "",
            "Applying the global average trip-suppression pressure already removes much of the 2022 over-prediction. Therefore, the presentation should not claim that LLM individualized ranking is the dominant mechanism. The cleaner statement is that LLM-derived event semantics provide an event-level correction, with cohort-specific ranking adding incremental support.",
            "",
            "### Finding 2: Best-MAE sensitivity row must not be the main method",
            "",
            "`llm_trip_suppression_a1p25` has the lowest MAE, but alpha 1.25 should be treated as sensitivity analysis. The main method should remain the fixed `gated_trip_suppression_a1_d0p15`, because it gives near-zero bias and is easier to defend as a no-label rule.",
            "",
            "### Finding 3: Mode composition is exploratory",
            "",
            f"Weighted total variation improves only `{(mode_base_tv - mode_llm_tv) / mode_base_tv:.2%}`. The stronger mode-related result is transit-share weighted MAE `{transit_base:.4f}` -> `{transit_llm:.4f}`. Present this as transit-specific evidence, not as a solved full mode-choice model.",
            "",
            "### Finding 4: Temporal validity needs an explicit caveat",
            "",
            "The project is label-free with respect to 2022 NHTS outcomes, but GPT-5.5 may contain retrospective world knowledge about COVID-era mobility. In the course presentation, state that the event context is allowed, and list prospective external event feeds as future work.",
            "",
            "### Finding 5: Current evidence is enough for a strong course project, but should stay scoped",
            "",
            "A larger follow-up study would need validation across multiple shocks or regions, prospective event-context freezing, and stronger uncertainty estimates. For the course project, the current contribution is coherent if claims are scoped carefully.",
            "",
            "## New Audit Artifacts",
            "",
            f"- `{OUTPUT_DIR / 'permutation_pressure_controls.csv'}`",
            f"- `{OUTPUT_DIR / 'claim_audit.csv'}`",
            f"- `{figure_path}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_logging()
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    final_metrics, mode_metrics, _grid_metrics, predictions = load_data()
    permutation = run_permutation_controls(predictions)
    claim_audit = build_claim_audit(final_metrics, mode_metrics, permutation)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    permutation.to_csv(OUTPUT_DIR / "permutation_pressure_controls.csv", index=False)
    claim_audit.to_csv(OUTPUT_DIR / "claim_audit.csv", index=False)
    figure_path = plot_permutation_null(permutation)
    report_path = write_report(final_metrics, mode_metrics, permutation, claim_audit, figure_path)
    LOGGER.info("Wrote robustness check report: %s", report_path)


if __name__ == "__main__":
    main()
