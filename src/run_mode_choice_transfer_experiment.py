"""Run a harmonized trip-level mode-choice transfer experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mode_choice_branch.common import (  # noqa: E402
    MODE_GROUP_COLUMN,
    MODE_GROUP_LABELS,
    MODE_GROUP_ORDER,
    SAMPLE_WEIGHT_COLUMN,
)
from mode_choice_branch.datasets import build_mode_choice_datasets  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "mode_choice_branch" / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mode_choice_transfer"


@dataclass(frozen=True)
class ModeChoiceData:
    """Loaded mode-choice transfer splits."""

    x_train: pd.DataFrame
    y_train: np.ndarray
    w_train: np.ndarray
    x_validation: pd.DataFrame
    y_validation: np.ndarray
    w_validation: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    w_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda", help="XGBoost device, usually cuda or cpu.")
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rebuild-datasets", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def ensure_datasets(dataset_dir: Path, rebuild: bool) -> None:
    required = [dataset_dir / name for name in ("train.csv", "validation.csv", "test.csv")]
    if rebuild or any(not path.exists() for path in required):
        LOGGER.info("Building harmonized mode-choice datasets at %s", dataset_dir)
        build_mode_choice_datasets(output_dir=dataset_dir)


def load_split(dataset_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = dataset_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/run_mode_choice_branch.py build-datasets first.")
    frame = pd.read_csv(path, low_memory=False)
    if MODE_GROUP_COLUMN not in frame.columns:
        raise ValueError(f"{path} does not contain harmonized target column {MODE_GROUP_COLUMN}.")
    if SAMPLE_WEIGHT_COLUMN in frame.columns:
        weights = pd.to_numeric(frame[SAMPLE_WEIGHT_COLUMN], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        weights = pd.Series(np.ones(len(frame), dtype=float))
    features = frame.drop(columns=[MODE_GROUP_COLUMN, SAMPLE_WEIGHT_COLUMN], errors="ignore")
    return features, frame[MODE_GROUP_COLUMN].astype(str).to_numpy(), weights.to_numpy(dtype=float)


def load_data(dataset_dir: Path, rebuild: bool) -> ModeChoiceData:
    ensure_datasets(dataset_dir, rebuild)
    x_train, y_train, w_train = load_split(dataset_dir, "train")
    x_validation, y_validation, w_validation = load_split(dataset_dir, "validation")
    x_test, y_test, w_test = load_split(dataset_dir, "test")
    return ModeChoiceData(x_train, y_train, w_train, x_validation, y_validation, w_validation, x_test, y_test, w_test)


def label_mapping(labels: np.ndarray) -> dict[str, int]:
    observed = set(labels)
    ordered = [label for label in MODE_GROUP_ORDER if label in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return {label: index for index, label in enumerate(ordered)}


def encode(labels: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown mode label(s): {unknown}")
    return np.array([mapping[label] for label in labels], dtype=np.int32)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    positive = np.clip(weights.astype(float), 0.0, None)
    mean_value = float(np.mean(positive)) if len(positive) else 0.0
    if mean_value <= 0.0:
        return np.ones_like(positive)
    return positive / mean_value


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    positive = np.clip(weights.astype(float), 0.0, None)
    if float(positive.sum()) <= 0.0:
        return float(np.mean(values))
    return float(np.average(values, weights=positive))


def prior_probabilities(y_train: np.ndarray, n_classes: int, weights: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_train, weights=np.clip(weights, 0.0, None), minlength=n_classes).astype(float)
    if float(counts.sum()) <= 0.0:
        counts = np.bincount(y_train, minlength=n_classes).astype(float)
    return counts / counts.sum()


def class_balanced_weights(y_train: np.ndarray, n_classes: int, base_weights: np.ndarray | None = None) -> np.ndarray:
    base = np.ones(len(y_train), dtype=float) if base_weights is None else normalize_weights(base_weights)
    counts = np.bincount(y_train, weights=base, minlength=n_classes).astype(float)
    weights = len(y_train) / (n_classes * np.maximum(counts, 1.0))
    return normalize_weights(weights[y_train] * base)


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    args: argparse.Namespace,
    sample_weight: np.ndarray | None,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=int(np.max(y_train)) + 1,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


def probability_rows(probabilities: np.ndarray, expected_classes: int) -> np.ndarray:
    if probabilities.shape[1] == expected_classes:
        return probabilities
    output = np.zeros((probabilities.shape[0], expected_classes), dtype=float)
    output[:, : probabilities.shape[1]] = probabilities
    row_sums = output.sum(axis=1, keepdims=True)
    return output / np.maximum(row_sums, 1e-12)


def evaluate_probabilities(
    method: str,
    split: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
    uses_2017_labels: bool,
    device: str,
) -> dict[str, str | int | float | bool]:
    probabilities = probability_rows(probabilities, len(labels))
    y_pred = probabilities.argmax(axis=1)
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    one_hot = np.eye(len(labels), dtype=float)[y_true]
    brier_rows = np.sum((probabilities - one_hot) ** 2, axis=1)
    top2_hit = np.array([truth in candidates for truth, candidates in zip(y_true, top2)], dtype=float)
    return {
        "method": method,
        "split": split,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_accuracy": float(accuracy_score(y_true, y_pred, sample_weight=weights)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y_true, y_pred, average="macro", sample_weight=weights, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "survey_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", sample_weight=weights, zero_division=0)
        ),
        "top2_accuracy": float(np.mean(top2_hit)),
        "weighted_top2_accuracy": weighted_average(top2_hit, weights),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(labels))))),
        "weighted_log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(labels))), sample_weight=weights)),
        "multiclass_brier": float(np.mean(brier_rows)),
        "weighted_multiclass_brier": weighted_average(brier_rows, weights),
        "uses_2017_labels": uses_2017_labels,
        "uses_2022_labels_for_training": False,
        "device": device,
    }


def per_class_metrics(
    method: str,
    split: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    y_pred = probability_rows(probabilities, len(labels)).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        sample_weight=weights,
        zero_division=0,
    )
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "method": method,
                "split": split,
                "mode_group": label,
                "label": MODE_GROUP_LABELS.get(label, label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "weighted_precision": float(weighted_precision[index]),
                "weighted_recall": float(weighted_recall[index]),
                "weighted_f1": float(weighted_f1[index]),
                "weighted_support": float(weighted_support[index]),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
    output_dir: Path,
    method: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = probability_rows(probabilities, len(labels)).argmax(axis=1)
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        sample_weight=weights,
        normalize="true",
    )
    display_labels = [MODE_GROUP_LABELS.get(label, label) for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted mode group")
    ax.set_ylabel("Observed mode group")
    ax.set_title("2022 Survey-Weighted Harmonized Mode-Choice Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "#1f2937",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_dir / f"{method}_test_confusion_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    selected_method: str,
    confusion_path: Path,
    output_dir: Path,
) -> Path:
    test_metrics = metrics.loc[metrics["split"] == "test_2022"].copy()
    selected_row = test_metrics.loc[test_metrics["method"] == selected_method].iloc[0]
    balanced_row = test_metrics.sort_values("weighted_balanced_accuracy", ascending=False).iloc[0]
    class_rows = per_class.loc[
        (per_class["method"] == selected_method) & (per_class["split"] == "test_2022")
    ].copy()
    class_rows = class_rows.sort_values("weighted_support", ascending=False)
    display_confusion_path = confusion_path.relative_to(PROJECT_ROOT).as_posix()

    lines = [
        "# Harmonized Mode-Choice Transfer Experiment",
        "",
        "## Scope",
        "",
        "This experiment evaluates trip-level mode-choice transfer after mapping year-specific NHTS `TRPTRANS` codes into comparable `MODE_GROUP` labels. It trains only on 2017 labels, selects the XGBoost operating point by 2017 validation survey-weighted macro F1, and uses 2022 labels only for final evaluation.",
        "",
        "## 2022 Test Metrics",
        "",
        "| Method | Weighted accuracy | Weighted balanced acc. | Weighted macro F1 | Survey-weighted F1 | Weighted top-2 | Weighted log loss | Unweighted accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_metrics.sort_values("weighted_macro_f1", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.weighted_accuracy:.4f} | {row.weighted_balanced_accuracy:.4f} | "
            f"{row.weighted_macro_f1:.4f} | {row.survey_weighted_f1:.4f} | "
            f"{row.weighted_top2_accuracy:.4f} | {row.weighted_log_loss:.4f} | {row.accuracy:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Selected Operating Point",
            "",
            f"The validation-selected model is `{selected_method}`. On 2022 it reaches survey-weighted accuracy `{selected_row.weighted_accuracy:.4f}`, weighted balanced accuracy `{selected_row.weighted_balanced_accuracy:.4f}`, and weighted macro F1 `{selected_row.weighted_macro_f1:.4f}`. The corresponding unweighted accuracy is `{selected_row.accuracy:.4f}`.",
            "",
            f"The highest weighted-balanced-accuracy model is `{balanced_row.method}` at `{balanced_row.weighted_balanced_accuracy:.4f}`, but it lowers weighted macro F1 and log-loss calibration. The paper-safe interpretation should therefore report the trade-off rather than treating ordinary accuracy as sufficient.",
            "",
            "The ordinary accuracy is high because private-vehicle trips dominate the target distribution. The paper-safe interpretation should therefore emphasize balanced accuracy, macro F1, and per-class behavior rather than claiming a solved full mode-choice task.",
            "",
            f"- Confusion matrix: `{display_confusion_path}`",
            "",
            "## Per-Class Test Metrics for Selected Model",
            "",
            "| Mode group | Label | Weighted precision | Weighted recall | Weighted F1 | Weighted support | Unweighted F1 | Rows |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows.itertuples(index=False):
        lines.append(
            f"| {row.mode_group} | {row.label} | {row.weighted_precision:.4f} | "
            f"{row.weighted_recall:.4f} | {row.weighted_f1:.4f} | "
            f"{row.weighted_support:.1f} | {row.f1:.4f} | {row.support} |"
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `mode_choice_transfer_metrics.csv`: validation and 2022 test aggregate metrics with unweighted and survey-weighted columns.",
            "- `mode_choice_per_class_metrics.csv`: per-class unweighted and survey-weighted precision, recall, F1, and support.",
            "- `*_test_confusion_matrix.png`: row-normalized survey-weighted 2022 confusion matrix for the validation-selected model.",
            "",
        ]
    )
    path = output_dir / "mode_choice_transfer_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_experiment(args: argparse.Namespace) -> None:
    data = load_data(args.dataset_dir, args.rebuild_datasets)
    mapping = label_mapping(data.y_train)
    labels = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    y_train = encode(data.y_train, mapping)
    y_validation = encode(data.y_validation, mapping)
    y_test = encode(data.y_test, mapping)
    n_classes = len(labels)

    train_weights = normalize_weights(data.w_train)
    validation_weights = normalize_weights(data.w_validation)
    test_weights = normalize_weights(data.w_test)

    prior = prior_probabilities(y_train, n_classes, train_weights)
    prior_validation = np.tile(prior, (len(y_validation), 1))
    prior_test = np.tile(prior, (len(y_test), 1))

    LOGGER.info("Training unweighted XGBoost mode-choice model on %s rows.", len(y_train))
    unweighted = train_xgboost(data.x_train, y_train, args, sample_weight=None)
    LOGGER.info("Training survey-weighted XGBoost mode-choice model on %s rows.", len(y_train))
    survey_weighted = train_xgboost(data.x_train, y_train, args, sample_weight=train_weights)
    LOGGER.info("Training class-balanced XGBoost mode-choice model on %s rows.", len(y_train))
    balanced = train_xgboost(
        data.x_train,
        y_train,
        args,
        sample_weight=class_balanced_weights(y_train, n_classes, train_weights),
    )

    probabilities = {
        ("historical_prior_2017", "validation"): prior_validation,
        ("historical_prior_2017", "test_2022"): prior_test,
        ("xgboost_unweighted", "validation"): unweighted.predict_proba(data.x_validation),
        ("xgboost_unweighted", "test_2022"): unweighted.predict_proba(data.x_test),
        ("xgboost_survey_weighted", "validation"): survey_weighted.predict_proba(data.x_validation),
        ("xgboost_survey_weighted", "test_2022"): survey_weighted.predict_proba(data.x_test),
        ("xgboost_class_balanced", "validation"): balanced.predict_proba(data.x_validation),
        ("xgboost_class_balanced", "test_2022"): balanced.predict_proba(data.x_test),
    }

    metrics_rows = []
    per_class_frames = []
    split_labels = {"validation": y_validation, "test_2022": y_test}
    split_weights = {"validation": validation_weights, "test_2022": test_weights}
    for (method, split), probability in probabilities.items():
        y_true = split_labels[split]
        weights = split_weights[split]
        metrics_rows.append(
            evaluate_probabilities(
                method,
                split,
                y_true,
                probability,
                weights,
                labels,
                uses_2017_labels=True,
                device=args.device if method.startswith("xgboost") else "none",
            )
        )
        per_class_frames.append(per_class_metrics(method, split, y_true, probability, weights, labels))

    metrics = pd.DataFrame(metrics_rows)
    validation_metrics = metrics.loc[metrics["split"] == "validation"]
    selected_method = validation_metrics.sort_values(
        ["weighted_macro_f1", "weighted_balanced_accuracy", "weighted_log_loss"],
        ascending=[False, False, True],
    ).iloc[0]["method"]
    selected_test_probability = probabilities[(str(selected_method), "test_2022")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "mode_choice_transfer_metrics.csv"
    per_class_path = args.output_dir / "mode_choice_per_class_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(per_class_path, index=False)
    confusion_path = save_confusion_matrix(
        y_test,
        selected_test_probability,
        test_weights,
        labels,
        args.output_dir,
        str(selected_method),
    )
    report_path = write_report(metrics, per_class, str(selected_method), confusion_path, args.output_dir)

    LOGGER.info("Selected method by validation survey-weighted macro F1: %s", selected_method)
    LOGGER.info("Wrote %s", metrics_path)
    LOGGER.info("Wrote %s", per_class_path)
    LOGGER.info("Wrote %s", report_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
