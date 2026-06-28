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

from mode_choice_branch.common import MODE_GROUP_COLUMN, MODE_GROUP_LABELS, MODE_GROUP_ORDER  # noqa: E402
from mode_choice_branch.datasets import build_mode_choice_datasets  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "mode_choice_branch" / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mode_choice_transfer"


@dataclass(frozen=True)
class ModeChoiceData:
    """Loaded mode-choice transfer splits."""

    x_train: pd.DataFrame
    y_train: np.ndarray
    x_validation: pd.DataFrame
    y_validation: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray


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


def load_split(dataset_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = dataset_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/run_mode_choice_branch.py build-datasets first.")
    frame = pd.read_csv(path, low_memory=False)
    if MODE_GROUP_COLUMN not in frame.columns:
        raise ValueError(f"{path} does not contain harmonized target column {MODE_GROUP_COLUMN}.")
    return frame.drop(columns=[MODE_GROUP_COLUMN]), frame[MODE_GROUP_COLUMN].astype(str).to_numpy()


def load_data(dataset_dir: Path, rebuild: bool) -> ModeChoiceData:
    ensure_datasets(dataset_dir, rebuild)
    x_train, y_train = load_split(dataset_dir, "train")
    x_validation, y_validation = load_split(dataset_dir, "validation")
    x_test, y_test = load_split(dataset_dir, "test")
    return ModeChoiceData(x_train, y_train, x_validation, y_validation, x_test, y_test)


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


def prior_probabilities(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    return counts / counts.sum()


def class_balanced_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / (n_classes * np.maximum(counts, 1.0))
    return weights[y_train]


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
    labels: list[str],
    uses_2017_labels: bool,
    device: str,
) -> dict[str, str | int | float | bool]:
    probabilities = probability_rows(probabilities, len(labels))
    y_pred = probabilities.argmax(axis=1)
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    one_hot = np.eye(len(labels), dtype=float)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "method": method,
        "split": split,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(np.mean([truth in candidates for truth, candidates in zip(y_true, top2)])),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(labels))))),
        "multiclass_brier": brier,
        "uses_2017_labels": uses_2017_labels,
        "uses_2022_labels_for_training": False,
        "device": device,
    }


def per_class_metrics(
    method: str,
    split: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    y_pred = probability_rows(probabilities, len(labels)).argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
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
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    output_dir: Path,
    method: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = probability_rows(probabilities, len(labels)).argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize="true")
    display_labels = [MODE_GROUP_LABELS.get(label, label) for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted mode group")
    ax.set_ylabel("Observed mode group")
    ax.set_title("2022 Harmonized Mode-Choice Transfer Confusion Matrix")
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
    balanced_row = test_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
    class_rows = per_class.loc[
        (per_class["method"] == selected_method) & (per_class["split"] == "test_2022")
    ].copy()
    class_rows = class_rows.sort_values("support", ascending=False)
    display_confusion_path = confusion_path.relative_to(PROJECT_ROOT).as_posix()

    lines = [
        "# Harmonized Mode-Choice Transfer Experiment",
        "",
        "## Scope",
        "",
        "This experiment evaluates trip-level mode-choice transfer after mapping year-specific NHTS `TRPTRANS` codes into comparable `MODE_GROUP` labels. It trains only on 2017 labels, selects the XGBoost operating point by 2017 validation macro F1, and uses 2022 labels only for final evaluation.",
        "",
        "## 2022 Test Metrics",
        "",
        "| Method | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Top-2 accuracy | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_metrics.sort_values("macro_f1", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | "
            f"{row.macro_f1:.4f} | {row.weighted_f1:.4f} | {row.top2_accuracy:.4f} | {row.log_loss:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Selected Operating Point",
            "",
            f"The validation-selected model is `{selected_method}`. On 2022 it reaches accuracy `{selected_row.accuracy:.4f}`, balanced accuracy `{selected_row.balanced_accuracy:.4f}`, and macro F1 `{selected_row.macro_f1:.4f}`.",
            "",
            f"The highest balanced-accuracy model is `{balanced_row.method}` at `{balanced_row.balanced_accuracy:.4f}`, but it lowers macro F1 and log-loss calibration. The paper-safe interpretation should therefore report the trade-off rather than treating ordinary accuracy as sufficient.",
            "",
            "The ordinary accuracy is high because private-vehicle trips dominate the target distribution. The paper-safe interpretation should therefore emphasize balanced accuracy, macro F1, and per-class behavior rather than claiming a solved full mode-choice task.",
            "",
            f"- Confusion matrix: `{display_confusion_path}`",
            "",
            "## Per-Class Test Metrics for Selected Model",
            "",
            "| Mode group | Label | Precision | Recall | F1 | Support |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows.itertuples(index=False):
        lines.append(
            f"| {row.mode_group} | {row.label} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.support} |"
        )
    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `mode_choice_transfer_metrics.csv`: validation and 2022 test aggregate metrics.",
            "- `mode_choice_per_class_metrics.csv`: per-class precision, recall, F1, and support.",
            "- `*_test_confusion_matrix.png`: row-normalized 2022 confusion matrix for the validation-selected model.",
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

    prior = prior_probabilities(y_train, n_classes)
    prior_validation = np.tile(prior, (len(y_validation), 1))
    prior_test = np.tile(prior, (len(y_test), 1))

    LOGGER.info("Training unweighted XGBoost mode-choice model on %s rows.", len(y_train))
    unweighted = train_xgboost(data.x_train, y_train, args, sample_weight=None)
    LOGGER.info("Training class-balanced XGBoost mode-choice model on %s rows.", len(y_train))
    balanced = train_xgboost(data.x_train, y_train, args, sample_weight=class_balanced_weights(y_train, n_classes))

    probabilities = {
        ("historical_prior_2017", "validation"): prior_validation,
        ("historical_prior_2017", "test_2022"): prior_test,
        ("xgboost_unweighted", "validation"): unweighted.predict_proba(data.x_validation),
        ("xgboost_unweighted", "test_2022"): unweighted.predict_proba(data.x_test),
        ("xgboost_class_balanced", "validation"): balanced.predict_proba(data.x_validation),
        ("xgboost_class_balanced", "test_2022"): balanced.predict_proba(data.x_test),
    }

    metrics_rows = []
    per_class_frames = []
    split_labels = {"validation": y_validation, "test_2022": y_test}
    for (method, split), probability in probabilities.items():
        y_true = split_labels[split]
        metrics_rows.append(
            evaluate_probabilities(
                method,
                split,
                y_true,
                probability,
                labels,
                uses_2017_labels=True,
                device=args.device if method.startswith("xgboost") else "none",
            )
        )
        per_class_frames.append(per_class_metrics(method, split, y_true, probability, labels))

    metrics = pd.DataFrame(metrics_rows)
    validation_metrics = metrics.loc[metrics["split"] == "validation"]
    selected_method = validation_metrics.sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"],
        ascending=[False, False, True],
    ).iloc[0]["method"]
    selected_test_probability = probabilities[(str(selected_method), "test_2022")]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "mode_choice_transfer_metrics.csv"
    per_class_path = args.output_dir / "mode_choice_per_class_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(per_class_path, index=False)
    confusion_path = save_confusion_matrix(y_test, selected_test_probability, labels, args.output_dir, str(selected_method))
    report_path = write_report(metrics, per_class, str(selected_method), confusion_path, args.output_dir)

    LOGGER.info("Selected method by validation macro F1: %s", selected_method)
    LOGGER.info("Wrote %s", metrics_path)
    LOGGER.info("Wrote %s", per_class_path)
    LOGGER.info("Wrote %s", report_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
