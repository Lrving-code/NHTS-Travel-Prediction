"""Build train, validation, and 2022 test data for trip mode prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import (
    DEFAULT_INTERIM_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_GROUPS,
    HOUSEHOLD_ID,
    INVALID_TARGET_CODES,
    MODE_CHOICE_FEATURES,
    PERSON_ID,
    TABLE_SPECS,
    TARGET_COLUMN,
    preprocessed_path,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    """Paths and dimensions from the dataset-building step."""

    output_dir: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    target_column: str
    feature_path: Path
    report_path: Path


def load_preprocessed_tables(interim_dir: Path) -> dict[str, pd.DataFrame]:
    """Load preprocessed table-year files."""
    data: dict[str, pd.DataFrame] = {}
    for table_name in TABLE_SPECS:
        for year in (2017, 2022):
            path = preprocessed_path(interim_dir, table_name, year)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing preprocessed file: {path}. Run `python src/run_mode_choice_branch.py preprocess` first."
                )
            data[f"{table_name}_{year}"] = pd.read_csv(path, low_memory=False)
    return data


def merge_trip_person_household(data: dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    """Merge trip, person, and household tables at the trip-person level."""
    trip = data[f"trip_{year}"]
    person = data[f"per_{year}"]
    household = data[f"hh_{year}"]

    required_trip_columns = {HOUSEHOLD_ID, PERSON_ID, TARGET_COLUMN}
    missing_trip = required_trip_columns - set(trip.columns)
    if missing_trip:
        raise ValueError(f"trip_{year} missing required column(s): {sorted(missing_trip)}")

    merged = trip.merge(person, on=[HOUSEHOLD_ID, PERSON_ID], how="left", suffixes=("_trip", "_per"))
    merged = merged.merge(household, on=HOUSEHOLD_ID, how="left", suffixes=("", "_hh"))
    return merged


def resolve_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Resolve feature columns after table merges and suffixing."""
    final_features: list[str] = []
    missing_features: list[str] = []
    for feature in MODE_CHOICE_FEATURES:
        candidates = (f"{feature}_trip", f"{feature}_per", feature, f"{feature}_hh")
        selected = next((candidate for candidate in candidates if candidate in frame.columns), None)
        if selected is None:
            missing_features.append(feature)
        else:
            final_features.append(selected)

    if not final_features:
        raise ValueError("No mode-choice features were found after merging source tables.")
    if missing_features:
        missing_text = ", ".join(missing_features)
        raise ValueError(f"Merged data missing expected mode-choice feature(s): {missing_text}")
    return tuple(final_features)


def normalize_target(series: pd.Series) -> pd.Series:
    """Return a cleaned string target label series."""
    target = series.astype("string").str.strip()
    target = target.mask(target.isna() | target.isin(INVALID_TARGET_CODES))
    return target


def drop_invalid_targets(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows without a usable target value."""
    normalized = normalize_target(target)
    keep_mask = normalized.notna()
    return features.loc[keep_mask].reset_index(drop=True), normalized.loc[keep_mask].reset_index(drop=True)


def encode_feature_frames(
    frame_2017: pd.DataFrame,
    frame_2022: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features consistently across 2017 and 2022."""
    row_count_2017 = len(frame_2017)
    combined = pd.concat([frame_2017, frame_2022], axis=0, ignore_index=True)
    encoded = pd.DataFrame(index=combined.index)

    for column in combined.columns:
        numeric_values = pd.to_numeric(combined[column], errors="coerce")
        numeric_ratio = float(numeric_values.notna().mean()) if len(numeric_values) else 0.0
        if numeric_ratio >= 0.95:
            median_value = numeric_values.median()
            fill_value = 0.0 if pd.isna(median_value) else float(median_value)
            encoded[column] = numeric_values.fillna(fill_value)
        else:
            values = combined[column].astype("string").fillna("Unknown")
            codes, _ = pd.factorize(values, sort=True)
            encoded[column] = codes.astype(np.int32)

    encoded_2017 = encoded.iloc[:row_count_2017].reset_index(drop=True)
    encoded_2022 = encoded.iloc[row_count_2017:].reset_index(drop=True)
    return encoded_2017, encoded_2022


def stratify_or_none(target: pd.Series, test_size: float) -> pd.Series | None:
    """Return a stratification target only when the split is feasible."""
    counts = target.value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None

    validation_rows = math.ceil(len(target) * test_size)
    train_rows = len(target) - validation_rows
    if validation_rows < len(counts) or train_rows < len(counts):
        return None
    return target


def save_dataset_files(
    output_dir: Path,
    features: tuple[str, ...],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Path:
    """Save combined and split feature/label files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.txt"
    feature_path.write_text("\n".join(features) + "\n", encoding="utf-8")

    split_payloads = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }
    for split_name, (features_frame, target_series) in split_payloads.items():
        combined = features_frame.copy()
        combined[TARGET_COLUMN] = target_series.to_numpy()
        combined.to_csv(output_dir / f"{split_name}.csv", index=False)
        features_frame.to_csv(output_dir / f"X_{split_name}.csv", index=False)
        target_series.to_csv(output_dir / f"y_{split_name}.csv", index=False)

    return feature_path


def target_distribution_frame(targets: dict[str, pd.Series]) -> pd.DataFrame:
    """Create a target-distribution table across splits."""
    distributions = {split: series.value_counts(normalize=True) * 100 for split, series in targets.items()}
    all_modes = sorted(set().union(*(distribution.index for distribution in distributions.values())))
    rows = []
    for mode in all_modes:
        row = {"TRPTRANS": mode}
        for split, distribution in distributions.items():
            row[f"{split}_percent"] = distribution.get(mode, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def write_dataset_report(
    output_dir: Path,
    report_path: Path,
    features: tuple[str, ...],
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    """Write a markdown report for the generated modeling data."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    distribution = target_distribution_frame(
        {"train": y_train, "validation": y_validation, "test": y_test}
    )
    distribution.to_csv(output_dir / "target_distribution.csv", index=False)

    lines = [
        "# Mode-Choice Dataset Report",
        "",
        "This branch predicts trip-level `TRPTRANS` using 2017 data for training/validation and 2022 data for transfer testing.",
        "",
        "| Split | Rows | Features |",
        "|---|---:|---:|",
        f"| train | {len(y_train):,} | {len(features):,} |",
        f"| validation | {len(y_validation):,} | {len(features):,} |",
        f"| test_2022 | {len(y_test):,} | {len(features):,} |",
        "",
        "## Feature Groups",
        "",
    ]
    for group_name, group_features in FEATURE_GROUPS.items():
        selected = [feature for feature in features if any(raw in feature for raw in group_features)]
        lines.append(f"- `{group_name}`: {', '.join(selected) if selected else 'none'}")

    lines.extend(
        [
            "",
            "## Target Distribution",
            "",
            "| TRPTRANS | Train (%) | Validation (%) | Test 2022 (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in distribution.itertuples(index=False):
        lines.append(
            f"| {row.TRPTRANS} | {row.train_percent:.2f} | "
            f"{row.validation_percent:.2f} | {row.test_percent:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `train.csv`, `validation.csv`, `test.csv`: features and target in one file.",
            "- `X_train.csv`, `X_validation.csv`, `X_test.csv`: feature matrices.",
            "- `y_train.csv`, `y_validation.csv`, `y_test.csv`: target vectors.",
            "- `features.txt`: resolved post-merge feature columns.",
            "- `target_distribution.csv`: target distribution across splits.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_mode_choice_datasets(
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR / "datasets",
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBuildResult:
    """Build train, validation, and transfer-test CSV files."""
    data = load_preprocessed_tables(interim_dir)
    merged_2017 = merge_trip_person_household(data, 2017)
    merged_2022 = merge_trip_person_household(data, 2022)
    features = resolve_feature_columns(merged_2017)

    x_2017_raw, y_2017 = drop_invalid_targets(merged_2017.loc[:, features], merged_2017[TARGET_COLUMN])
    x_2022_raw, y_2022 = drop_invalid_targets(merged_2022.loc[:, features], merged_2022[TARGET_COLUMN])
    x_2017, x_2022 = encode_feature_frames(x_2017_raw, x_2022_raw)

    stratify_target = stratify_or_none(y_2017, test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_2017,
        y_2017,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    feature_path = save_dataset_files(
        output_dir,
        features,
        x_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        x_validation.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        x_2022.reset_index(drop=True),
        y_2022.reset_index(drop=True),
    )
    report_path = output_dir / "dataset_report.md"
    write_dataset_report(output_dir, report_path, features, y_train, y_validation, y_2022)

    return DatasetBuildResult(
        output_dir=output_dir,
        train_rows=len(y_train),
        validation_rows=len(y_validation),
        test_rows=len(y_2022),
        feature_count=len(features),
        target_column=TARGET_COLUMN,
        feature_path=feature_path,
        report_path=report_path,
    )
