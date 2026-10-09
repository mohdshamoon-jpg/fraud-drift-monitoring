"""Load, inspect, split, and preprocess transaction data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH,
    LATER_RATIO,
    MIN_FRAUD_CASES_PER_SPLIT,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


@dataclass
class PreparedData:
    """Leakage-safe model inputs and untouched labels/cost amounts."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_later: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_later: pd.Series
    amount_train: pd.Series
    amount_validation: pd.Series
    amount_later: pd.Series
    scaler: StandardScaler


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and check the schema and basic data integrity."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )

    frame = pd.read_csv(path)
    required = {TIME_COLUMN, TARGET_COLUMN, "Amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no transactions.")
    if frame[[TIME_COLUMN, TARGET_COLUMN, "Amount"]].isna().any().any():
        raise ValueError("Time, Class, and Amount must not contain missing values.")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")
    return frame


def time_split(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by event time and split into contiguous 60/20/20 windows."""
    if not {TIME_COLUMN, TARGET_COLUMN}.issubset(frame.columns):
        raise ValueError(f"Input must contain {TIME_COLUMN!r} and {TARGET_COLUMN!r}.")
    if len(frame) < 3:
        raise ValueError("At least three rows are required for chronological splitting.")
    if abs(TRAIN_RATIO + VALIDATION_RATIO + LATER_RATIO - 1.0) > 1e-9:
        raise ValueError("Train, validation, and later ratios must sum to 1.")

    ordered = frame.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    train_end = int(len(ordered) * TRAIN_RATIO)
    validation_end = int(len(ordered) * (TRAIN_RATIO + VALIDATION_RATIO))
    windows = (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
    for name, window in zip(("train", "validation", "later"), windows):
        fraud_count = int(window[TARGET_COLUMN].sum())
        if fraud_count < MIN_FRAUD_CASES_PER_SPLIT:
            raise ValueError(
                f"The chronological {name} window has {fraud_count} fraud cases; "
                f"at least {MIN_FRAUD_CASES_PER_SPLIT} are required for meaningful evaluation."
            )
    return windows


def preprocess(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    later: pd.DataFrame,
) -> PreparedData:
    """Scale model features using train statistics and retain raw amounts.

    ``Time`` and ``Amount`` are model features. Separate unscaled amount series
    are returned so business-cost calculations use the original transaction
    value. The target is never included in the scaler.
    """
    feature_columns = [
        column
        for column in train.columns
        if column not in {TARGET_COLUMN}
    ]
    if TIME_COLUMN not in feature_columns or "Amount" not in feature_columns:
        raise ValueError("Train data must contain Time and Amount features.")

    expected = set(feature_columns) | {TARGET_COLUMN}
    for name, window in (("validation", validation), ("later", later)):
        if set(window.columns) != expected:
            raise ValueError(f"{name} data columns do not match training data columns.")

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train[feature_columns]),
        columns=feature_columns,
        index=train.index,
    )
    X_validation = pd.DataFrame(
        scaler.transform(validation[feature_columns]),
        columns=feature_columns,
        index=validation.index,
    )
    X_later = pd.DataFrame(
        scaler.transform(later[feature_columns]),
        columns=feature_columns,
        index=later.index,
    )

    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_later=X_later,
        y_train=train[TARGET_COLUMN].astype("int8").copy(),
        y_validation=validation[TARGET_COLUMN].astype("int8").copy(),
        y_later=later[TARGET_COLUMN].astype("int8").copy(),
        amount_train=train["Amount"].copy(),
        amount_validation=validation["Amount"].copy(),
        amount_later=later["Amount"].copy(),
        scaler=scaler,
    )
