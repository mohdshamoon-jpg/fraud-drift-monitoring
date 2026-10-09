"""Measure feature and model-score drift over time."""

from typing import Dict

import numpy as np
import pandas as pd


def psi(reference, new, bins=10):
    """Calculate Population Stability Index using reference quantile bins."""
    reference_values = np.asarray(reference, dtype=float)
    new_values = np.asarray(new, dtype=float)
    if reference_values.ndim != 1 or new_values.ndim != 1:
        raise ValueError("PSI inputs must be one-dimensional.")
    reference_values = reference_values[np.isfinite(reference_values)]
    new_values = new_values[np.isfinite(new_values)]
    if not len(reference_values) or not len(new_values):
        raise ValueError("PSI inputs must each contain finite values.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(new_values == reference_values[0]) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0]
    new_counts = np.histogram(new_values, bins=edges)[0]
    # Smoothing avoids undefined logarithms when a bin is empty.
    reference_share = (reference_counts + 0.5) / (reference_counts.sum() + 0.5 * len(reference_counts))
    new_share = (new_counts + 0.5) / (new_counts.sum() + 0.5 * len(new_counts))
    return float(np.sum((new_share - reference_share) * np.log(new_share / reference_share)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, bins=10) -> Dict[str, float]:
    """Compute PSI for shared numeric predictors, excluding chronological Time.

    ``Time`` is used to create the evaluation windows and naturally advances
    from train to later data. Including it would make a high PSI nearly
    guaranteed by the split itself rather than indicate changed transaction
    behavior, so it is omitted from the feature drift alert.
    """
    common = [column for column in reference.columns if column in current.columns]
    if not common:
        raise ValueError("No shared features available for drift comparison.")
    return {
        column: psi(reference[column], current[column], bins=bins)
        for column in common
        if column != "Time"
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(current[column])
    }


def simulate_drift(frame: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Create an explicitly synthetic shifted copy for a monitoring demo.

    Numeric non-label features are shifted by ``severity`` training standard
    deviations. Labels and timestamps remain unchanged so the synthetic data
    can be evaluated separately without being confused with observed drift.
    """
    if not 0 <= severity <= 1:
        raise ValueError("severity must be between 0 and 1.")
    shifted = frame.copy()
    for column in shifted.columns:
        if column in {"Class", "Time"} or not pd.api.types.is_numeric_dtype(shifted[column]):
            continue
        standard_deviation = shifted[column].std()
        if np.isfinite(standard_deviation) and standard_deviation > 0:
            shifted[column] = shifted[column] + severity * standard_deviation
    return shifted
