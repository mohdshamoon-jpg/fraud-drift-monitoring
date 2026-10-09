"""Estimate fraud losses and customer-friction costs."""

from typing import Dict, Tuple

import numpy as np

from src.config import (
    CHARGEBACK_FEE,
    FALSE_NEGATIVE_COST_MULTIPLIER,
    FALSE_POSITIVE_AMOUNT_RATE,
    FALSE_POSITIVE_FIXED_COST,
)


def calculate_cost(y_true, predictions, amounts) -> Dict[str, float]:
    """Estimate alert friction and missed-fraud losses from original amounts.

    Assumptions are illustrative: a false positive costs a fixed review/block
    amount plus a small fraction of transaction value; a false negative costs
    the missed transaction value plus a chargeback fee, scaled by a configurable
    multiplier. Replace these assumptions with company loss data when available.
    """
    labels = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predictions, dtype=bool)
    transaction_amounts = np.asarray(amounts, dtype=float)
    if not (labels.ndim == predicted.ndim == transaction_amounts.ndim == 1):
        raise ValueError("Labels, predictions, and amounts must be one-dimensional.")
    if not (len(labels) == len(predicted) == len(transaction_amounts)):
        raise ValueError("Labels, predictions, and amounts must have equal lengths.")
    if not np.isfinite(transaction_amounts).all() or (transaction_amounts < 0).any():
        raise ValueError("Transaction amounts must be finite and non-negative.")

    false_positive = predicted & (labels == 0)
    false_negative = (~predicted) & (labels == 1)
    false_positive_cost = (
        FALSE_POSITIVE_FIXED_COST + FALSE_POSITIVE_AMOUNT_RATE * transaction_amounts
    )
    false_negative_cost = FALSE_NEGATIVE_COST_MULTIPLIER * (
        transaction_amounts + CHARGEBACK_FEE
    )
    fp_total = float(false_positive_cost[false_positive].sum())
    fn_total = float(false_negative_cost[false_negative].sum())
    return {
        "false_positive_count": int(false_positive.sum()),
        "false_negative_count": int(false_negative.sum()),
        "false_positive_cost": fp_total,
        "false_negative_cost": fn_total,
        "total_cost": fp_total + fn_total,
    }


def select_cost_threshold(y_validation, scores, amounts) -> Tuple[float, Dict[str, float]]:
    """Choose the minimum estimated cost threshold on validation data only."""
    score_values = np.asarray(scores, dtype=float)
    if score_values.ndim != 1 or len(score_values) == 0 or not np.isfinite(score_values).all():
        raise ValueError("Validation scores must be a non-empty finite 1D array.")
    # Evaluate a compact deterministic grid. This keeps threshold tuning cheap
    # even when every transaction receives a unique score.
    thresholds = np.unique(np.quantile(score_values, np.linspace(0, 1, 101)))
    best_threshold = None
    best_cost = None
    for threshold in thresholds:
        predictions = score_values >= threshold
        cost = calculate_cost(y_validation, predictions, amounts)
        key = (cost["total_cost"], -float(threshold))
        if best_cost is None or key < best_cost[0]:
            best_cost = (key, cost)
            best_threshold = float(threshold)
    return best_threshold, best_cost[1]
