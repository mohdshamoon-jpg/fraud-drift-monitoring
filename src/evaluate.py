"""Evaluate fraud scores and choose operating thresholds on validation data."""

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, scores, threshold=None) -> Dict[str, float]:
    """Return PR-AUC, ROC-AUC, and optional threshold-based metrics.

    Scores should be larger for transactions considered more likely to be
    fraudulent. PR-AUC uses average precision, the project's primary metric.
    """
    labels = np.asarray(y_true, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or score_values.ndim != 1 or len(labels) != len(score_values):
        raise ValueError("y_true and scores must be one-dimensional arrays of equal length.")
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty set of transactions.")
    if not np.isfinite(score_values).all():
        raise ValueError("Scores must contain only finite values.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    metrics = {
        "pr_auc": float(average_precision_score(labels, score_values)),
        "roc_auc": (
            float(roc_auc_score(labels, score_values))
            if len(np.unique(labels)) == 2
            else float("nan")
        ),
    }
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        predictions = score_values >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics.update(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )
    return metrics


def select_threshold(
    y_validation,
    validation_scores,
    thresholds: Iterable[float] = None,
    objective: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """Select a threshold using validation labels only.

    The default maximizes validation F1. The returned threshold should be
    frozen before evaluating the later time window. Ties prefer the higher
    threshold, which produces fewer alerts.
    """
    if objective not in {"f1", "recall", "precision"}:
        raise ValueError("objective must be 'f1', 'recall', or 'precision'.")
    labels = np.asarray(y_validation, dtype=int)
    scores = np.asarray(validation_scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Validation labels and scores must be equal-length 1D arrays.")
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Validation data must be non-empty and scores must be finite.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Validation data must contain both classes for threshold selection.")

    candidates = (
        np.unique(scores)
        if thresholds is None
        else np.asarray(list(thresholds), dtype=float)
    )
    if candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all():
        raise ValueError("thresholds must be a non-empty collection of finite values.")

    best_threshold = None
    best_metrics = None
    best_key = None
    for threshold in candidates:
        metrics = evaluate_model(labels, scores, float(threshold))
        # Threshold is the secondary key so ties favor fewer positive alerts.
        key = (metrics[objective], float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
