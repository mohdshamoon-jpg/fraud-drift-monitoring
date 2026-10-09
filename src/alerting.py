"""Translate drift measurements into monitoring actions."""

from src.config import (
    FEATURE_PSI_RETRAIN,
    FEATURE_PSI_WARNING,
    PR_AUC_DROP_RETRAIN,
    SCORE_PSI_RETRAIN,
    SCORE_PSI_WARNING,
)


def build_drift_alert(feature_psi, score_psi_value, pr_auc_validation=None, pr_auc_later=None):
    """Return an alert decision and readable reasons for retraining review.

    PSI and performance bounds are initial project settings, not validated
    production policies. The function reports measurements; it does not launch
    retraining automatically.
    """
    feature_psi = feature_psi or {}
    retrain_features = [name for name, value in feature_psi.items() if value >= FEATURE_PSI_RETRAIN]
    warning_features = [name for name, value in feature_psi.items() if value >= FEATURE_PSI_WARNING]
    reasons = []
    if retrain_features:
        reasons.append("feature PSI reached the retrain bound: " + ", ".join(retrain_features))
    if score_psi_value >= SCORE_PSI_RETRAIN:
        reasons.append("model-score PSI reached the retrain bound")
    elif score_psi_value >= SCORE_PSI_WARNING:
        reasons.append("model-score PSI crossed the warning bound")
    if pr_auc_validation is not None and pr_auc_later is not None and pr_auc_validation > 0:
        relative_drop = (pr_auc_validation - pr_auc_later) / pr_auc_validation
        if relative_drop >= PR_AUC_DROP_RETRAIN:
            reasons.append(f"PR-AUC fell {relative_drop:.1%} from validation to later window")
    else:
        relative_drop = None
    if not reasons and warning_features:
        reasons.append("feature PSI crossed the warning bound: " + ", ".join(warning_features))

    return {
        "status": "RETRAIN_REVIEW" if retrain_features or (score_psi_value >= SCORE_PSI_RETRAIN) or (relative_drop is not None and relative_drop >= PR_AUC_DROP_RETRAIN) else ("WARNING" if reasons else "OK"),
        "reasons": reasons,
        "warning_features": warning_features,
        "retrain_features": retrain_features,
        "relative_pr_auc_drop": relative_drop,
    }
