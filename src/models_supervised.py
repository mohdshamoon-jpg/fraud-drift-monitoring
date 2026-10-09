"""Supervised fraud classifiers for highly imbalanced labels."""

from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED
from src.evaluate import evaluate_model


def train_supervised_models(X_train, y_train, X_validation, y_validation):
    """Train a regularized, class-weighted logistic regression baseline.

    Class weighting avoids throwing away legitimate transactions or creating
    synthetic examples. The operating threshold is handled separately using
    validation business costs; the default probability cutoff is not assumed.
    """
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    validation_scores = model.predict_proba(X_validation)[:, 1]
    metrics = evaluate_model(y_validation, validation_scores)
    return {"class_weighted_logistic_regression": (model, validation_scores, metrics)}
