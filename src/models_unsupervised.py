"""Unsupervised anomaly detection using Isolation Forest."""

from sklearn.ensemble import IsolationForest

from src.config import RANDOM_SEED


def train_isolation_forest(X_train):
    """Fit an Isolation Forest and return fraud-oriented anomaly scores.

    The model sees every training row without labels. A bounded subsample keeps
    training practical on a laptop; anomaly score is the negated decision
    function so larger values consistently mean more suspicious.
    """
    model = IsolationForest(
        n_estimators=160,
        max_samples=512,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model, features):
    """Return Isolation Forest scores where higher means more anomalous."""
    return -model.decision_function(features)
