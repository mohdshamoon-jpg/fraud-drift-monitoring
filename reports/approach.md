# AIML-02: Fraud Detection and Drift Monitoring

## Problem and evaluation design

Fraud models can lose value as transaction behavior changes. This project
implements a small, reproducible benchmark that compares a supervised model
with an unsupervised detector and monitors a later chronological window.
Transactions are sorted by `Time`; the earliest 60% is used for fitting, the
next 20% for model and threshold decisions, and the final 20% is held for
later-window reporting. No later-window labels or features influence fitting
or threshold selection. `RANDOM_SEED=42` controls stochastic components.

## Modeling decisions

The supervised baseline is regularized logistic regression with balanced class
weights. Fraud is extremely rare, so an ordinary unweighted loss could favor
the majority class. Class weighting makes fraud errors matter more during
fitting without discarding legitimate examples or treating synthetic examples
as observed transactions. Isolation Forest is trained without labels and
produces a ranking in which more anomalous transactions receive higher scores.
Its bounded subsampling makes it practical on a laptop.

Average precision, computed with scikit-learn's `average_precision_score`, is
the primary ranking metric. ROC-AUC is reported as a secondary measure. Under
extreme imbalance, a high ROC-AUC may coexist with an alert list dominated by
false positives; precision-recall analysis directly exposes that tradeoff.

## Threshold, costs, and monitoring

For each model, a deterministic score-quantile grid is evaluated against
validation labels and untouched transaction amounts. The selected threshold
minimizes estimated validation cost and is then fixed for the later window.
The initial assumptions charge a false positive `$5 + 1% of transaction
Amount` and a false negative `Amount + $15`, with configurable multipliers.
These values illustrate how threshold choice can reflect both customer
friction and missed-fraud losses; they are not production-calibrated.

Feature PSI compares training and later distributions, while score PSI compares
validation and later supervised scores. Configurable bounds classify drift as
a warning or retraining review. A separate synthetic demonstration shifts
numeric features (except `Time` and `Class`) by half a standard deviation. It is
reported independently from observed chronological drift. Drift is a review
signal rather than proof of performance failure, and the alert does not
automatically retrain a model.

## Results and interpretation

The Kaggle file used for this run has 284,807 transactions and 492 frauds
(0.173%). The chronological windows contain 170,884 train rows (360 frauds),
56,961 validation rows (57 frauds), and 56,962 later rows (75 frauds).

| Model | Window | PR-AUC | ROC-AUC | Frozen threshold | Precision | Recall | Estimated cost |
|---|---|---:|---:|---:|---:|---:|---:|
| Class-weighted logistic regression | Validation | 0.7719 | 0.9709 | 0.793241 | 0.0860 | 0.8596 | $7,683.50 |
| Class-weighted logistic regression | Later | 0.7436 | 0.9820 | 0.793241 | 0.1404 | 0.8533 | $5,999.99 |
| Isolation Forest | Validation | 0.0213 | 0.9336 | 0.283851 | 0.0000 | 0.0000 | $13,150.74 |
| Isolation Forest | Later | 0.0394 | 0.9389 | 0.283851 | 0.0000 | 0.0000 | $9,116.17 |

On this chronological split, the supervised model ranks fraud substantially
better. Its validation-selected threshold catches 49 of 57 validation frauds
and 64 of 75 later frauds. The Isolation Forest's cost-minimizing validation
threshold flags one transaction in each evaluation window and catches none of
the frauds. This is a useful negative result: unsupervised anomaly ranking is
not automatically an effective fraud detector, and its threshold is sensitive
to the assumed costs. The displayed costs use the illustrative assumptions
above and are not real issuer losses.

Feature PSI triggers `RETRAIN_REVIEW` for V1, V3, V11, V12, V15, V25, and V28.
Time is excluded from PSI because it advances by construction across
chronological windows. Validation-to-later supervised PR-AUC decreases 3.7%,
below the configured 20% performance-drop bound. The alert is therefore driven
by feature PSI, not demonstrated performance degradation. PSI bounds are
heuristics and should be calibrated; this two-day 2013 benchmark cannot
establish how a live fintech model will drift.

The full run is recorded in `results/run_summary.md`, `results/metrics.csv`,
and `results/drift_report.json`. Rerun `python run_all.py` after changing data
or configuration so these generated values match the run being discussed.

## Reproduction

Install dependencies with `python -m pip install -r requirements.txt`, place
the Kaggle file at `data/creditcard.csv`, and run `python run_all.py` from the
repository root. Setup details and output descriptions are in `README.md`.
