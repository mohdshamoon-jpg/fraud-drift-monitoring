"""Run the reproducible fraud and drift-monitoring experiment."""

import argparse
import json
import os
import pickle
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "fraud_detection_matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.alerting import build_drift_alert
from src.config import DATA_PATH, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.cost_analysis import calculate_cost, select_cost_threshold
from src.data import load_data, preprocess, time_split
from src.drift import feature_drift, psi, simulate_drift
from src.evaluate import evaluate_model
from src.models_supervised import train_supervised_models
from src.models_unsupervised import anomaly_scores, train_isolation_forest


def _model_predictions(model_name, validation_scores, later_scores, prepared):
    """Choose a validation cost threshold and score both evaluation windows."""
    threshold, _ = select_cost_threshold(
        prepared.y_validation, validation_scores, prepared.amount_validation
    )
    rows = []
    for split_name, labels, scores, amounts in (
        ("validation", prepared.y_validation, validation_scores, prepared.amount_validation),
        ("later", prepared.y_later, later_scores, prepared.amount_later),
    ):
        metrics = evaluate_model(labels, scores, threshold=threshold)
        predictions = scores >= threshold
        costs = calculate_cost(labels, predictions, amounts)
        rows.append({"model": model_name, "split": split_name, **metrics, **costs})
    return threshold, rows


def main():
    """Train both approaches, evaluate later data, and write reproducible outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to Kaggle creditcard.csv (defaults to data/creditcard.csv).",
    )
    args = parser.parse_args()

    print("Fraud and Drift Monitoring")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Dataset: {args.data}")
    frame = load_data(args.data)
    train, validation, later = time_split(frame)
    prepared = preprocess(train, validation, later)

    print("\nChronological windows")
    for name, window in (("train", train), ("validation", validation), ("later", later)):
        print(
            f"{name:10} rows={len(window):>8,} fraud={int(window['Class'].sum()):>5,} "
            f"time={window['Time'].min():g}..{window['Time'].max():g}"
        )

    supervised = train_supervised_models(
        prepared.X_train,
        prepared.y_train,
        prepared.X_validation,
        prepared.y_validation,
    )
    rows = []
    model_objects = {}
    model_scores = {}
    for name, (model, validation_scores, validation_metrics) in supervised.items():
        later_scores = model.predict_proba(prepared.X_later)[:, 1]
        model_objects[name] = model
        model_scores[name] = (validation_scores, later_scores)
        _, model_rows = _model_predictions(name, validation_scores, later_scores, prepared)
        rows.extend(model_rows)
        print(f"\n{name}: validation PR-AUC={validation_metrics['pr_auc']:.6f}")

    isolation_forest = train_isolation_forest(prepared.X_train)
    isolation_validation_scores = anomaly_scores(isolation_forest, prepared.X_validation)
    isolation_later_scores = anomaly_scores(isolation_forest, prepared.X_later)
    isolation_name = "isolation_forest"
    model_objects[isolation_name] = isolation_forest
    model_scores[isolation_name] = (isolation_validation_scores, isolation_later_scores)
    _, isolation_rows = _model_predictions(
        isolation_name, isolation_validation_scores, isolation_later_scores, prepared
    )
    rows.extend(isolation_rows)
    print(
        f"{isolation_name}: validation PR-AUC="
        f"{evaluate_model(prepared.y_validation, isolation_validation_scores)['pr_auc']:.6f}"
    )

    metrics_frame = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    supervised_name = next(iter(supervised))
    supervised_validation, supervised_later = model_scores[supervised_name]
    baseline_metrics = evaluate_model(prepared.y_validation, supervised_validation)
    later_metrics = evaluate_model(prepared.y_later, supervised_later)
    feature_psi = feature_drift(prepared.X_train, prepared.X_later)
    score_psi = psi(supervised_validation, supervised_later)

    # Synthetic drift is a monitoring demonstration and is explicitly reported
    # separately from the observed chronological later window.
    synthetic_later = simulate_drift(later, severity=0.5)
    synthetic_scaled = prepared.scaler.transform(synthetic_later[prepared.X_train.columns])
    synthetic_features = pd.DataFrame(
        synthetic_scaled, columns=prepared.X_train.columns, index=later.index
    )
    synthetic_feature_psi = feature_drift(prepared.X_train, synthetic_features)
    alert = build_drift_alert(
        feature_psi,
        score_psi,
        pr_auc_validation=baseline_metrics["pr_auc"],
        pr_auc_later=later_metrics["pr_auc"],
    )
    drift_report = {
        "observed_later_window": {
            "feature_psi": feature_psi,
            "supervised_score_psi": score_psi,
            "validation_pr_auc": baseline_metrics["pr_auc"],
            "later_pr_auc": later_metrics["pr_auc"],
            "alert": alert,
        },
        "synthetic_demo": {
            "label": "synthetic; each numeric feature (except Time and Class) shifted by 0.5 standard deviations",
            "feature_psi": synthetic_feature_psi,
            "max_feature_psi": max(synthetic_feature_psi.values(), default=0.0),
        },
        "threshold_note": "Each model threshold was selected by estimated cost on validation and held fixed on the later window.",
        "psi_note": "PSI thresholds are initial monitoring heuristics and require operational calibration.",
    }
    drift_path = RESULTS_DIR / "drift_report.json"
    drift_path.write_text(json.dumps(drift_report, indent=2, allow_nan=False) + "\n")

    model_path = MODELS_DIR / "models.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(
            {"models": model_objects, "feature_columns": list(prepared.X_train.columns), "scaler": prepared.scaler},
            model_file,
        )

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, (validation_scores, _) in model_scores.items():
        precision, recall, _ = precision_recall_curve(prepared.y_validation, validation_scores)
        axis.plot(recall, precision, label=name)
    axis.set(title="Validation precision-recall curves", xlabel="Recall", ylabel="Precision")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "validation_pr_curves.png", dpi=160)
    plt.close(figure)

    report_path = RESULTS_DIR / "run_summary.md"
    report_lines = [
        "# Experiment run summary",
        "",
        "This file is generated by `python run_all.py`. Metrics below come from this dataset run.",
        "Thresholds were selected using validation data and held fixed for the later window.",
        "",
        "## Model results",
        "",
        metrics_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Drift and alert",
        "",
        f"Observed later-window alert: **{alert['status']}**.",
        "",
        *(f"- {reason}" for reason in (alert["reasons"] or ["No configured warning or retrain bound was crossed."])),
        "",
        f"Synthetic demo maximum feature PSI: {drift_report['synthetic_demo']['max_feature_psi']:.6f}.",
        "The synthetic result is separate from the observed later-window result.",
        "",
        "## Outputs",
        "",
        "- `metrics.csv`: PR-AUC, secondary ROC-AUC, threshold metrics, and estimated costs.",
        "- `drift_report.json`: observed and synthetic PSI measurements and alert details.",
        "- `validation_pr_curves.png`: validation precision-recall curves.",
        "- `models/models.pkl`: fitted models and preprocessing scaler.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nResults written under {RESULTS_DIR}")
    print(f"Observed later-window alert: {alert['status']}")
    for row in metrics_frame[metrics_frame["split"] == "later"].to_dict("records"):
        print(
            f"{row['model']}: later PR-AUC={row['pr_auc']:.6f}, "
            f"estimated cost={row['total_cost']:.2f}"
        )


if __name__ == "__main__":
    main()
