# Fraud and Anomaly Detection with Drift Monitoring

This project compares a class-weighted supervised fraud model with an
unsupervised Isolation Forest on Kaggle's Credit Card Fraud Detection dataset.
It evaluates a later chronological window, measures feature and score drift,
raises a retraining review alert, and estimates the cost of false positives and
missed fraud.

## Dataset

Download `creditcard.csv` from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in `data/creditcard.csv`. The file is not included in this project.
See [data/README.md](data/README.md) for the expected schema. The `Class` column
contains the target (`1` is fraud); `Amount` is retained in its original units
for cost estimates.

## Setup

Python 3.10 or newer is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the experiment

```bash
python run_all.py
```

To use a CSV at another location:

```bash
python run_all.py --data /path/to/creditcard.csv
```

## Visual dashboard

Launch the upload-and-run dashboard from the project root:

```bash
streamlit run streamlit_app.py
```

Upload `creditcard.csv` in the sidebar (or place it at the default path), then
select **Run experiment**. The page displays later-window alert status, model
metrics and estimated costs, feature PSI, validation PR curves, and a separate
synthetic-drift panel. Download the metrics CSV and drift JSON from the page.

The run sorts transactions by `Time` and uses the first 60% for training, the
next 20% for validation, and the final 20% as the later window. The random seed
is 42. Preprocessing is fit on training data only. The later window is never
used to fit models or choose thresholds. Each model's cost-minimizing threshold
is selected on validation and held fixed during later-window evaluation.

## Approach and decisions

- **Imbalance:** Logistic regression uses `class_weight="balanced"`. This
  adjusts the supervised loss for the rare fraud class while retaining the
  original transactions. No synthetic oversampling is needed for the initial
  baseline. Isolation Forest provides a separate label-free anomaly ranking.
- **Comparison:** Average precision (`average_precision_score`) is the primary
  metric and ROC-AUC is secondary. With very rare fraud, PR-AUC focuses on the
  precision/recall tradeoff among positive alerts; ROC-AUC can look strong even
  when many alerts are false positives.
- **Threshold and costs:** Validation thresholds minimize an illustrative
  cost function on a deterministic 101-point score-quantile grid. A false
  positive costs `$5 + 1% of Amount`; a false negative costs `Amount + $15`,
  multiplied by the configured severity factor. These are example assumptions,
  not calibrated fintech loss estimates. Replace them with company data before
  operational use.
- **Drift:** Population Stability Index (PSI) compares train distributions to
  the later window and validation scores to later scores. The runner also
  creates a separate, clearly labeled synthetic example that shifts numeric
  features by 0.5 standard deviations. PSI warning/retraining bounds are
  configurable starting heuristics, not validated policies. Alerts request a
  retraining review; they do not automatically retrain or deploy a model.

## Outputs

`run_all.py` writes these files under `results/`:

- `metrics.csv`: PR-AUC, ROC-AUC, threshold-based metrics, and estimated costs
  for validation and later windows.
- `drift_report.json`: observed and synthetic PSI values and the alert decision.
- `validation_pr_curves.png`: validation precision-recall curves.
- `run_summary.md`: a readable summary generated from the current run.
- `models/models.pkl`: fitted estimators and the training-fitted scaler.

The upload-enabled visual dashboard in `streamlit_app.py` displays the outputs
and lets you download the metrics CSV and drift JSON.

## Current benchmark run

Using the Kaggle CSV (284,807 transactions; 492 frauds), the current
chronological run produced validation/later PR-AUC of **0.7719 / 0.7436** for
class-weighted logistic regression and **0.0213 / 0.0394** for Isolation
Forest. The validation-selected logistic threshold yielded 85.3% recall on the
later window at an estimated cost of `$5,999.99` under the illustrative cost
assumptions. The later window triggered `RETRAIN_REVIEW` because seven features
crossed the configured PSI retrain bound; the supervised PR-AUC decline was
3.7%, below its 20% alert bound. See [reports/approach.md](reports/approach.md)
for the full results and caveats. Rerunning `python run_all.py` overwrites
these outputs with the current run.

Run `python run_all.py` again to rerun training and evaluation from the CSV.
Review `results/run_summary.md` and `results/drift_report.json`; results are
specific to the dataset version and configured cost/drift assumptions.

## Limitations

The Kaggle dataset is an anonymized, historical benchmark, not a live fintech
stream. Chronological evaluation is more realistic than a shuffled split but
cannot fully represent future fraud tactics or operational alert handling.
The linear classifier and Isolation Forest are baselines. PSI can identify
distribution change but does not prove model degradation or causal drift.
Threshold costs and alert bounds require domain calibration.
