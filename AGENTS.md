
# Project Rules

## Project
Fraud and anomaly detection with drift monitoring using Kaggle's
creditcard.csv dataset. The target column is Class, where 1 means fraud.

## Reproducibility
- Use RANDOM_SEED = 42 from src/config.py.
- Sort by Time and split chronologically: 60% train, 20% validation,
  and 20% later.
- Never use random train/validation/test splits.
- Use fixed random seeds for stochastic models and simulations.

## Preventing Data Leakage
- Fit preprocessing on training data only.
- Apply resampling such as SMOTE to training data only.
- Select thresholds and tune models using training/validation data only.
- Never fit, tune, or select thresholds using the later window.
- Keep original transaction Amount available for cost analysis.

## Evaluation
- Primary metric: PR-AUC using sklearn's average_precision_score.
- ROC-AUC is a secondary comparison metric.
- Do not use accuracy as the headline metric.
- Select the operating threshold on validation and keep it unchanged
  when evaluating the later window.

## Code Organization
- Python modules belong in src/.
- Notebooks belong in notebooks/.
- Charts, metrics, and reports belong in results/.
- Automated tests belong in tests/.

## Code Quality
- Write beginner-friendly Python with descriptive names and docstrings.
- Explain why important decisions are made.
- Keep the project CPU-friendly.
- Train the autoencoder in a few minutes on a laptop where practical.

## Honest Reporting
- Run code and report actual outputs.
- Never invent metrics or claim an experiment succeeded without
  checking its output.
- Label artificially generated drift as synthetic.
- Do not overstate conclusions or hide limitations.

## Git
- Never commit changes automatically. The project owner reviews
  and commits changes manually.