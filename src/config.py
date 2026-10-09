
"""Central configuration for the fraud detection project."""

from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# Dataset
DATA_PATH = DATA_DIR / "creditcard.csv"
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"

# Reproducibility
RANDOM_SEED = 42

# Chronological data split ratios
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
LATER_RATIO = 0.20

# Data checks
MIN_FRAUD_CASES_PER_SPLIT = 30

# Illustrative business cost assumptions.
# These will be reviewed during the cost-analysis phase.
CHARGEBACK_FEE = 15.0
FALSE_POSITIVE_FIXED_COST = 5.0
FALSE_POSITIVE_AMOUNT_RATE = 0.01
FALSE_NEGATIVE_COST_MULTIPLIER = 1.0

# Initial drift-monitoring thresholds.
# These are starting assumptions, not validated business rules.
FEATURE_PSI_WARNING = 0.10
FEATURE_PSI_RETRAIN = 0.25
SCORE_PSI_WARNING = 0.10
SCORE_PSI_RETRAIN = 0.25
PR_AUC_DROP_RETRAIN = 0.20

# Create output directories when the project is run.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
