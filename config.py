# config.py - Configuration Settings

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR            = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR  = os.path.join(DATA_DIR, "processed")
MODELS_DIR          = os.path.join(BASE_DIR, "models")
VISUALIZATIONS_DIR  = os.path.join(BASE_DIR, "visualizations")
RESULTS_DIR         = os.path.join(BASE_DIR, "results")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
RF_MODEL_PATH       = os.path.join(MODELS_DIR, "random_forest_model.pkl")
LR_MODEL_PATH       = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
METRICS_PATH        = os.path.join(RESULTS_DIR, "model_metrics.txt")
PREDICTIONS_PATH    = os.path.join(RESULTS_DIR, "predictions_sample.csv")

# ─── Dataset ──────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "stephanmatzka/predictive-maintenance-dataset-ai4i-2020"
TARGET_COLUMN  = "Machine failure"

# ─── Features ────────────────────────────────────────────────────────────────
DROP_COLUMNS = ["UDI", "Product ID",
                "TWF", "HDF", "PWF", "OSF", "RNF"]   # failure sub-types & IDs

CATEGORICAL_COLUMNS = ["Type"]
NUMERICAL_COLUMNS   = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ─── Training ─────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = None

# Logistic Regression
LR_MAX_ITER = 1000

# ─── Misc ─────────────────────────────────────────────────────────────────────
FIGURE_DPI = 150