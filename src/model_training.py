# src/model_training.py - Train Random Forest and Logistic Regression models

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, RF_MODEL_PATH, LR_MODEL_PATH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, LR_MAX_ITER, RANDOM_STATE,
)


def train_models(X_train, y_train):
    """
    Train Random Forest and Logistic Regression classifiers.

    Returns
    -------
    dict  {name: trained_model}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("[train] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_MODEL_PATH)
    print(f"  Saved → {RF_MODEL_PATH}")
    models["Random Forest"] = rf

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("[train] Training Logistic Regression …")
    lr = LogisticRegression(
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, LR_MODEL_PATH)
    print(f"  Saved → {LR_MODEL_PATH}")
    models["Logistic Regression"] = lr

    print("[train] All models trained and saved.")
    return models


if __name__ == "__main__":
    from config import PROCESSED_DATA_PATH
    from feature_engineering import engineer_features

    df = pd.read_csv(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test, _ = engineer_features(df)
    train_models(X_train, y_train)