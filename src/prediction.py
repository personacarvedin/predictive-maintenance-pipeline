# src/prediction.py - Make predictions on new machine readings

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RF_MODEL_PATH, PREDICTIONS_PATH, RESULTS_DIR


LABEL_MAP = {0: "Normal ✅", 1: "FAULTY ⚠️"}


def predict(model, sample: dict) -> dict:
    """
    Predict failure for a single machine reading.

    Parameters
    ----------
    model  : trained sklearn estimator
    sample : dict with feature values (already scaled / encoded)

    Returns
    -------
    dict  {'prediction': int, 'label': str, 'confidence': float}
    """
    X = pd.DataFrame([sample])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf  = proba[pred]
    return {
        "prediction": int(pred),
        "label"     : LABEL_MAP[int(pred)],
        "confidence": float(conf),
        "proba_normal": float(proba[0]),
        "proba_faulty": float(proba[1]),
    }


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on a DataFrame and return it with result columns added."""
    preds  = model.predict(df)
    probas = model.predict_proba(df)
    df = df.copy()
    df["prediction"]   = preds
    df["label"]        = [LABEL_MAP[p] for p in preds]
    df["prob_normal"]  = probas[:, 0]
    df["prob_faulty"]  = probas[:, 1]
    return df


def demo_predictions(model):
    """Run a handful of synthetic samples for demonstration."""
    print("\n[predict] Demo predictions on synthetic samples:")
    print("─" * 55)

    # Synthetic samples (values are already standardised-ish for demo)
    samples = [
        {"Type": 0, "Air temperature [K]": -0.5, "Process temperature [K]": -0.4,
         "Rotational speed [rpm]": 0.1, "Torque [Nm]": -0.2, "Tool wear [min]": -1.0,
         "temperature_diff": 0.1, "power_proxy": -0.02},

        {"Type": 1, "Air temperature [K]": 1.8, "Process temperature [K]": 2.1,
         "Rotational speed [rpm]": -1.5, "Torque [Nm]": 2.5, "Tool wear [min]": 2.8,
         "temperature_diff": 0.3, "power_proxy": -3.75},

        {"Type": 2, "Air temperature [K]": 0.0, "Process temperature [K]": 0.1,
         "Rotational speed [rpm]": 0.5, "Torque [Nm]": 0.3, "Tool wear [min]": 0.5,
         "temperature_diff": 0.1, "power_proxy": 0.15},
    ]

    rows = []
    for i, s in enumerate(samples, 1):
        r = predict(model, s)
        print(f"  Sample {i}: {r['label']}  (confidence {r['confidence']*100:.1f}%)")
        rows.append({**s, **r})

    # Save sample predictions
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n[predict] Saved → {PREDICTIONS_PATH}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else RF_MODEL_PATH
    print(f"[predict] Loading model: {model_path}")
    model = joblib.load(model_path)
    demo_predictions(model)