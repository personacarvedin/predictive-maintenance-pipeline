# src/feature_engineering.py - Feature selection, engineering, and train/test split

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def engineer_features(df: pd.DataFrame):
    """
    1. Create derived features
    2. Separate X / y
    3. Train / test split

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    print("[features] Engineering features …")
    df = df.copy()

    # ── Derived features ──────────────────────────────────────────────────────
    air_col     = "Air temperature [K]"
    process_col = "Process temperature [K]"
    if air_col in df.columns and process_col in df.columns:
        df["temperature_diff"] = df[process_col] - df[air_col]
        print("  Created 'temperature_diff' feature")

    torque_col = "Torque [Nm]"
    speed_col  = "Rotational speed [rpm]"
    if torque_col in df.columns and speed_col in df.columns:
        df["power_proxy"] = df[torque_col] * df[speed_col]
        print("  Created 'power_proxy' feature")

    # ── Split X / y ───────────────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Target  : '{TARGET_COLUMN}'  |  classes: {sorted(y.unique())}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print("[features] Done.")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    from config import PROCESSED_DATA_PATH
    df = pd.read_csv(PROCESSED_DATA_PATH)
    engineer_features(df)