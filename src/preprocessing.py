# src/preprocessing.py - Data cleaning and preparation

import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, DROP_COLUMNS, CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS, PROCESSED_DATA_PATH, PROCESSED_DATA_DIR,
)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Drop irrelevant columns
      2. Remove duplicates
      3. Handle missing values
      4. Encode categorical features
      5. Scale numerical features
      6. Save processed CSV
    """
    print("[preprocess] Starting preprocessing …")
    df = df.copy()

    # ── 1. Drop irrelevant columns ────────────────────────────────────────────
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped columns: {cols_to_drop}")

    # ── 2. Remove duplicates ──────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Removed {before - len(df)} duplicate rows")

    # ── 3. Handle missing values ──────────────────────────────────────────────
    missing = df.isnull().sum().sum()
    if missing:
        df.dropna(inplace=True)
        print(f"  Dropped rows with missing values (was {missing} cells)")
    else:
        print("  No missing values found ✓")

    # ── 4. Encode categorical features ───────────────────────────────────────
    le = LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            print(f"  Encoded '{col}': {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # ── 5. Scale numerical features ───────────────────────────────────────────
    num_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"  Scaled columns: {num_cols}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Saved → {PROCESSED_DATA_PATH}")
    print(f"[preprocess] Done. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    from data_loader import load_data, explore_data
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/ai4i2020.csv"
    df_raw = load_data(raw_path)
    explore_data(df_raw)
    preprocess(df_raw)