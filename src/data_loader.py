# src/data_loader.py - Load and explore the dataset

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMN


def load_data(csv_path: str) -> pd.DataFrame:
    """Read CSV and return a DataFrame."""
    print(f"[loader] Reading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[loader] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print a concise exploration summary."""
    sep = "─" * 60

    print(f"\n{sep}")
    print("DATASET OVERVIEW")
    print(sep)
    print(f"Shape : {df.shape}")
    print(f"Columns:\n  {list(df.columns)}\n")

    print("Data Types:")
    print(df.dtypes.to_string())

    print(f"\n{sep}")
    print("FIRST 5 ROWS")
    print(sep)
    print(df.head().to_string())

    print(f"\n{sep}")
    print("DESCRIPTIVE STATISTICS")
    print(sep)
    print(df.describe().to_string())

    print(f"\n{sep}")
    print("MISSING VALUES")
    print(sep)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values ✓")
    else:
        print(missing[missing > 0].to_string())

    if TARGET_COLUMN in df.columns:
        print(f"\n{sep}")
        print("CLASS DISTRIBUTION  (target: '{TARGET_COLUMN}')")
        print(sep)
        counts = df[TARGET_COLUMN].value_counts()
        for label, n in counts.items():
            bar = "█" * int(n / counts.max() * 30)
            print(f"  {label:>3}  {bar}  {n:,}  ({n/len(df)*100:.1f}%)")

    print(f"{sep}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ai4i2020.csv"
    df = load_data(path)
    explore_data(df)