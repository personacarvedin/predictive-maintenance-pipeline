# src/download_data.py - Download dataset from Kaggle

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAGGLE_DATASET


def download_dataset() -> str:
    """
    Download the AI4I 2020 Predictive Maintenance dataset via kagglehub.

    Returns
    -------
    str
        Path to the downloaded CSV file.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub not installed. Run: pip install kagglehub")

    print(f"[download] Downloading dataset: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"[download] Dataset saved to: {path}")

    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    csv_path = csv_files[0]
    print(f"[download] Using CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    csv_path = download_dataset()
    print(f"\nReady to use: {csv_path}")