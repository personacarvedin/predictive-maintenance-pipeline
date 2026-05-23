#!/usr/bin/env python3
"""
main.py — Machine Fault Detection Pipeline
==========================================
Orchestrates: Download → Load → Preprocess → Feature Engineering
             → Train → Evaluate → Predict
"""

import os
import sys
import time
import traceback

# ── Ensure project root is on the path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import PROCESSED_DATA_PATH


def banner(text: str) -> None:
    width = 62
    print(f"\n{'━'*width}")
    print(f"  {text}")
    print(f"{'━'*width}")


def run_pipeline(skip_download: bool = False, csv_path: str = None):
    t0 = time.time()
    print("\n🔧  Machine Fault Detection Pipeline  🔧")
    print("=" * 62)

    # ── Step 1: Download ───────────────────────────────────────────────────
    banner("STEP 1 / 6  —  Data Acquisition")
    if skip_download and csv_path:
        print(f"  Skipping download, using: {csv_path}")
    else:
        from src.download_data import download_dataset
        csv_path = download_dataset()

    # ── Step 2: Load ───────────────────────────────────────────────────────
    banner("STEP 2 / 6  —  Load & Explore Data")
    from src.data_loader import load_data, explore_data
    df_raw = load_data(csv_path)
    explore_data(df_raw)

    # ── Step 3: Preprocess ─────────────────────────────────────────────────
    banner("STEP 3 / 6  —  Preprocessing")
    from src.preprocessing import preprocess
    df_clean = preprocess(df_raw)

    # ── Step 4: Feature Engineering ────────────────────────────────────────
    banner("STEP 4 / 6  —  Feature Engineering")
    from src.feature_engineering import engineer_features
    X_train, X_test, y_train, y_test, feature_names = engineer_features(df_clean)

    # ── Step 5: Train ──────────────────────────────────────────────────────
    banner("STEP 5 / 6  —  Model Training")
    from src.model_training import train_models
    models = train_models(X_train, y_train)

    # ── Step 6: Evaluate ───────────────────────────────────────────────────
    banner("STEP 6 / 6  —  Evaluation & Visualizations")
    from src.model_evaluation import evaluate_models
    results = evaluate_models(models, X_test, y_test, feature_names)

    # ── Bonus: Demo predictions ────────────────────────────────────────────
    banner("Sample Predictions")
    from src.prediction import demo_predictions
    best_model_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"  Using best model: {best_model_name}")
    demo_predictions(models[best_model_name])

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print("✅  PIPELINE COMPLETE")
    print(f"   Total time: {elapsed:.1f}s")
    print("\n   Accuracy Summary:")
    for name, r in results.items():
        print(f"     {name:<25} {r['accuracy']*100:.2f}%")
    print("\n   Outputs:")
    print(f"     Processed data   → data/processed/processed_data.csv")
    print(f"     Models           → models/")
    print(f"     Visualizations   → visualizations/")
    print(f"     Metrics          → results/model_metrics.txt")
    print(f"     Predictions      → results/predictions_sample.csv")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Machine Fault Detection Pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to existing CSV (skips Kaggle download)")
    args = parser.parse_args()

    try:
        run_pipeline(
            skip_download=bool(args.csv),
            csv_path=args.csv,
        )
    except Exception as e:
        print(f"\n❌  Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)