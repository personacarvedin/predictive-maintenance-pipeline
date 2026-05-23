# src/model_evaluation.py - Evaluate models and produce visualizations

import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VISUALIZATIONS_DIR, RESULTS_DIR, METRICS_PATH,
    FIGURE_DPI,
)


def evaluate_models(models: dict, X_test, y_test, feature_names=None) -> dict:
    """
    Evaluate each model and generate:
      • Confusion matrix plots
      • Accuracy comparison bar chart
      • Feature importance chart (Random Forest)
      • metrics text file

    Returns
    -------
    dict  {model_name: {'accuracy': float, 'report': str}}
    """
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    lines = []

    for name, model in models.items():
        print(f"\n[eval] Evaluating: {name}")
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Normal", "Faulty"])
        cm     = confusion_matrix(y_test, y_pred)

        results[name] = {"accuracy": acc, "report": report, "cm": cm}

        # ── Console output ────────────────────────────────────────────────────
        print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"\n{report}")

        # ── Confusion Matrix ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Faulty"],
            yticklabels=["Normal", "Faulty"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        slug = name.lower().replace(" ", "_")
        cm_path = os.path.join(VISUALIZATIONS_DIR, f"confusion_matrix_{slug}.png")
        fig.savefig(cm_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"  Confusion matrix saved → {cm_path}")

        # ── Collect metrics text ──────────────────────────────────────────────
        lines += [
            f"{'='*60}",
            f"Model : {name}",
            f"Accuracy : {acc:.4f}",
            "",
            report,
        ]

    # ── Accuracy comparison bar chart ─────────────────────────────────────────
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=colors, width=0.45, edgecolor="white")
    ax.bar_label(bars, labels=[f"{a*100:.2f}%" for a in accs], padding=4, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    cmp_path = os.path.join(VISUALIZATIONS_DIR, "accuracy_comparison.png")
    fig.savefig(cmp_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"\n[eval] Accuracy comparison saved → {cmp_path}")

    # ── Feature importance (Random Forest) ────────────────────────────────────
    if "Random Forest" in models and feature_names is not None:
        rf = models["Random Forest"]
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_imp      = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(sorted_features[::-1], sorted_imp[::-1], color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest — Feature Importances")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fi_path = os.path.join(VISUALIZATIONS_DIR, "feature_importance.png")
        fig.savefig(fi_path, dpi=FIGURE_DPI)
        plt.close(fig)
        print(f"[eval] Feature importance saved → {fi_path}")

    # ── Save metrics text ─────────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[eval] Metrics saved → {METRICS_PATH}")

    return results


if __name__ == "__main__":
    import joblib, pandas as pd
    from config import RF_MODEL_PATH, LR_MODEL_PATH, PROCESSED_DATA_PATH
    from feature_engineering import engineer_features

    df = pd.read_csv(PROCESSED_DATA_PATH)
    _, X_test, _, y_test, feature_names = engineer_features(df)
    models = {
        "Random Forest"      : joblib.load(RF_MODEL_PATH),
        "Logistic Regression": joblib.load(LR_MODEL_PATH),
    }
    evaluate_models(models, X_test, y_test, feature_names)