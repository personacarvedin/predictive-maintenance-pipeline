# 🔧 Machine Fault Detection & Predictive Maintenance Pipeline

## Overview
This is a Python-based production machine learning pipeline designed to predict industrial equipment failures before they happen. In manufacturing and industrial IoT environments, waiting for a machine to break down leads to costly unplanned downtime, expensive secondary damage, and operational bottlenecks. This project addresses the challenge by continuously processing high-frequency sensor streams—such as temperatures, rotational speeds, and torque—to accurately classify equipment health, catch anomalous operational signatures, and flag imminent failures.

The system is engineered as a highly organized, modular, and production-ready codebase rather than a loose collection of experimental notebooks. It features automated data acquisition, a custom physics-driven feature engineering engine, rigorous handling of extreme real-world class imbalances, and automated multi-model benchmarking. Every execution generates serialized models, reproducible data states, complete performance metrics, and a suite of analytical plots.

---

## Why This Is Required
Industrial asset management historically relied on reactive maintenance (fixing things after they break) or scheduled maintenance (replacing parts based on a fixed timeline). Reactive maintenance is financially draining, while scheduled maintenance wastes functional components unnecessarily. Predictive maintenance solves this by identifying the exact window when a machine is exhibiting early micro-faults but has not yet catastrophically failed.

However, training machine learning models for industrial environments introduces severe engineering challenges that this project explicitly solves:
* **Severe Class Imbalance:** In a well-managed factory, machines operate normally 96% to 99% of the time. Critical breakdowns are rare events (~3.4% in this dataset). Standard machine learning algorithms trained on this data will simply predict "Normal" for everything and look 96.6% accurate while failing to catch a single actual breakdown.
* **Complex Feature Interactions:** Raw sensor boundaries (like absolute temperature or raw speed) don't always tell the full story. Breakdowns are often triggered by *relationships* between variables—such as a machine drawing massive torque while spinning at an uncharacteristically low speed. 
* **Target Leakage:** Raw datasets frequently include downstream diagnostic flags that indicate *why* a machine failed after the fact. Leaving these in during training creates a predictive system that looks perfect on paper but fails completely in the real world because it relies on information from the future.

This pipeline provides a clean, automated framework to safely navigate these pitfalls and deliver actionable, highly precise operational alerts.

---

## Goals and Objectives
* **Primary Goal:** Provide an enterprise-ready, reproducible ML pipeline that takes raw industrial sensor data and outputs high-confidence operational risk predictions.
* **Objectives:**
  * Build a clean, decoupled project architecture following standard `src/` modular design patterns.
  * Inject custom domain logic via physics-informed feature engineering to surface latent structural anomalies.
  * Enforce strict target leakage isolation by identifying and stripping retrospective diagnostic tags.
  * Guarantee reproducible evaluations on highly imbalanced target spaces using stratified splitting logic.
  * Train, optimize, and benchmark distinct model classes (tree-based ensembles vs. linear models) across robust operational metrics like Precision and Recall rather than vanilla accuracy.
  * Export high-fidelity visualizations and performance reports automatically on every pipeline run.

---

## What the Project Does
The pipeline coordinates a deterministic 6-step execution flow that processes raw operational parameters into production-ready predictive models.

```text
       [Raw Industrial CSV Data]
                   │
                   ▼
     STEP 1: Data Acquisition (Kaggle Ingestion)
                   │
                   ▼
     STEP 2: Load & Exploratory Validation
                   │
                   ▼
     STEP 3: Preprocessing (Encoding & Standard Scaling)
                   │
                   ▼
     STEP 4: Physics-Informed Feature Engineering
                   │
                   ▼
     STEP 5: Model Training (Random Forest vs. Logistic Regression)
                   │
                   ▼
     STEP 6: Imbalance-Aware Evaluation & Visualizations
                   │
                   ▼
      [Serialized Models, Reports & PNG Outputs]

```

### 1. Data Ingestion & Quality Exploration (Steps 1 & 2)

The pipeline automatically pulls the `ai4i2020` Predictive Maintenance Dataset directly from Kaggle, storing it within a localized directory cache. Upon loading, it runs an initial data validation pass, verifying a uniform shape of 10,000 rows across 14 distinct columns, mapping out raw explicit data types, and asserting the absolute absence of missing values.

### 2. Leakage Mitigation & Preprocessing (Step 3)

The data preprocessing engine prepares the dataset for mathematical modeling while strictly preventing target leakage. It drops data identifiers (`UDI`, `Product ID`) and, computationally, purges five specific failure mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`). Because these columns represent structural post-mortems (e.g., Tool Wear Failure or Heat Dissipation Failure), keeping them would allow models to cheat. Categorical variables like machine `Type` ('L', 'M', 'H') are encoded numerically, and all continuous physical sensors are systematically standard-scaled.

### 3. Physics-Informed Feature Engineering (Step 4)

Instead of forcing the model to infer basic laws of thermodynamics and mechanics, the feature engineering module explicitly creates two highly predictive derived features:

* `temperature_diff`: Calculates the raw mathematical delta between `Process temperature [K]` and `Air temperature [K]`. This acts as a direct proxy for thermal dissipation efficiency and heat stress.
* `power_proxy`: Multiplies `Torque [Nm]` by `Rotational speed [rpm]`. Since mechanical power is directly proportional to the product of torque and angular velocity, this single feature gives the model an elegant, unscaled look at the true workload strain under which the asset is operating.

Following feature creation, the module applies a strict train-test split (80/20). It enforces a `stratify=y` rule to guarantee that the tiny 3.4% failure class is mirrored exactly at identical ratios across both the training set (8,000 rows) and the testing set (2,000 rows).

### 4. Model Training & Benchmarking (Steps 5 & 6)

The pipeline handles model training asynchronously across two distinct algorithmic families:

* **Random Forest Classifier:** An ensemble of 100 decision trees constructed without depth limits, optimized to capture highly non-linear decision boundaries and complex feature interactions.
* **Logistic Regression:** A linear baseline model configured with a high iteration threshold (`max_iter=1000`) to test linear separability across the engineered feature space.

---

## Project Structure

Your codebase follows a production-grade, highly decoupled modular layout:

```text
.
├── config.py                  # Global configurations, feature maps, and hyperparameter tokens
├── data/
│   └── processed/
│       └── processed_data.csv # Final, scaled, and encoded feature matrix
├── main.py                    # Master pipeline orchestrator and execution entry-point
├── models/                    # Serialized production models (*.pkl)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt           # Explicit environment dependencies
├── results/
│   ├── model_metrics.txt      # Text-based classification reports and metadata summaries
│   └── predictions_sample.csv # In-run demonstration predictions on synthetic states
├── src/                       # Core functional source modules
│   ├── __init__.py
│   ├── data_loader.py         # Handles dataset loading and initial verification
│   ├── download_data.py       # Automated Kaggle data pull and caching layer
│   ├── feature_engineering.py # Physics-informed feature generation and stratified splitting
│   ├── model_evaluation.py    # Multi-model benchmarking and report generation
│   ├── model_training.py      # Model training logic and serialization routines
│   ├── prediction.py          # Post-train synthetic batch simulation and prediction testing
│   └── preprocessing.py       # Handles target leakage extraction and attribute scaling
├── tests/                     # Pipeline verification layer
│   ├── __init__.py
│   └── test_preprocessing.py  # Unit tests ensuring consistency in preprocessing transformations
└── visualizations/            # Exported metrics plots and model evaluations
    ├── accuracy_comparison.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_random_forest.png
    └── feature_importance.png

```

---

## The Data

### Source Dataset

The pipeline ingests the UCI **AI4I 2020 Predictive Maintenance Dataset**, an industry-standard benchmark reflecting real-world milling machine operations. It contains 10,000 independent observations measuring structural parameters:

* **Type:** Machine quality variant categorized as Low (L, 50% of data), Medium (M, 30%), or High (H, 20%).
* **Air Temperature [K]:** Ambient room temperature generated via a random walk process.
* **Process Temperature [K]:** Internal tool operational temperature.
* **Rotational Speed [rpm]:** Calculated spindle speed.
* **Torque [Nm]:** Torque values centered around 40 Nm with no negative values.
* **Tool Wear [min]:** The cumulative minutes of structural stress applied to the active tool piece.
* **Machine Failure:** The explicit binary target variable indicating whether the machine broke down during that specific operation window.

---

## Pipeline Execution Output

### Console Execution Log

When `main.py` runs, it executes all steps sequentially in a single automated process:

```text
🔧  Machine Fault Detection Pipeline  🔧
==============================================================

━━┃ STEP 1 / 6  —  Data Acquisition ┃━━━━━━━━━━━━━━━━━━━━━━━━━
[download] Downloading dataset: stephanmatzka/predictive-maintenance-dataset-ai4i-2020
[download] Dataset saved to: /home/codespace/.cache/kagglehub/datasets/...
[download] Using CSV: /home/codespace/.cache/kagglehub/datasets/.../ai4i2020.csv

━━┃ STEP 2 / 6  —  Load & Explore Data ┃━━━━━━━━━━━━━━━━━━━━━━
[loader] Reading: /home/codespace/.cache/kagglehub/datasets/.../ai4i2020.csv
[loader] Loaded 10,000 rows × 14 columns

DATASET OVERVIEW
--------------------------------------------------------------
Shape : (10000, 14)
Columns: ['UDI', 'Product ID', 'Type', 'Air temperature [K]', ...]
Data Types:
UDI                         int64
Product ID                    str
Type                          str
Air temperature [K]       float64
Process temperature [K]   float64
...
Machine failure             int64

FIRST 5 ROWS
--------------------------------------------------------------
   UDI Product ID Type  Air temperature [K]  Process temperature [K] ...
0    1     M14860    M                298.1                    308.6 ...
1    2     L47181    L                298.2                    308.7 ...

DESCRIPTIVE STATISTICS
--------------------------------------------------------------
[... Statistical distribution summary ...]

MISSING VALUES
--------------------------------------------------------------
 No missing values ✓

CLASS DISTRIBUTION (target: 'Machine failure')
--------------------------------------------------------------
  0 █ 9,661 (96.6%)
  1 █ 339 (3.4%)

━━┃ STEP 3 / 6  —  Preprocessing ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[preprocess] Starting preprocessing ...
Dropped columns: ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
Removed 0 duplicate rows
No missing values found ✓
Encoded 'Type': ['H', 'L', 'M'] → [0, 1, 2]
Scaled columns: ['Air temperature [K]', 'Process temperature [K]', ...]
Saved → /workspaces/codespaces-blank/data/processed/processed_data.csv
[preprocess] Done. Shape: (10000, 7)

━━┃ STEP 4 / 6  —  Feature Engineering ┃━━━━━━━━━━━━━━━━━━━━━━
[features] Engineering features ...
  Created 'temperature_diff' feature
  Created 'power_proxy' feature
  Features (8): ['Type', 'Air temperature [K]', ..., 'temperature_diff', 'power_proxy']
  Target  : 'Machine failure'  |  classes: [0, 1]
  Train: 8,000 rows  |  Test: 2,000 rows
[features] Done.

━━┃ STEP 5 / 6  —  Model Training ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━
[train] Training Random Forest ...
  Saved → /workspaces/codespaces-blank/models/random_forest_model.pkl
[train] Training Logistic Regression ...
  Saved → /workspaces/codespaces-blank/models/logistic_regression_model.pkl
[train] All models trained and saved.

━━┃ STEP 6 / 6  —  Evaluation & Visualizations ┃━━━━━━━━━━━━━
[eval] Evaluating: Random Forest
Accuracy : 0.9850  (98.50%)

               precision    recall  f1-score   support

      Normal       0.99      1.00      0.99      1932
      Faulty       0.93      0.60      0.73        68

    accuracy                           0.98      2000
   macro avg       0.96      0.80      0.86      2000
weighted avg       0.98      0.98      0.98      2000

Confusion matrix saved → /workspaces/codespaces-blank/visualizations/confusion_matrix_random_forest.png

[eval] Evaluating: Logistic Regression
Accuracy : 0.8615  (86.15%)

               precision    recall  f1-score   support

      Normal       1.00      0.86      0.92      1932
      Faulty       0.18      0.88      0.30        68

    accuracy                           0.86      2000
   macro avg       0.59      0.87      0.61      2000
weighted avg       0.97      0.86      0.90      2000

Confusion matrix saved → /workspaces/codespaces-blank/visualizations/confusion_matrix_logistic_regression.png
[eval] Accuracy comparison saved → /workspaces/codespaces-blank/visualizations/accuracy_comparison.png
[eval] Feature importance saved → /workspaces/codespaces-blank/visualizations/feature_importance.png
[eval] Metrics saved → /workspaces/codespaces-blank/results/model_metrics.txt

━━┃ Sample Predictions ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Using best model: Random Forest
[predict] Demo predictions on synthetic samples:
  Sample 1: Normal  ✅  (confidence 100.0%)
  Sample 2: FAULTY  ⚠️  (confidence 70.0%)
  Sample 3: Normal  ✅  (confidence 99.0%)
[predict] Saved → /workspaces/codespaces-blank/results/predictions_sample.csv

==============================================================
✅  PIPELINE COMPLETE
   Total time: 8.7s

   Accuracy Summary:
     Random Forest             98.50%
     Logistic Regression       86.15%

   Outputs:
     Processed data   → data/processed/processed_data.csv
     Models           → models/
     Visualizations   → visualizations/
     Metrics          → results/model_metrics.txt
     Predictions      → results/predictions_sample.csv
==============================================================

```

---

## Installation

Clone the repository and set up a local isolated virtual environment (mac and linux):

```bash
# Clone the repository
git clone [https://github.com/personacarvedin/predictive-maintenance-pipeline.git](https://github.com/personacarvedin/predictive-maintenance-pipeline.git)
cd predictive-maintenance-pipeline

# Establish an isolated virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all explicit project dependencies
pip install -r requirements.txt

```

---

## Running the Project

### Run the Default Automated Pipeline

To execute the complete operational flow—including automated remote data fetching, scaling, engineering, training, and plot exporting:

```bash
python main.py

```

### Run with Local Dataset Injection

If you want to run the model pipeline with an existing localized source file to bypass Kaggle integration entirely:

```bash
python main.py --csv data/raw/ai4i2020.csv

```

### Resetting and Regenerating Pipeline States

To strip out all locally generated model configurations, cached data rows, and analytical outputs to trigger a clean slate execution:

```bash
rm -rf data/processed/*.csv models/*.pkl visualizations/*.png results/*
python main.py

```

---

## Configuration Settings

All core analytical parameters, pipeline column lists, and execution hyperparameters are cleanly isolated within `config.py`. Modifying pipeline behavior requires zero logic alterations:

* **`DROP_COLUMNS`:** Array containing target leakage files to purge during training.
* **`TEST_SIZE = 0.20`:** Sets the explicit validation volume fraction.
* **`RANDOM_STATE = 42`:** Enforces exact algorithmic reproducibility across all initializations.
* **`RF_N_ESTIMATORS = 100`:** Sets tree density configurations for the Random Forest ensemble.
