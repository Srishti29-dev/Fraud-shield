# FraudShield — ML-Powered Credit Card Fraud Detection System

A machine learning system that detects fraudulent credit card transactions
in real time. Trains and compares two classifiers (Logistic Regression
baseline vs. Random Forest) on an imbalanced transaction dataset, evaluates
them with metrics suited to fraud detection (Precision, Recall, F1, ROC-AUC),
and exposes a CLI to score new transactions — one at a time or in bulk.

## Why This Project
Fraud detection is a classic **imbalanced classification** problem — fraud
is rare (~2-3% of transactions) but costly to miss. This project demonstrates
handling class imbalance correctly (`class_weight="balanced"`, stratified
splits) and evaluating with metrics that actually matter for fraud (Recall
and ROC-AUC), not just raw accuracy, which is misleading on imbalanced data.

## Features
- **Synthetic data generator** — creates a realistic, imbalanced transaction
  dataset (~2.5% fraud rate) with behavioral features like distance from
  home, transaction amount ratio, chip/PIN usage, and time-of-day patterns.
- **Two-model comparison** — Logistic Regression (interpretable baseline)
  vs. Random Forest (higher-performing primary model).
- **Fraud-appropriate evaluation** — Precision, Recall, F1, ROC-AUC, and a
  full confusion matrix (accuracy alone is misleading on imbalanced data).
- **Feature importance** — shows which signals the model relies on most.
- **CLI prediction** — score a single transaction or an entire CSV batch
  using the trained model.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn (Random Forest, Logistic Regression,
StandardScaler), Joblib

## Setup
```bash
git clone https://github.com/<your-username>/fraudshield.git
cd fraudshield
pip install -r requirements.txt
```

## Usage

**1. Train the model** (generates data + trains + saves model):
```bash
python fraud_detector.py --train
```

**2. Score a single transaction:**
```bash
python fraud_detector.py --predict --amount 850 --hour 2 --distance_home 120 \
  --distance_last 95 --ratio_median 6.5 --repeat_retailer 0 --used_chip 0 \
  --used_pin 0 --online_order 1
```

**3. Score a batch of transactions from a CSV:**
```bash
python fraud_detector.py --predict_file sample_batch.csv
```

## Screenshot

![Training output](screenshots/training_output.png)

## Sample Results
