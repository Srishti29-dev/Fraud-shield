"""
FraudShield — ML-Powered Credit Card Fraud Detection System
--------------------------------------------------------------
Generates a realistic (synthetic) imbalanced transaction dataset, trains
and compares two classifiers (Logistic Regression baseline vs. Random
Forest), evaluates them with metrics suited to imbalanced fraud data
(Precision, Recall, F1, ROC-AUC), and exposes a CLI to score new
transactions with a saved model.

Author: Srishti Sinha
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model/fraud_model.joblib"
SCALER_PATH = "model/scaler.joblib"
DATA_PATH = "data/transactions.csv"

FEATURES = [
    "amount", "hour_of_day", "distance_from_home", "distance_from_last_txn",
    "ratio_to_median_price", "repeat_retailer", "used_chip", "used_pin", "online_order",
]


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------------

def generate_transactions(n: int = 20000, fraud_rate: float = 0.025, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic, imbalanced synthetic transaction dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_block(n_rows, fraud: bool):
        if fraud:
            amount = rng.gamma(shape=2.2, scale=180, size=n_rows)
            distance_from_home = rng.exponential(scale=90, size=n_rows)
            distance_from_last = rng.exponential(scale=70, size=n_rows)
            ratio_to_median = rng.gamma(shape=2.0, scale=3.5, size=n_rows)
            repeat_retailer = rng.binomial(1, 0.25, size=n_rows)
            used_chip = rng.binomial(1, 0.15, size=n_rows)
            used_pin = rng.binomial(1, 0.10, size=n_rows)
            online_order = rng.binomial(1, 0.80, size=n_rows)
            hour = rng.choice(range(24), size=n_rows, p=_night_weighted_hours())
        else:
            amount = rng.gamma(shape=2.0, scale=45, size=n_rows)
            distance_from_home = rng.exponential(scale=8, size=n_rows)
            distance_from_last = rng.exponential(scale=6, size=n_rows)
            ratio_to_median = rng.gamma(shape=2.0, scale=0.6, size=n_rows)
            repeat_retailer = rng.binomial(1, 0.75, size=n_rows)
            used_chip = rng.binomial(1, 0.65, size=n_rows)
            used_pin = rng.binomial(1, 0.55, size=n_rows)
            online_order = rng.binomial(1, 0.35, size=n_rows)
            hour = rng.choice(range(24), size=n_rows, p=_day_weighted_hours())

        return pd.DataFrame({
            "amount": np.round(amount, 2),
            "hour_of_day": hour,
            "distance_from_home": np.round(distance_from_home, 2),
            "distance_from_last_txn": np.round(distance_from_last, 2),
            "ratio_to_median_price": np.round(ratio_to_median, 2),
            "repeat_retailer": repeat_retailer,
            "used_chip": used_chip,
            "used_pin": used_pin,
            "online_order": online_order,
            "is_fraud": int(fraud),
        })

    df = pd.concat([make_block(n_legit, False), make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df.insert(0, "transaction_id", [f"TXN{100000+i}" for i in range(len(df))])
    return df


def _night_weighted_hours():
    """Fraud is more likely late at night / early morning."""
    weights = np.array([6 if (h >= 0 and h <= 5) or h >= 22 else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


def _day_weighted_hours():
    """Legitimate transactions cluster around daytime hours."""
    weights = np.array([4 if 9 <= h <= 21 else 1 for h in range(24)], dtype=float)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# 2. TRAINING
# ---------------------------------------------------------------------------

def train(n_rows: int = 20000):
    print("Generating synthetic transaction dataset...")
    df = generate_transactions(n=n_rows)
    Path("data").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Dataset generated: {len(df):,} transactions ({fraud_count:,} fraudulent, "
          f"{fraud_count/len(df)*100:.2f}%)\n")

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 58)
    print(" MODEL 1: Logistic Regression (baseline)")
    print("=" * 58)
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")

    print("\n" + "=" * 58)
    print(" MODEL 2: Random Forest (primary model)")
    print("=" * 58)
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    evaluate_model(rf, X_test, y_test, "Random Forest")

    print("\nFeature Importance (Random Forest):")
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 60)
        print(f"  {feat:<26} {imp:.3f}  {bar}")

    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")
    print("Training complete.\n")


def evaluate_model(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} — Performance on held-out test set:")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  ROC-AUC   : {auc:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Legit   Predicted Fraud")
    print(f"  Actual Legit  {cm[0][0]:>15,}   {cm[0][1]:>15,}")
    print(f"  Actual Fraud  {cm[1][0]:>15,}   {cm[1][1]:>15,}")


# ---------------------------------------------------------------------------
# 3. PREDICTION
# ---------------------------------------------------------------------------

def predict_single(args):
    if not Path(MODEL_PATH).exists():
        print("No trained model found. Run with --train first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "amount": args.amount,
        "hour_of_day": args.hour,
        "distance_from_home": args.distance_home,
        "distance_from_last_txn": args.distance_last,
        "ratio_to_median_price": args.ratio_median,
        "repeat_retailer": args.repeat_retailer,
        "used_chip": args.used_chip,
        "used_pin": args.used_pin,
        "online_order": args.online_order,
    }])

    proba = model.predict_proba(row[FEATURES])[0][1]
    pred = "FRAUD" if proba >= 0.5 else "LEGITIMATE"

    print("\n" + "=" * 50)
    print(" FRAUDSHIELD — TRANSACTION RISK ASSESSMENT")
    print("=" * 50)
    print(f"  Amount               : \u20B9{args.amount:,.2f}")
    print(f"  Hour of day          : {args.hour}:00")
    print(f"  Distance from home   : {args.distance_home} km")
    print(f"  Distance from last   : {args.distance_last} km")
    print(f"  Ratio to median price: {args.ratio_median}x")
    print("-" * 50)
    print(f"  Fraud Probability    : {proba*100:.1f}%")
    print(f"  Prediction           : {pred}")
    print("=" * 50 + "\n")


def predict_batch(csv_path: str):
    if not Path(MODEL_PATH).exists():
        print("No trained model found. Run with --train first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_path)
    proba = model.predict_proba(df[FEATURES])[:, 1]
    df["fraud_probability"] = np.round(proba, 4)
    df["prediction"] = np.where(proba >= 0.5, "FRAUD", "LEGITIMATE")
    out_path = "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nScored {len(df)} transactions. Results saved to '{out_path}'.")
    print(df[["transaction_id", "amount", "fraud_probability", "prediction"]].to_string(index=False))


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FraudShield — Credit Card Fraud Detection")
    parser.add_argument("--train", action="store_true", help="Generate data and train models")
    parser.add_argument("--rows", type=int, default=20000, help="Number of synthetic transactions to generate")

    parser.add_argument("--predict", action="store_true", help="Predict a single transaction")
    parser.add_argument("--amount", type=float, default=250.0)
    parser.add_argument("--hour", type=int, default=14)
    parser.add_argument("--distance_home", type=float, default=5.0)
    parser.add_argument("--distance_last", type=float, default=3.0)
    parser.add_argument("--ratio_median", type=float, default=1.2)
    parser.add_argument("--repeat_retailer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--used_chip", type=int, choices=[0, 1], default=1)
    parser.add_argument("--used_pin", type=int, choices=[0, 1], default=1)
    parser.add_argument("--online_order", type=int, choices=[0, 1], default=0)

    parser.add_argument("--predict_file", type=str, help="CSV of transactions to batch score")

    args = parser.parse_args()

    if args.train:
        train(args.rows)
    elif args.predict:
        predict_single(args)
    elif args.predict_file:
        predict_batch(args.predict_file)
    else:
        print("Use --train to train the model, --predict for a single transaction, "
              "or --predict_file <csv> for batch scoring.")


if __name__ == "__main__":
    main()
