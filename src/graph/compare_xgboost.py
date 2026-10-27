
"""
Compare XGBoost with the GNN using the same sampled chronological split.
Run from project root:
python src/graph/compare_xgboost.py --rows 500000 --sample-frac 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transactions.csv"
OUTPUT_PATH = ROOT / "reports" / "xgb_same_split_metrics.json"
SEED = 42

REQUIRED = ["step", "type", "amount", "nameDest", "isFraud"]


def load_sampled_data(max_rows, sample_frac):
    chunks = []

    for i, chunk in enumerate(
        pd.read_csv(DATA_PATH, usecols=REQUIRED, chunksize=250_000)
    ):
        sampled = (
            chunk.sample(frac=sample_frac, random_state=SEED + i)
            if sample_frac < 1
            else chunk
        )
        if not sampled.empty:
            chunks.append(sampled)

        print(f"Read source chunk {i + 1}...", flush=True)

    if not chunks:
        raise ValueError("No data was sampled. Check the CSV file.")

    df = pd.concat(chunks, ignore_index=True)

    for col in ["step", "amount", "isFraud"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED).copy()
    df = df[(df["amount"] >= 0) & df["isFraud"].isin([0, 1])].copy()
    df["step"] = df["step"].astype(int)
    df["isFraud"] = df["isFraud"].astype(int)
    df["type"] = df["type"].astype(str)
    df["nameDest"] = df["nameDest"].astype(str)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)

    return df.sort_values("step").reset_index(drop=True)


def chronological_split(df):
    counts = df.groupby("step").size().sort_index()
    steps = counts.index.to_numpy()
    cumulative = counts.cumsum().to_numpy()

    if len(steps) < 3:
        raise ValueError("At least three distinct steps are required.")

    train_target = len(df) * 0.70
    val_target = len(df) * 0.85

    train_idx = int(np.argmin(np.abs(cumulative - train_target)))
    train_idx = min(max(train_idx, 0), len(steps) - 3)

    candidates = np.arange(train_idx + 1, len(steps) - 1)
    val_idx = candidates[
        np.argmin(np.abs(cumulative[candidates] - val_target))
    ]

    train_end = steps[train_idx]
    val_end = steps[val_idx]

    train = df[df.step <= train_end].copy()
    val = df[(df.step > train_end) & (df.step <= val_end)].copy()
    test = df[df.step > val_end].copy()

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        fraud = int(part["isFraud"].sum())
        print(
            f"{name}: rows={len(part):,}, fraud={fraud:,}, "
            f"legitimate={len(part) - fraud:,}"
        )
        if part["isFraud"].nunique() < 2:
            raise ValueError(f"{name} contains only one class.")

    print(f"Step boundaries: train <= {train_end}, validation <= {val_end}")
    return train, val, test


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["step"] = df["step"].astype(float)
    X["amount"] = df["amount"].astype(float)
    X["logAmount"] = np.log1p(df["amount"].astype(float))
    X["type"] = df["type"].astype(str)
    X["isMerchantDest"] = df["nameDest"].str.startswith("M").astype(int)
    X["hourOfDay"] = (df["step"] % 24).astype(int)
    return X


def best_threshold(y_true, probabilities):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "rows": int(len(y_true)),
        "fraud_count": int(np.sum(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--sample-frac", type=float, default=0.2)
    args = parser.parse_args()

    if args.rows < 100 or not 0 < args.sample_frac <= 1:
        parser.error("--rows must be >= 100 and --sample-frac in (0, 1].")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading and sampling PaySim data...", flush=True)
    df = load_sampled_data(args.rows, args.sample_frac)
    print(f"Sample ready: {len(df):,} rows", flush=True)

    train, val, test = chronological_split(df)

    X_train = make_features(train)
    X_val = make_features(val)
    X_test = make_features(test)

    y_train = train["isFraud"].to_numpy()
    y_val = val["isFraud"].to_numpy()
    y_test = test["isFraud"].to_numpy()

    numeric_features = [
        "step", "amount", "logAmount", "isMerchantDest", "hourOfDay"
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
        ]
    )

    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=positive_weight,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    print("Training XGBoost...", flush=True)
    model.fit(X_train, y_train)

    print("Evaluating validation set...", flush=True)
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold, validation_f1 = best_threshold(y_val, val_probabilities)

    print(f"Validation-selected threshold: {threshold:.3f}")
    print(f"Validation F1: {validation_f1:.6f}")

    print("Evaluating held-out test set...", flush=True)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probabilities, threshold)

    results = {
        "model": "XGBoost",
        "seed": SEED,
        "sample_rows": int(len(df)),
        "sample_frac": args.sample_frac,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_positive_weight": positive_weight,
        "validation_best_f1": validation_f1,
        "test": metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== XGBoost test results ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
