from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

# Read only fields needed for our initial model.
columns = [
    "step", "type", "amount",
    "nameDest", "isFraud"
]
df = pd.read_csv(DATA_PATH, usecols=columns)

# Basic validation.
if df[columns].isnull().any().any():
    raise ValueError("Missing values found; inspect the dataset.")

if not set(df["isFraud"].unique()).issubset({0, 1}):
    raise ValueError("isFraud must contain binary labels 0 and 1.")

# Feature engineering.
df["isMerchantDest"] = (
    df["nameDest"].str.startswith("M").astype("int8")
)
df["hourOfDay"] = (df["step"] % 24).astype("int8")
df["logAmount"] = np.log1p(df["amount"].clip(lower=0))

feature_columns = [
    "step", "amount", "logAmount",
    "type", "isMerchantDest", "hourOfDay"
]
target = "isFraud"

# Chronological split.
train_df = df[df["step"] <= 520].copy()
valid_df = df[
    (df["step"] > 520) & (df["step"] <= 630)
].copy()
test_df = df[df["step"] > 630].copy()

if min(len(train_df), len(valid_df), len(test_df)) == 0:
    raise ValueError("A time split is empty; check step values.")

X_train, y_train = train_df[feature_columns], train_df[target]
X_valid, y_valid = valid_df[feature_columns], valid_df[target]
X_test, y_test = test_df[feature_columns], test_df[target]

print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))
print("Test rows:", len(test_df))
print("Training fraud count:", int(y_train.sum()))

if y_train.sum() == 0 or y_valid.sum() == 0 or y_test.sum() == 0:
    raise ValueError(
        "A split has no fraud cases. Adjust time boundaries "
        "after inspecting fraud counts by step."
    )

categorical_features = ["type"]
numeric_features = [
    "step", "amount", "logAmount",
    "isMerchantDest", "hourOfDay"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "type",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

scale_pos_weight = (
    (len(y_train) - int(y_train.sum()))
    / int(y_train.sum())
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

print("\nTraining XGBoost...")
pipeline.fit(X_train, y_train)

# Choose a decision threshold using validation data only.
valid_scores = pipeline.predict_proba(X_valid)[:, 1]
thresholds = np.arange(0.05, 0.96, 0.05)

threshold_results = []
for threshold in thresholds:
    preds = (valid_scores >= threshold).astype(int)
    threshold_results.append(
        (
            f1_score(y_valid, preds, zero_division=0),
            threshold,
        )
    )

best_f1, best_threshold = max(threshold_results)

# Evaluate once on the later test period.
test_scores = pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_scores >= best_threshold).astype(int)

results = {
    "threshold_selected_on_validation": float(best_threshold),
    "validation_best_f1": float(best_f1),
    "test_average_precision_pr_auc": float(
        average_precision_score(y_test, test_scores)
    ),
    "test_precision": float(
        precision_score(y_test, test_preds, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, test_preds, zero_division=0)
    ),
    "test_f1": float(
        f1_score(y_test, test_preds, zero_division=0)
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).tolist(),
    "test_rows": int(len(y_test)),
    "test_fraud_count": int(y_test.sum()),
    "train_scale_pos_weight": float(scale_pos_weight),
}

joblib.dump(pipeline, MODEL_DIR / "fraud_model.joblib")

with open(REPORT_DIR / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nTest results:")
print(json.dumps(results, indent=2))
print("\nSaved model to models/fraud_model.joblib")
print("Saved metrics to reports/model_metrics.json")
