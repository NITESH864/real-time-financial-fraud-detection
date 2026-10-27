
from pathlib import Path
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, encoding="utf-8") as f:
    metrics = json.load(f)

THRESHOLD = float(metrics["threshold_selected_on_validation"])

app = FastAPI(
    title="PaySim Fraud Detection API",
    description="API for scoring financial transactions for fraud risk.",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    step: int = Field(ge=0)
    amount: float = Field(ge=0)
    type: str
    nameDest: str = Field(min_length=1)

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("Amount must be finite")
        return value

    @field_validator("type")
    @classmethod
    def validate_type(cls, value):
        allowed = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
        if value not in allowed:
            raise ValueError(f"Type must be one of {sorted(allowed)}")
        return value


@app.get("/")
def home():
    return {
        "message": "PaySim Fraud Detection API",
        "docs": "/docs",
        "score_endpoint": "/score",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/score")
def score_transaction(transaction: TransactionRequest):
    features = pd.DataFrame([{
        "step": transaction.step,
        "amount": transaction.amount,
        "logAmount": math.log1p(transaction.amount),
        "type": transaction.type,
        "isMerchantDest": int(transaction.nameDest.startswith("M")),
        "hourOfDay": transaction.step % 24,
    }])

    probability = float(model.predict_proba(features)[0, 1])
    flagged = probability >= THRESHOLD

    return {
        "fraud_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "flagged": flagged,
        "risk_level": "high" if flagged else "not_flagged",
        "model_version": "xgboost-baseline-v1",
    }


from pathlib import Path
import json

# Add this endpoint inside your existing FastAPI application.
@app.get("/monitoring/status")
def monitoring_status():
    project_root = Path(__file__).resolve().parents[2]
    status_path = project_root / "reports" / "drift_status.json"

    if not status_path.exists():
        return {
            "status": "NOT_AVAILABLE",
            "message": "Run the data drift monitoring scripts first."
        }

    with status_path.open("r", encoding="utf-8") as file:
        return json.load(file)
