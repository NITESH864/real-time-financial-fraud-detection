from pathlib import Path
import json
import math

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Model metrics not found: {METRICS_PATH}"
    )

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, encoding="utf-8") as file:
    metrics = json.load(file)

# Use the threshold selected on validation data.
DEFAULT_THRESHOLD = float(
    metrics["threshold_selected_on_validation"]
)

app = FastAPI(
    title="PaySim Fraud Detection API",
    description="Scores PaySim transactions for fraud risk.",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    step: int = Field(ge=0, description="PaySim simulation step")
    amount: float = Field(ge=0, description="Transaction amount")
    type: str = Field(
        description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER"
    )
    nameDest: str = Field(
        min_length=1,
        description="Destination account identifier",
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Amount must be finite")
        return value

    @field_validator("type")
    @classmethod
    def transaction_type_must_be_valid(cls, value: str) -> str:
        allowed = {
            "CASH_IN", "CASH_OUT", "DEBIT",
            "PAYMENT", "TRANSFER",
        }
        if value not in allowed:
            raise ValueError(
                f"type must be one of {sorted(allowed)}"
            )
        return value


class FraudScoreResponse(BaseModel):
    fraud_probability: float
    threshold: float
    flagged: bool
    risk_level: str
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/")
def root():
    return {
        "message": "PaySim Fraud Detection API",
        "docs": "/docs",
        "score_endpoint": "/score",
    }


@app.post("/score", response_model=FraudScoreResponse)
def score_transaction(transaction: TransactionRequest):
    # Match the six features and their names used in training.
    features = pd.DataFrame([{
        "step": transaction.step,
        "amount": transaction.amount,
        "logAmount": math.log1p(transaction.amount),
        "type": transaction.type,
        "isMerchantDest": int(
            transaction.nameDest.startswith("M")
        ),
        "hourOfDay": transaction.step % 24,
    }])

    probability = float(
        model.predict_proba(features)[0, 1]
    )

    flagged = probability >= DEFAULT_THRESHOLD

    return FraudScoreResponse(
        fraud_probability=round(probability, 6),
        threshold=DEFAULT_THRESHOLD,
        flagged=flagged,
        risk_level="high" if flagged else "not_flagged",
        model_version="xgboost-baseline-v1",
    )
