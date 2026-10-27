
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_score_transaction():
    payload = {
        "step": 278,
        "amount": 1500.0,
        "type": "PAYMENT",
        "nameDest": "C123456789",
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.0 <= result["threshold"] <= 1.0
    assert isinstance(result["flagged"], bool)
    assert result["risk_level"] in {"high", "not_flagged"}


def test_invalid_transaction_type():
    payload = {
        "step": 278,
        "amount": 1500.0,
        "type": "UNKNOWN",
        "nameDest": "C123456789",
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_negative_amount():
    payload = {
        "step": 278,
        "amount": -10,
        "type": "PAYMENT",
        "nameDest": "C123456789",
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 422
