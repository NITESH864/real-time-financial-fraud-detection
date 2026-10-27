# Real-Time Financial Fraud Detection Platform

An end-to-end machine learning project for detecting potentially fraudulent financial transactions using XGBoost, Kafka, FastAPI, MLflow, Feast, PyTorch Geometric, and Evidently AI.

## Overview

Financial fraud detection is challenging because fraudulent transactions are rare compared with legitimate transactions. This project explores machine learning and streaming technologies to score transactions, identify potentially suspicious activity, and monitor changes in transaction data.

The platform combines a supervised XGBoost baseline with a graph neural network (GNN) experiment, a REST API, a Kafka streaming pipeline, experiment tracking, feature-store experimentation, and data-drift reporting.

**Project status:** Working prototype and ongoing development. Not a production-ready fraud prevention system.

## Features

- **Exploratory Data Analysis:** Transaction distribution and fraud-pattern visualizations.
- **XGBoost:** Supervised fraud classification with an imbalanced dataset.
- **MLflow:** Experiment tracking for the baseline model.
- **Feast:** Local feature-store demonstration using historical transaction features.
- **FastAPI:** REST endpoints for health checks, transaction scoring, and monitoring status.
- **Apache Kafka:** Transaction producer and consumer pipeline for streaming scores.
- **PyTorch Geometric:** Graph neural network experiment for transaction fraud detection.
- **Evidently AI:** Historical feature-drift reporting with HTML and JSON outputs.
- **Automated tests:** API tests for health, scoring, and invalid input handling.

## Technology Stack

- Python
- pandas, NumPy, scikit-learn
- XGBoost
- PyTorch and PyTorch Geometric
- Apache Kafka and kafka-python
- FastAPI and Uvicorn
- MLflow
- Feast
- Evidently AI
- pytest and HTTPX
- Docker Compose

## Architecture

```text
PaySim Transaction Dataset
          |
          v
  Data Exploration
          |
          v
 Feature Preparation
          |
          +----------------------+
          |                      |
          v                      v
   XGBoost Baseline        GNN Experiment
          |
          v
  Saved ML Model
          |
          v
      FastAPI
      /      \
     v        v
Transaction  Monitoring
 Scoring      Status
     ^
     |
 Kafka Consumer
     ^
     |
 Kafka Transactions Topic
     ^
     |
 Kafka Producer

Supporting components:
- MLflow: experiment tracking
- Feast: local feature-store demonstration
- Evidently AI: historical data-drift reports
```

## Repository Structure

```text
fraud-detection-platform/
├── data/
│   └── transactions.csv
├── feature_repo/
│   ├── feature_store.yaml
│   ├── feature_definitions.py
│   └── data/
├── models/
│   ├── fraud_model.joblib
│   └── fraud_gnn.pt
├── reports/
│   ├── class_distribution.png
│   ├── fraud_by_type.png
│   ├── model_metrics.json
│   ├── xgb_same_split_metrics.json
│   ├── gnn_metrics.json
│   ├── data_drift_report.html
│   ├── data_drift_summary.json
│   └── drift_status.json
├── src/
│   ├── api/
│   │   └── main.py
│   ├── graph/
│   │   ├── compare_xgboost.py
│   │   └── train_gnn.py
│   ├── monitoring/
│   │   ├── data_drift.py
│   │   └── summarize_drift.py
│   ├── streaming/
│   │   ├── producer.py
│   │   └── consumer.py
│   ├── eda.py
│   ├── train.py
│   ├── log_existing_model.py
│   ├── prepare_feast_data.py
│   └── test_feast.py
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

*Note: Include only files that actually exist in your repository. Large datasets, model binaries, virtual environments, and generated database files may be excluded from Git and recreated using documented steps.*

## Dataset

The project uses the PaySim synthetic mobile-money transaction dataset.

Important fields include transaction step, transaction type, amount, source and destination accounts, and the fraud label.

The baseline model uses engineered features including:

- Transaction step
- Transaction amount and log-transformed amount
- Transaction type
- Destination merchant indicator
- Hour-of-day derived from the transaction step

Balance fields and `isFlaggedFraud` are excluded from the baseline feature set to reduce leakage risk.

## Setup

Create and activate a Python virtual environment. Commands below are for Windows PowerShell.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the PaySim CSV at:

```text
data/transactions.csv
```

Install and start Docker Desktop if you intend to run the Kafka streaming demo.

## Run the API

From the project root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Monitoring status: http://127.0.0.1:8000/monitoring/status

### Example scoring request

```powershell
$body = @{
    step = 100
    amount = 50000
    type = "TRANSFER"
    nameDest = "M123456789"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/score `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The response includes a fraud probability, decision threshold, flag decision, risk level, and model version.

## Kafka Streaming Demo

Start Kafka using Docker Compose:

```powershell
docker compose up -d
```

In separate PowerShell terminals:

**Terminal 1 — API**

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Consumer**

```powershell
python -u -m src.streaming.consumer
```

**Terminal 3 — Producer**

```powershell
python -u -m src.streaming.producer --limit 20 --delay 0.1
```

The producer publishes transactions to Kafka. The consumer calls the scoring API and publishes scored results to the `fraud_scored` topic.

## Evidently AI Monitoring

Generate the historical data-drift report:

```powershell
python -u src/monitoring/data_drift.py
python -u src/monitoring/summarize_drift.py
```

Generated artifacts:

- `reports/data_drift_report.html`
- `reports/data_drift_summary.json`
- `reports/drift_status.json`

Open the HTML report to inspect feature-level distribution changes.

The monitoring endpoint exposes the saved summary. The current implementation does not automatically retrain the model or establish that detected drift has reduced predictive performance.

## MLflow Experiment Tracking

Start the local MLflow tracking server using the project's configured SQLite database:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 to inspect logged experiments.

## Feast Feature Store

The project includes a local Feast configuration and a demonstration of historical feature retrieval.

From the feature repository directory:

```powershell
cd feature_repo
feast apply
```

Run the project's documented materialization and test commands after preparing the feature data. This component is a local demonstration, not a complete online feature-serving system for newly arriving transactions.

## Testing

Run the API tests from the project root:

```powershell
python -m pytest tests/test_api.py -v
```

The existing API test suite has passed four tests covering health, scoring, and invalid inputs.

## Model Evaluation

The project evaluates fraud classification using precision, recall, F1 score, and average precision (PR-AUC). Since fraud is rare, accuracy alone is not a sufficient evaluation metric.

A same-split benchmark experiment reported the following results:

| Metric | XGBoost baseline |
|---|---:|
| Average precision (PR-AUC) | 0.2721 |
| Precision | 0.4105 |
| Recall | 0.2400 |
| F1 score | 0.3029 |
| False positives | 112 |

These results apply to the sampled chronological evaluation used by the benchmark script. Reproduce the experiment before comparing it with other models.

The current GNN experiment has substantially weaker ranking performance and produces many false positives. It is retained as an experimental component rather than the recommended scoring model.

## Limitations and Future Improvements

- Improve graph construction and the handling of previously unseen accounts.
- Evaluate models on exactly the same transaction rows and chronological split.
- Validate probability calibration and decision thresholds against operational requirements.
- Evaluate drift alerts alongside delayed fraud labels and model-performance metrics.
- Improve streaming resilience, retries, error handling, and observability.
- Add containerized API deployment and Kubernetes manifests.
- Add production-grade authentication, logging, secrets management, and monitoring.

## Disclaimer

This project is for educational and demonstration purposes. It uses synthetic transaction data and has not been validated for real financial decisions. A flagged transaction should be treated as a signal for further review, not proof of fraud.
