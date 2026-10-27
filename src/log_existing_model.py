from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
TRACKING_DB = ROOT / "mlflow.db"

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not METRICS_PATH.exists():
    raise FileNotFoundError(METRICS_PATH)

with open(METRICS_PATH, encoding="utf-8") as f:
    metrics = json.load(f)

# Use a persistent local SQLite tracking database.
mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)
mlflow.set_experiment("paysim-fraud-detection")

model = joblib.load(MODEL_PATH)

with mlflow.start_run(run_name="xgboost-baseline-existing"):
    mlflow.set_tags({
        "dataset": "PaySim",
        "model_type": "XGBoost",
        "stage": "baseline",
        "data_split": "chronological",
        "source": "existing_training_run",
    })

    mlflow.log_params({
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.08,
        "scale_pos_weight": 1051.0683272790175,
        "decision_threshold": metrics[
            "threshold_selected_on_validation"
        ],
    })

    # Log scalar evaluation metrics.
    metric_keys = [
        "validation_best_f1",
        "test_average_precision_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    for key in metric_keys:
        mlflow.log_metric(key, float(metrics[key]))

    mlflow.log_metric(
        "test_rows", float(metrics["test_rows"])
    )
    mlflow.log_metric(
        "test_fraud_count", float(metrics["test_fraud_count"])
    )

    # Preserve the original model and exact evaluation report.
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")

    print("Logged existing baseline to MLflow.")
    print("Run ID:", mlflow.active_run().info.run_id)

print("Tracking database:", TRACKING_DB)
