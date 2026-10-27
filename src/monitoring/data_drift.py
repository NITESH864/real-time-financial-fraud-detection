
"""Compare earlier and later PaySim transactions using Evidently."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transactions.csv"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "step",
    "amount",
    "type",
    "nameDest",
    "isFraud",
]

FEATURE_COLUMNS = [
    "step",
    "amount",
    "logAmount",
    "type",
    "isMerchantDest",
    "hourOfDay",
]


def load_chronological_sample(path, sample_size=100_000, seed=42):
    """Sample rows separately from earlier and later transaction periods."""
    df = pd.read_csv(path, usecols=FEATURES)

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["isFraud"] = pd.to_numeric(df["isFraud"], errors="coerce")

    df = df.dropna(subset=FEATURES).copy()
    df = df.sort_values("step").reset_index(drop=True)

    # Split chronologically BEFORE sampling to avoid mixing periods.
    split_index = int(len(df) * 0.70)
    reference = df.iloc[:split_index].copy()
    current = df.iloc[split_index:].copy()

    # Sample each period independently to keep memory use manageable
    # for subsequent processing. The CSV itself is still loaded in memory.
    reference = reference.sample(
        n=min(sample_size, len(reference)),
        random_state=seed,
    )
    current = current.sample(
        n=min(sample_size, len(current)),
        random_state=seed,
    )

    def prepare_features(data):
        data = data.copy()
        data["logAmount"] = np.log1p(data["amount"].clip(lower=0))
        data["isMerchantDest"] = (
            data["nameDest"].astype(str).str.startswith("M").astype(int)
        )
        data["hourOfDay"] = (data["step"] % 24).astype(int)
        return data[FEATURE_COLUMNS].reset_index(drop=True)

    return prepare_features(reference), prepare_features(current)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and sorting transactions chronologically...", flush=True)
    reference, current = load_chronological_sample(DATA_PATH)

    print(f"Reference rows: {len(reference):,}", flush=True)
    print(f"Current rows:   {len(current):,}", flush=True)
    print(
        f"Reference step range: {reference['step'].min()} "
        f"to {reference['step'].max()}",
        flush=True,
    )
    print(
        f"Current step range: {current['step'].min()} "
        f"to {current['step'].max()}",
        flush=True,
    )
    print("Generating Evidently drift report...", flush=True)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference,
        current_data=current,
    )

    html_path = REPORT_DIR / "data_drift_report.html"
    json_path = REPORT_DIR / "data_drift_summary.json"

    snapshot.save_html(str(html_path))

    # Evidently's report object supports dict serialization in the
    # installed version; default=str handles non-JSON-native values.
    report_data = snapshot.dict()
    json_path.write_text(
        json.dumps(report_data, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nMonitoring completed successfully.")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")
    print("Review the report for features whose distributions changed.")


if __name__ == "__main__":
    main()
