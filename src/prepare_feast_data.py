from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "transactions.csv"
OUTPUT = ROOT / "feature_repo" / "data" / "transaction_features.parquet"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

columns = ["step", "type", "amount", "nameDest"]

df = pd.read_csv(SOURCE, usecols=columns)

# Reproducible sample for the local feature-store demo.
df = df.sample(
    n=min(100_000, len(df)),
    random_state=42,
).reset_index(drop=True)

df["transaction_id"] = df.index.astype("int64")
df["isMerchantDest"] = (
    df["nameDest"].str.startswith("M").astype("int64")
)
df["hourOfDay"] = (df["step"] % 24).astype("int64")
df["logAmount"] = (
    df["amount"].clip(lower=0).map(lambda x: __import__("math").log1p(x))
)

# Map simulation hours to timestamps for Feast.
base_time = pd.Timestamp("2024-01-01", tz="UTC")
df["event_timestamp"] = (
    base_time + pd.to_timedelta(df["step"], unit="h")
)

features = df[
    [
        "transaction_id",
        "event_timestamp",
        "step",
        "amount",
        "logAmount",
        "type",
        "isMerchantDest",
        "hourOfDay",
    ]
].copy()

features["step"] = features["step"].astype("int64")
features["amount"] = features["amount"].astype("float64")
features["logAmount"] = features["logAmount"].astype("float64")
features["type"] = features["type"].astype(str)
features["isMerchantDest"] = features["isMerchantDest"].astype("int64")
features["hourOfDay"] = features["hourOfDay"].astype("int64")

features.to_parquet(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Rows:", len(features))
print(features.head())
