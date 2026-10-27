from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/transactions.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFraud distribution:")
print(df["isFraud"].value_counts())

fraud_rate = df["isFraud"].mean() * 100
print(f"\nFraud rate: {fraud_rate:.4f}%")

print("\nFraud by transaction type:")
print(
    df.groupby("type")["isFraud"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

# Save the fraud class distribution.
df["isFraud"].value_counts().sort_index().plot(
    kind="bar", rot=0
)
plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "class_distribution.png")
plt.close()

# Save fraud rate by transaction type.
(
    df.groupby("type")["isFraud"]
    .mean()
    .sort_values(ascending=False)
    .plot(kind="bar", rot=30)
)
plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate")
plt.tight_layout()
plt.savefig(REPORT_DIR / "fraud_by_type.png")
plt.close()

print("\nEDA completed. Charts saved in reports/.")
