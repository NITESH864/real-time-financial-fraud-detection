from pathlib import Path
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "feature_repo"

store = FeatureStore(repo_path=str(REPO))

result = store.get_online_features(
    entity_rows=[{"transaction_id": 0}],
    features=[
        "transaction_features:step",
        "transaction_features:amount",
        "transaction_features:logAmount",
        "transaction_features:type",
        "transaction_features:isMerchantDest",
        "transaction_features:hourOfDay",
    ],
).to_dict()

print("Feast online lookup:")
for key, value in result.items():
    print(f"{key}: {value}")

# Verify the entity was returned.
assert result["transaction_id"][0] == 0

# Validate features using their actual returned keys.
amount_key = next(
    key for key in result
    if key == "amount" or key.endswith(":amount")
)

amount = result[amount_key][0]

assert amount is not None, "Amount feature is missing."
assert float(amount) >= 0, "Amount must be non-negative."

print("\nFeast online lookup passed.")
