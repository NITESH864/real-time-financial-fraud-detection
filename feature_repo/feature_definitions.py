from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    description="PaySim transaction replay identifier",
)

transaction_source = FileSource(
    name="transaction_source",
    path="data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="step", dtype=Int64),
        Field(name="amount", dtype=Float64),
        Field(name="logAmount", dtype=Float64),
        Field(name="type", dtype=String),
        Field(name="isMerchantDest", dtype=Int64),
        Field(name="hourOfDay", dtype=Int64),
    ],
    source=transaction_source,
    online=True,
    description="Features used by the PaySim fraud model",
)
