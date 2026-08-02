from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

# Entity
stock = Entity(
    name="stock_name",
    join_keys=["stock_name"],
)

# Offline source
stock_source = FileSource(
    name="stock_source",
    path="data/train.parquet",
    timestamp_field="timestamp",
)

# Feature View
stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
    online=True,
)