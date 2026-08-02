import pandas as pd
from feast import FeatureStore

# Load Feast Feature Store
store = FeatureStore(repo_path="feature_repo/feature_repo")

# Use a representative subset (100 rows)
entity_df = (
    pd.read_parquet("feature_repo/feature_repo/data/train.parquet")
    [["stock_name", "timestamp"]]
    .head(100)
)

print(f"Entity dataframe shape: {entity_df.shape}")

# Point-in-time retrieval
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
).to_df()

print("=" * 60)
print("Historical Features Retrieved Successfully")
print("=" * 60)
print(training_df.head())
print(f"\nShape: {training_df.shape}")
print(f"\nColumns: {training_df.columns.tolist()}")

training_df.to_parquet(
    "feature_repo/feature_repo/data/feast_training.parquet",
    index=False,
)

print("\nSaved to feature_repo/feature_repo/data/feast_training.parquet")