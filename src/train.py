import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument(
    "--iteration",
    type=int,
    required=True,
    choices=[1, 2],
)

args = parser.parse_args()

if args.iteration == 1:
    data_path = "data/processed/processed_v0.parquet"
    model_path = "models/model_iteration1.joblib"
else:
    data_path = "data/processed/processed_v0_v1.parquet"
    model_path = "models/model_iteration2.joblib"


print(f"\nLoading {data_path}")

df = pd.read_parquet(data_path)

# Features
X = df[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = df["target"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report\n")
print(classification_report(y_test, pred))

# Save model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, model_path)

print(f"\nModel saved to {model_path}")

# Save metrics
Path("reports").mkdir(exist_ok=True)

with open(
    f"reports/iteration_{args.iteration}_metrics.txt",
    "w",
) as f:

    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(classification_report(y_test, pred))

print("\nMetrics saved.")