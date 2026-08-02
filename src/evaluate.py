import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Local MLflow tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load latest registered model
import joblib

model = joblib.load("models/best_model_iteration2.joblib")
# Load processed dataset
df = pd.read_parquet(
    "data/processed/processed_v0_v1.parquet"
)

X = df[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = df["target"]

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

with open("report.md", "w") as f:

    f.write("# Stock Movement Predictor\n\n")

    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Accuracy | {accuracy:.4f} |\n")
    f.write(f"| Precision | {precision:.4f} |\n")
    f.write(f"| Recall | {recall:.4f} |\n")
    f.write(f"| F1 Score | {f1:.4f} |\n")

print("Evaluation completed.")