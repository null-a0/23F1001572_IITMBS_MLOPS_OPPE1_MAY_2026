import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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

# ---------------------------------
# Dataset Selection
# ---------------------------------

if args.iteration == 1:
    data_path = "data/processed/processed_v0.parquet"
else:
    data_path = "data/processed/processed_v0_v1.parquet"

print(f"\nLoading {data_path}")

df = pd.read_parquet(data_path)

X = df[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ---------------------------------
# MLflow
# ---------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    f"StockMovement_Iteration_{args.iteration}"
)

# ---------------------------------
# Hyperparameter Grid
# ---------------------------------

param_grid = [

    {
        "n_estimators":100,
        "max_depth":10,
    },

    {
        "n_estimators":100,
        "max_depth":20,
    },

    {
        "n_estimators":200,
        "max_depth":10,
    },

    {
        "n_estimators":200,
        "max_depth":20,
    },

]

best_accuracy = -1
best_model = None

Path("models").mkdir(exist_ok=True)

# ---------------------------------
# Hyperparameter Search
# ---------------------------------

for params in param_grid:

    with mlflow.start_run():

        print("\nRunning:", params)

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            pred,
        )

        report = classification_report(
            y_test,
            pred,
        )

        print("Accuracy:", accuracy)

        mlflow.log_params(params)
        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        report_file = "classification_report.txt"

        with open(report_file,"w") as f:
            f.write(report)

        mlflow.log_artifact(report_file)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_params = params

# ---------------------------------
# Save Best Model
# ---------------------------------

model_path = f"models/best_model_iteration{args.iteration}.joblib"

joblib.dump(
    best_model,
    model_path,
)

print("\nBest Accuracy:", best_accuracy)
print("Best Parameters:", best_params)

print("\nBest model saved:", model_path)

# ---------------------------------
# Register Model
# ---------------------------------

with mlflow.start_run(run_name="Best_Model"):

    mlflow.log_params(best_params)
    mlflow.log_metric(
        "best_accuracy",
        best_accuracy,
    )

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="StockMovementPredictor",
    )

print("\nModel Registered Successfully.")