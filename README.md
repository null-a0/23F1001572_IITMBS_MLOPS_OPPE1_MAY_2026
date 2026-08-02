# 📈 Stock Movement Predictor - End-to-End MLOps Pipeline

## IITM BS MLOps OPPE-1

**Repository:** `23F1001572_IITMBS_MLOPS_OPPE1_MAY_2026`

---

# Project Overview

This project implements an **end-to-end MLOps pipeline** for predicting stock price movement using minute-level stock market data.

The objective is to predict whether the closing price of a stock **5 minutes in the future** will be higher than the current closing price.

The project demonstrates the complete MLOps lifecycle using modern tools including:

* Google Cloud Platform (GCP)
* DVC
* Feast Feature Store
* MLflow
* GitHub Actions
* PyTest
* Scikit-Learn

---

# Problem Statement

Given historical minute-level stock market data, predict whether the stock price will move **UP (1)** or **DOWN (0)** after the next five minutes.

Target:

* **1** → Close price after 5 minutes is greater than current close price
* **0** → Otherwise

Features used:

* rolling_avg_10
* volume_sum_10

---

# Technology Stack

| Component           | Tool                  |
| ------------------- | --------------------- |
| Cloud Platform      | Google Cloud Platform |
| Version Control     | Git & GitHub          |
| Data Versioning     | DVC                   |
| Remote Storage      | Google Cloud Storage  |
| Feature Store       | Feast                 |
| ML Framework        | Scikit-Learn          |
| Experiment Tracking | MLflow                |
| Model Registry      | MLflow Registry       |
| CI                  | GitHub Actions        |
| Testing             | PyTest                |

---

# Repository Structure

```text
.
├── data/
│   ├── v0
│   ├── v1
│   ├── processed/
│   ├── v0.dvc
│   └── v1.dvc
│
├── feature_repo/
│
├── models/
│
├── reports/
│
├── src/
│   ├── preprocess.py
│   ├── feast_retrieval.py
│   ├── train.py
│   ├── hyperparameter_tuning.py
│   └── evaluate.py
│
├── tests/
│
├── .github/workflows/
│
├── requirements.txt
└── README.md
```

---

# MLOps Pipeline

```
Raw Stock Data
       │
       ▼
      DVC
       │
       ▼
 Google Cloud Storage
       │
       ▼
 Feast Feature Store
       │
       ▼
 Data Preprocessing
       │
       ▼
 Model Training
       │
       ▼
 Hyperparameter Tuning
       │
       ▼
 MLflow Tracking
       │
       ▼
 Model Registry
       │
       ▼
 Evaluation
       │
       ▼
 PyTest
       │
       ▼
 GitHub Actions CI
```

---

# Deliverable 1 – Git Repository & GCP

* Private GitHub Repository
* Google Cloud Compute Engine VM
* Google Cloud Storage Bucket
* Collaborator Added

---

# Deliverable 2 – Data Versioning using DVC

Implemented:

* Initialized DVC
* Added Version 0 dataset
* Added Version 1 dataset
* Configured Google Cloud Storage Remote
* Uploaded datasets using:

```bash
dvc push
```

Retrieve dataset:

```bash
dvc pull
```

---

# Deliverable 3 – Feast Feature Store

Implemented Feature Store with:

Entity

* stock_name

Features

* rolling_avg_10
* volume_sum_10

Feature retrieval script:

```bash
python src/feast_retrieval.py
```

---

# Deliverable 4 – Training & Evaluation

Two incremental training iterations were performed.

### Iteration 1

Dataset:

* Version 0

Command

```bash
python src/train.py --iteration 1
```

---

### Iteration 2

Dataset

* Version 0 + Version 1

Command

```bash
python src/train.py --iteration 2
```

Evaluation metrics are stored inside:

```
reports/
```

---

# Deliverable 5 – Hyperparameter Tuning & MLflow

Hyperparameter tuning performed using Random Forest.

Tracked using MLflow:

* Parameters
* Metrics
* Artifacts

Best model registered into the MLflow Model Registry.

Run:

```bash
python src/hyperparameter_tuning.py --iteration 1
```

```bash
python src/hyperparameter_tuning.py --iteration 2
```

---

# Deliverable 6 – Continuous Integration

GitHub Actions automatically executes:

* Install dependencies
* Pull dataset using DVC
* Train model
* Evaluate model
* Execute feature sanity tests
* Upload evaluation report

Workflow:

```
.github/workflows/ci.yml
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Pull Data

```bash
dvc pull
```

---

## Feature Retrieval

```bash
python src/feast_retrieval.py
```

---

## Preprocessing

Iteration 1

```bash
python src/preprocess.py --iteration 1
```

Iteration 2

```bash
python src/preprocess.py --iteration 2
```

---

## Training

Iteration 1

```bash
python src/train.py --iteration 1
```

Iteration 2

```bash
python src/train.py --iteration 2
```

---

## Hyperparameter Tuning

Iteration 1

```bash
python src/hyperparameter_tuning.py --iteration 1
```

Iteration 2

```bash
python src/hyperparameter_tuning.py --iteration 2
```

---

## Evaluation

```bash
python src/evaluate.py
```

---

## Unit Testing

```bash
pytest tests/ -v
```

---

# Output

Generated artifacts include:

* Trained Models (`models/`)
* Processed Dataset (`data/processed/`)
* Evaluation Reports (`reports/`)
* Final Markdown Report (`report.md`)
* MLflow Tracking Database (`mlflow.db`)
* MLflow Experiments (`mlruns/`)

---

# Future Improvements

* XGBoost and LightGBM model comparison
* Docker containerization
* FastAPI deployment
* Kubernetes deployment
* Continuous Model Monitoring

---

# Author

**Abhishek Saha**

IIT Madras BS Degree Program in Data Science and Applications

Roll Number: **23F1001572**
