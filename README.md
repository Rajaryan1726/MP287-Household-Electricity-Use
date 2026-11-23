# MP287 - Household Electricity Use (Dehradun)

SCSE3040 MLOps Mini Project, Bennett University

| | |
|---|---|
| **Group ID** | G153 |
| **Problem ID** | MP287 |

## Team Members

| Name | Enrolment No. |
|---|---|
| Raj Aryan | S24CSEU2450 |
| Vedika Sharma | S24CSEU2437 |
| Rakesh Kukkadapu | S24CSEU2438 |

## Problem

The Dehradun Power Distribution Office wants to predict each household's monthly electricity use (`units_kwh`) so that it can warn families mid-month about a high bill and plan transformer load. This is a regression problem and models are measured with MAE (Mean Absolute Error, in kWh).

## What This Repository Contains

An end-to-end MLOps pipeline:

- Data versioning with DVC and code that detects and fixes seven kinds of data problems
- Experiment tracking and model registry with MLflow (SQLite backend, `champion` alias)
- A FastAPI prediction service with `POST /predict` and `GET /health`
- JSON logging, one line per prediction, written to stdout
- A Docker image (`mp287-api:1.0`)
- GitHub Actions CI that runs pytest on every push
- Kubernetes manifests (Deployment `mp287-api`, Service `mp287-svc`)
- Drift monitoring (PSI), MAE per feeder, retraining (v2) and a promotion rule
- Final predictions for `holdout_features.csv`

## Repository Structure

```
README.md
requirements.txt              pinned dependencies
params.yaml                   paths and model settings
src/mp287/                    data checks, cleaning, training, monitoring, logging
tests/                        pytest tests (including a TestClient test)
data/raw/                     DVC-tracked train.csv and later.csv, plus holdout_features.csv
app/main.py                   FastAPI service
models/model.joblib           champion model loaded by the service
mlflow.db                     MLflow tracking database
Dockerfile
.dockerignore
.github/workflows/ci.yml      CI pipeline
k8s/deployment.yaml
k8s/service.yaml
reports/findings.yaml         data issues, tracking and monitoring findings
predictions/holdout_predictions.csv
```

## Setup

```bash
conda create -n mp287 python=3.12 -y
conda activate mp287
pip install -r requirements.txt
```

The CSV files are not committed to Git. Place the original `train.csv` and `later.csv` in `data/raw/` (they are tracked with DVC).

## Run the Tests

```bash
pytest
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rooms": 4, "members": 5, "ac_hours_per_day": 6, "avg_temp_c": 32,
       "house_area_sqft": 1400, "solar_kw": 0, "connection": "domestic",
       "feeder": "Feeder-1"}'
```

Input rules enforced by the request schema (invalid input returns status 422):

- `ac_hours_per_day` must not be above 25
- `avg_temp_c` must not be below 4

## Run with Docker

```bash
docker build -t mp287-api:1.0 .
docker run -p 8000:8000 mp287-api:1.0
```

## Kubernetes

The manifests in `k8s/` define a Deployment `mp287-api` (4 replicas, image `mp287-api:1.0`, container port 8000) and a Service `mp287-svc` (port 80 forwarding to 8000). Readiness and liveness probes use `/health`.

## Monitoring and Retraining

`later.csv` (July and August 2026) is used to compute PSI per numeric column, v1's MAE per feeder, and to retrain v2. v2 is promoted to `champion` only if its MAE on the last 30% of `later.csv` is at least 8% lower than v1's. Results are recorded in `reports/findings.yaml`.