# AegisML

**An ML reliability platform that detects data drift before a production model quietly fails.** AegisML trains a customer-churn classifier, compares live batches to the training baseline, assigns risk severity, generates incident records, and recommends actions such as investigation or retraining.

## Why this matters

Most ML projects end at model accuracy. AegisML demonstrates the operational work after deployment: serving predictions, protecting input quality, detecting feature/prediction drift, measuring model performance when labels arrive, and making retraining decisions auditable.

## Features

- reproducible churn-model training with feature metadata and baseline distributions;
- prediction API with schema validation and data-quality checks;
- Population Stability Index (PSI) for feature drift and prediction drift;
- alert severity: `healthy`, `warning`, or `critical`;
- delayed-label performance monitoring: accuracy, precision, recall, F1, ROC-AUC;
- incident timeline with recommended operator action;
- scenario simulator for normal, warning, and severe production batches;
- CSV batch upload, an individual prediction form, and delayed-label performance checks;
- FastAPI dashboard, Docker, automated tests, and CI.

## Architecture

```text
Training data -> Logistic Regression -> model + baseline profile
Live batch -> quality gates -> PSI drift monitor -> alert policy -> incident store/dashboard
Delayed labels -----------------------------------------------> performance monitor -> retrain recommendation
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
uvicorn aegisml.main:app --reload
```

Open `http://127.0.0.1:8000`. Click a simulator scenario, then inspect drift signals and incidents. API documentation is at `/docs`.

### Five-minute demo

1. Open the dashboard and click **Clear demo events**.
2. Click **Healthy batch**, **Warning drift**, then **Critical drift**. Notice that the recommended action changes by severity.
3. Upload `data/demo/customer_batch_warning.csv` to exercise the real CSV ingestion path.
4. Change values in **Score one customer** and inspect the churn probability and input-quality warnings.
5. Click **Simulate delayed-label performance check**. This illustrates that accuracy/F1 can only be measured after real outcomes arrive.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/model/train` | Create/recreate model and baseline |
| `POST` | `/api/v1/predictions` | Score one customer record |
| `POST` | `/api/v1/batches/simulate/{scenario}` | Run `healthy`, `warning`, or `critical` batch |
| `GET` | `/api/v1/overview` | Dashboard data and most recent incident |
| `GET` | `/api/v1/incidents` | Audit trail |

## Design notes

AegisML keeps monitoring decisions explicit: PSI provides an interpretable drift signal, delayed-label metrics are calculated only when outcomes arrive, and severity is separated from retraining recommendations. The current local dashboard is intentionally small; production deployments could add streaming ingestion, a feature store, telemetry, model versioning, and role-based incident ownership.
