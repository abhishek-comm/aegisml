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

## Interview talking points

1. **Why PSI?** PSI compares binned distributions and is easy to explain, operate, and threshold; it complements model performance checks.
2. **Why delayed labels?** Production labels are often delayed, so data/prediction drift provides an early warning—not a false claim that model accuracy is already known.
3. **Why separate severity from retraining?** Drift can be a tracking bug or expected seasonality. AegisML escalates, preserves evidence, and recommends a human decision.
4. **Production next steps:** streaming ingestion (Kafka), feature store, OpenTelemetry metrics, canary models, model registry, and role-based incident ownership.
