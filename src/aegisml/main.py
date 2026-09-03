import csv
import json
import uuid
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .modeling import FEATURES, get_model, train_model
from .monitoring import (
    delayed_label_metrics,
    evaluate_batch,
    monitor_values,
    recent_incidents,
    reset_demo,
)
from .schemas import BatchResult, CustomerFeatures, PredictionResponse
from .storage import connection, initialize, latest_model, now


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    get_model()
    yield


app = FastAPI(title="AegisML", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": latest_model() is not None}


@app.post("/api/v1/model/train")
def train() -> dict:
    return train_model()


@app.post("/api/v1/predictions", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> dict:
    values = np.array([[getattr(customer, name) for name in FEATURES]], dtype=float)
    probability = float(get_model().predict_proba(values)[0, 1])
    quality_warnings = []
    if customer.monthly_charges > 6000:
        quality_warnings.append("monthly_charges is above the training range")
    if customer.support_tickets > 15:
        quality_warnings.append("support_tickets is above the training range")
    band = "high" if probability >= 0.65 else "medium" if probability >= 0.35 else "low"
    with connection() as conn:
        conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
                     (str(uuid.uuid4()), probability, json.dumps(quality_warnings), now()))
    return {"churn_probability": round(probability, 4), "risk_band": band, "quality_warnings": quality_warnings}


@app.post("/api/v1/batches/simulate/{scenario}", response_model=BatchResult)
def simulate(scenario: str) -> dict:
    if scenario not in {"healthy", "warning", "critical"}:
        raise HTTPException(422, "Scenario must be healthy, warning, or critical.")
    return evaluate_batch(scenario)


@app.post("/api/v1/batches/upload", response_model=BatchResult)
async def upload_batch(file: UploadFile = File(...)) -> dict:  # noqa: B008
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Upload a CSV containing the five feature columns shown in the dashboard.")
    try:
        rows = list(csv.DictReader(StringIO((await file.read()).decode("utf-8-sig"))))
        if not rows:
            raise ValueError("The CSV contains no rows.")
        values = np.array([[
            float(row["tenure_months"]), float(row["monthly_charges"]), float(row["support_tickets"]),
            float(row["engagement_score"]), float(str(row["auto_pay"]).lower() in {"true", "1", "yes"}),
        ] for row in rows])
    except (KeyError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, "CSV must contain tenure_months, monthly_charges, support_tickets, engagement_score, auto_pay.") from exc
    return monitor_values(values, "uploaded_csv", file.filename or "uploaded CSV")


@app.get("/api/v1/performance/simulate/{scenario}")
def delayed_labels(scenario: str) -> dict:
    if scenario not in {"healthy", "warning", "critical"}:
        raise HTTPException(422, "Scenario must be healthy, warning, or critical.")
    return delayed_label_metrics(scenario)


@app.post("/api/v1/demo/reset")
def reset() -> dict:
    reset_demo()
    return {"status": "Demo events cleared. The trained model was retained."}


@app.get("/api/v1/incidents")
def incidents() -> list[dict]:
    return recent_incidents()


@app.get("/api/v1/overview")
def overview() -> dict:
    model = latest_model()
    with connection() as conn:
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        incident_count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    return {"model": model["metrics"] if model else None, "predictions": prediction_count,
            "incidents": incident_count, "latest_incident": (recent_incidents(1) or [None])[0]}
