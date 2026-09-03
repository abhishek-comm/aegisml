import json
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings
from .storage import connection, now

FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "engagement_score", "auto_pay"]
MODEL_PATH = settings.model_dir / "churn_model.joblib"


def generate_customers(size: int, scenario: str = "healthy", seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(settings.random_seed if seed is None else seed)
    tenure = rng.integers(1, 73, size)
    charges = rng.normal(1800, 650, size).clip(200, 6000)
    tickets = rng.poisson(1.4, size).clip(0, 15)
    engagement = rng.beta(5, 2, size)
    auto_pay = rng.integers(0, 2, size)
    if scenario == "warning":
        charges = (charges * 1.18).clip(200, 9000)
        engagement = (engagement - 0.10).clip(0, 1)
    if scenario == "critical":
        tenure = rng.integers(1, 16, size)
        charges = rng.normal(3600, 900, size).clip(200, 9000)
        tickets = rng.poisson(5, size).clip(0, 40)
        engagement = rng.beta(2, 5, size)
        auto_pay = rng.binomial(1, 0.2, size)
    values = np.column_stack([tenure, charges, tickets, engagement, auto_pay]).astype(float)
    logit = -1.8 - 0.04 * tenure + 0.00055 * charges + 0.42 * tickets - 2.3 * engagement - 0.6 * auto_pay
    probability = 1 / (1 + np.exp(-logit))
    labels = rng.binomial(1, probability)
    return values, labels


def baseline_profile(values: np.ndarray) -> dict:
    profile = {}
    for index, name in enumerate(FEATURES):
        edges = np.quantile(values[:, index], np.linspace(0, 1, 11)).tolist()
        # Equal quantiles can occur for discrete fields; histogram still handles them after unique cleanup.
        edges = sorted(set(edges))
        if len(edges) < 2:
            edges = [edges[0] - 0.5, edges[0] + 0.5]
        histogram, edges = np.histogram(values[:, index], bins=edges)
        profile[name] = {"edges": edges.tolist(), "proportions": (histogram / histogram.sum()).tolist(),
                         "mean": round(float(values[:, index].mean()), 4)}
    return profile


def train_model() -> dict:
    values, labels = generate_customers(4000)
    train_x, test_x, train_y, test_y = train_test_split(values, labels, test_size=0.2,
                                                        random_state=settings.random_seed, stratify=labels)
    pipeline = Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    probabilities = pipeline.predict_proba(test_x)[:, 1]
    metrics = {"accuracy": round(float(accuracy_score(test_y, predictions)), 3),
               "f1": round(float(f1_score(test_y, predictions)), 3),
               "roc_auc": round(float(roc_auc_score(test_y, probabilities)), 3),
               "training_rows": len(train_x)}
    joblib.dump(pipeline, MODEL_PATH)
    run_id = str(uuid.uuid4())
    baseline = baseline_profile(train_x)
    with connection() as conn:
        conn.execute("INSERT INTO model_runs VALUES (?, ?, ?, ?)",
                     (run_id, json.dumps(metrics), json.dumps(baseline), now()))
    return {"id": run_id, "metrics": metrics, "baseline": baseline}


def get_model() -> Pipeline:
    if not Path(MODEL_PATH).exists():
        train_model()
    return joblib.load(MODEL_PATH)
