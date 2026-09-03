import json
import uuid

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import settings
from .modeling import FEATURES, generate_customers, get_model
from .storage import connection, latest_model, now


def psi(expected: list[float], actual: list[float]) -> float:
    expected_values = np.clip(np.asarray(expected, dtype=float), 0.0001, None)
    actual_values = np.clip(np.asarray(actual, dtype=float), 0.0001, None)
    return round(float(np.sum((actual_values - expected_values) * np.log(actual_values / expected_values))), 4)


def monitor_values(values: np.ndarray, scenario: str, source: str) -> dict:
    model_run = latest_model()
    if not model_run:
        from .modeling import train_model
        model_run = train_model()
    feature_scores: dict[str, float] = {}
    for index, name in enumerate(FEATURES):
        profile = model_run["baseline"][name]
        bins = np.asarray(profile["edges"])
        counts, _ = np.histogram(values[:, index], bins=bins)
        actual = (counts / max(counts.sum(), 1)).tolist()
        feature_scores[name] = psi(profile["proportions"], actual)
    max_psi = max(feature_scores.values())
    if max_psi >= settings.drift_critical_threshold:
        severity, recommendation = "critical", "Pause automated decisions; investigate data source and approve retraining only after validation."
    elif max_psi >= settings.drift_warning_threshold:
        severity, recommendation = "warning", "Review affected features and collect delayed labels before scheduling retraining."
    else:
        severity, recommendation = "healthy", "Continue monitoring; no operator action is required."
    drifted = [name for name, score in feature_scores.items() if score >= settings.drift_warning_threshold]
    result = {"id": str(uuid.uuid4()), "scenario": scenario, "severity": severity, "max_psi": round(max_psi, 4),
              "drifted_features": drifted, "recommendation": recommendation, "created_at": now(),
              "details": {"feature_psi": feature_scores,
                          "source": source, "batch_rows": len(values),
                          "mean_churn_probability": round(float(get_model().predict_proba(values)[:, 1].mean()), 3)}}
    with connection() as conn:
        conn.execute("INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (result["id"], scenario, severity, result["max_psi"], json.dumps(drifted), recommendation,
                      json.dumps(result["details"]), result["created_at"]))
    return result


def evaluate_batch(scenario: str) -> dict:
    values, _ = generate_customers(700, scenario=scenario,
                                   seed={"healthy": 100, "warning": 200, "critical": 300}[scenario])
    return monitor_values(values, scenario, "built-in simulator")


def delayed_label_metrics(scenario: str) -> dict:
    """Simulate labels arriving after an already-monitored production batch."""
    values, labels = generate_customers(700, scenario=scenario,
                                        seed={"healthy": 100, "warning": 200, "critical": 300}[scenario])
    probabilities = get_model().predict_proba(values)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {"scenario": scenario, "rows": len(values), "accuracy": round(float(accuracy_score(labels, predictions)), 3),
            "precision": round(float(precision_score(labels, predictions, zero_division=0)), 3),
            "recall": round(float(recall_score(labels, predictions, zero_division=0)), 3),
            "f1": round(float(f1_score(labels, predictions, zero_division=0)), 3),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 3),
            "note": "These metrics are available only after ground-truth churn labels arrive."}


def reset_demo() -> None:
    with connection() as conn:
        conn.execute("DELETE FROM incidents")
        conn.execute("DELETE FROM predictions")


def recent_incidents(limit: int = 20) -> list[dict]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(row), "drifted_features": json.loads(row["drifted_features_json"]),
             "details": json.loads(row["details_json"])} for row in rows]
