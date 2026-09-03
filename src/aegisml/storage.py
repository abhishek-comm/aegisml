import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .config import settings


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = Path(settings.database_url.removeprefix("sqlite:///"))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize() -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_runs (
          id TEXT PRIMARY KEY, metrics_json TEXT NOT NULL, baseline_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS incidents (
          id TEXT PRIMARY KEY, scenario TEXT NOT NULL, severity TEXT NOT NULL, max_psi REAL NOT NULL,
          drifted_features_json TEXT NOT NULL, recommendation TEXT NOT NULL, details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
          id TEXT PRIMARY KEY, probability REAL NOT NULL, quality_warnings_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)


def now() -> str:
    return datetime.now(UTC).isoformat()


def latest_model() -> dict | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM model_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return None
    payload = dict(row)
    payload["metrics"] = json.loads(payload.pop("metrics_json"))
    payload["baseline"] = json.loads(payload.pop("baseline_json"))
    return payload

