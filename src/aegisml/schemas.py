from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=10000)
    support_tickets: int = Field(ge=0, le=100)
    engagement_score: float = Field(ge=0, le=1)
    auto_pay: bool


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: Literal["low", "medium", "high"]
    quality_warnings: list[str]


class BatchResult(BaseModel):
    id: str
    scenario: str
    severity: Literal["healthy", "warning", "critical"]
    max_psi: float
    drifted_features: list[str]
    recommendation: str
    created_at: datetime

