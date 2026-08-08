"""Pydantic schemas for early-warning alerts."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.prediction import GeoPoint, RiskLevel


class AlertResponse(BaseModel):
    id: int
    region_name: str
    location: GeoPoint
    risk_level: RiskLevel
    probability: float = Field(..., ge=0, le=1)
    valid_from: datetime
    valid_until: datetime
    message: str = Field(..., description="Human-readable warning text, including the AI's reasoning")
    issued_at: datetime
    is_active: bool
