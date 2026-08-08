"""
Pydantic schemas defining the prediction API contract.

These are the single source of truth for what the frontend sends and
receives — model implementations (Phases 3–6) must conform to them.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Rainfall impact classification, aligned with IMD-style severity tiers."""

    LOW = "low"
    MODERATE = "moderate"
    HEAVY = "heavy"
    EXTREME = "extreme"


class GeoPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class WeatherFeatures(BaseModel):
    """Meteorological inputs to the tabular model.

    All fields optional: when omitted, the service fetches the latest
    observed values for the location from the database (Phase 7).
    """

    temperature_c: float | None = Field(None, description="Surface air temperature, °C")
    humidity_pct: float | None = Field(None, ge=0, le=100, description="Relative humidity, %")
    pressure_hpa: float | None = Field(None, description="Mean sea-level pressure, hPa")
    wind_speed_ms: float | None = Field(None, ge=0, description="Wind speed, m/s")
    cloud_cover_pct: float | None = Field(None, ge=0, le=100, description="Total cloud cover, %")


class PredictionRequest(BaseModel):
    location: GeoPoint
    region_name: str | None = Field(None, description="Human-readable place name, e.g. 'Kerala'")
    horizon_hours: int = Field(12, ge=1, le=72, description="Forecast lead time in hours")
    weather: WeatherFeatures | None = None
    include_explanation: bool = Field(True, description="Attach SHAP/Grad-CAM explanation payload")


class FeatureAttribution(BaseModel):
    """One SHAP feature contribution, for the explanation panel."""

    feature: str
    value: float | None = Field(None, description="The input value the model saw")
    contribution: float = Field(..., description="Signed SHAP value (log-odds or probability units)")


class ImageExplanation(BaseModel):
    """Grad-CAM output for satellite-image-based predictions."""

    satellite_image_id: str
    heatmap_url: str = Field(..., description="URL of the Grad-CAM overlay rendered by the backend")
    description: str = Field(..., description="Plain-language summary of highlighted regions")


class Explanation(BaseModel):
    feature_attributions: list[FeatureAttribution] = []
    image_explanation: ImageExplanation | None = None
    narrative: str = Field(
        "",
        description="Human-readable sentence explaining the main drivers of the prediction",
    )


class PredictionResponse(BaseModel):
    location: GeoPoint
    region_name: str | None
    generated_at: datetime
    horizon_hours: int
    probability: float = Field(..., ge=0, le=1, description="Probability of a high-impact rain event")
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    model_version: str
    explanation: Explanation | None = None
