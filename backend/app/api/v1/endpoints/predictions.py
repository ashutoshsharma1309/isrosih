"""
Rainfall prediction endpoints.

The API contract (request/response schemas) is defined now so the frontend
and model teams can develop against a stable interface. The endpoints
return HTTP 501 until trained models are integrated (Phases 3–6) — the
platform never fabricates predictions.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService, ModelNotAvailableError

router = APIRouter()

_service = PredictionService()


@router.post("", response_model=PredictionResponse)
def predict_rainfall(request: PredictionRequest) -> PredictionResponse:
    """Run the rainfall risk model for a location and forecast horizon.

    Returns a probability-based risk assessment with an attached
    explanation payload (SHAP feature attributions; Grad-CAM overlays when
    satellite imagery is part of the input).
    """
    try:
        return _service.predict(request)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc


@router.get("/model-info")
def model_info() -> dict:
    """Metadata about the currently loaded model (version, training window,
    feature set). Used by the dashboard's transparency panel."""
    return _service.model_info()
