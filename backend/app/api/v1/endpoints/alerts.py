"""
Early-warning alert endpoints.

Alerts are generated from model predictions that cross configured risk
thresholds (Phase 7 wires this to the prediction pipeline). The listing
endpoint is functional now and returns whatever alerts exist in the
database — an empty list until the pipeline produces real ones.
"""

from fastapi import APIRouter

from app.schemas.alert import AlertResponse
from app.services.alert_service import AlertService

router = APIRouter()

_service = AlertService()


@router.get("", response_model=list[AlertResponse])
def list_active_alerts() -> list[AlertResponse]:
    """All currently active early-warning alerts, newest first."""
    return _service.list_active()
