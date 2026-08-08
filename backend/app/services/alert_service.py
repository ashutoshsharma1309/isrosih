"""
Alert service — reads/writes early-warning alerts.

Phase 7 connects this to the PostGIS `alerts` table (see
database/schema.sql) and to the prediction pipeline that generates
alerts when risk thresholds are crossed. Until then it returns an empty
list rather than fabricated alerts.
"""

from app.schemas.alert import AlertResponse


class AlertService:
    def list_active(self) -> list[AlertResponse]:
        # Phase 7: SELECT from alerts WHERE is_active ORDER BY issued_at DESC
        return []
