"""Versioned API router — aggregates all v1 endpoint modules."""

from fastapi import APIRouter

from app.api.v1.endpoints import alerts, health, predictions

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
