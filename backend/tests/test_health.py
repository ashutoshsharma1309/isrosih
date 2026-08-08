"""Smoke tests for the API skeleton — verify the app boots and the contract holds."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "VARUNA AI"


def test_prediction_returns_501_until_model_deployed():
    """The API must refuse to fabricate predictions before a model exists."""
    response = client.post(
        "/api/v1/predictions",
        json={"location": {"latitude": 10.0, "longitude": 76.3}, "region_name": "Kerala"},
    )
    assert response.status_code == 501


def test_alerts_list_is_empty_initially():
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    assert response.json() == []
