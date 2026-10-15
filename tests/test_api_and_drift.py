"""Tests for FastAPI serving endpoints and drift detection logic."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.drift import check_performance_drift


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Test the /health endpoint returns HTTP 200 and operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "scalers_loaded" in data


def test_forecast_endpoint_structure(client):
    """Test the /forecast endpoint handles valid input schemas."""
    payload = {
        "start_datetime": "2018-06-01T00:00:00Z",
        "horizon_hours": 24,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "point_forecast" in data
    assert "p10_lower" in data
    assert "p90_upper" in data
    assert len(data["point_forecast"]) == 24
    assert len(data["p10_lower"]) == 24
    assert len(data["p90_upper"]) == 24
    # Sanity check: P10 <= Point <= P90
    for p10, pt, p90 in zip(data["p10_lower"], data["point_forecast"], data["p90_upper"]):
        assert p10 <= pt <= p90


def test_performance_drift_detector():
    """Verify performance drift alert triggers when RMSE degrades beyond threshold."""
    np.random.seed(42)
    baseline_rmse = 600.0
    actuals = np.random.normal(loc=28000, scale=3500, size=168)

    # In-control scenario: errors have standard deviation ~ 620 MW (ratio ~ 1.03 < 1.25)
    forecasts_ok = actuals + np.random.normal(loc=0, scale=620.0, size=168)
    drift_ok = check_performance_drift(actuals, forecasts_ok, baseline_rmse=baseline_rmse, threshold_ratio=1.25)
    assert not drift_ok["drift_detected"]
    assert drift_ok["status"] == "NORMAL_OPERATION"

    # Out-of-control scenario: errors spike to 950 MW (ratio ~ 1.58 > 1.25)
    forecasts_drift = actuals + np.random.normal(loc=0, scale=950.0, size=168)
    drift_alert = check_performance_drift(actuals, forecasts_drift, baseline_rmse=baseline_rmse, threshold_ratio=1.25)
    assert drift_alert["drift_detected"] is True
    assert drift_alert["status"] == "DRIFT_ALERT_RETRAIN_RECOMMENDED"
