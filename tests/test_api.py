"""Contract tests for the FastAPI prediction service."""

from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

from crop_yield import api


def fake_artifacts():
    model = Mock()
    model.predict.return_value = np.array([24500.0])
    metadata = {
        "model_version": "test",
        "countries": ["Ghana"],
        "crops": ["Maize"],
        "limitations": [],
    }
    return model, metadata


def test_prediction_contract(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_artifacts", fake_artifacts)
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        json={
            "area": "Ghana",
            "item": "Maize",
            "year": 2013,
            "average_rainfall_mm_per_year": 1187,
            "pesticides_tonnes": 100,
            "average_temperature_c": 26,
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction_hg_per_ha"] == 24500
    assert response.json()["prediction_tonnes_per_ha"] == 2.45


def test_unknown_crop_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_artifacts", fake_artifacts)
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        json={
            "area": "Ghana",
            "item": "Cassava",
            "year": 2013,
            "average_rainfall_mm_per_year": 1187,
            "pesticides_tonnes": 100,
            "average_temperature_c": 26,
        },
    )

    assert response.status_code == 422
