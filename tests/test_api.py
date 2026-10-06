import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.features.preprocessing import CATEGORICAL_FEATURES


class FakePredictor:
    classes_ = np.array([0, 1, 2])

    def __init__(self):
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        fatal = 0.1 + X["speed_limit"].fillna(0).to_numpy(dtype=float) / 1000
        serious = np.full(len(X), 0.2)
        slight = 1 - fatal - serious
        return np.column_stack([fatal, serious, slight])


@pytest.fixture
def client(tmp_path):
    codebook = {field: {-1: "Data missing or out of range", 1: f"{field} one", 2: f"{field} two", 9: "Unknown"}
                for field in CATEGORICAL_FEATURES}
    codebook["junction_detail_historic"].update({0: "Outside junction", 3: "T junction"})
    codebook["junction_control"].update({0: "Outside junction", 4: "Give way"})
    model_path = tmp_path / "fake_model.joblib"
    joblib.dump(FakePredictor(), model_path)
    reference = {field: 1 for field in CATEGORICAL_FEATURES}
    reference.update(date="01/07/2019", time="12:00", speed_limit=30)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    application = create_app(model_path=model_path, reference_path=reference_path, codebook=codebook)
    with TestClient(application) as test_client:
        yield test_client


def sample_payload(client):
    options = client.get("/options").json()["categories"]
    values = {field: items[0]["value"] for field, items in options.items()}
    values.update(date="2020-06-15", time="22:05", speed_limit=30,
                  junction_detail_historic=3, junction_control=4)
    return values


def test_health_model_loads_once_and_options_are_available(client):
    assert client.get("/health").json()["model_loaded"] is True
    result = client.get("/options")
    assert result.status_code == 200
    assert result.json()["speed_limit_mph"] == [20, 30, 40, 50, 60, 70]
    loaded = client.app.state.model
    client.get("/health")
    assert client.app.state.model is loaded


def test_predict_severity_returns_scores_and_explanations(client):
    response = client.post("/predict-severity", json=sample_payload(client))
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_severity"] in {"Fatal", "Serious", "Slight"}
    assert sum(result["class_scores"].values()) == pytest.approx(1)
    assert len(result["important_factors"]) == 4
    assert "not interpret" in result["score_calibration"]
    assert "risk" not in result


def test_predict_alias_and_request_validation(client):
    payload = sample_payload(client)
    assert client.post("/predict", json=payload).status_code == 200
    payload["date"] = "31/02/2020"
    assert client.post("/predict-severity", json=payload).status_code == 422
    payload = sample_payload(client)
    payload["weather_conditions"] = 987
    assert client.post("/predict-severity", json=payload).status_code == 422
    payload["unexpected"] = "value"
    assert client.post("/predict-severity", json=payload).status_code == 422
