import pytest
from fastapi.testclient import TestClient

from app import main


class FakeModel:
    def predict(self, df):
        return [123.4]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    return TestClient(main.app)


VALID = {
    "rooms": 4, "members": 5, "ac_hours_per_day": 6.0, "avg_temp_c": 32.0,
    "house_area_sqft": 1400.0, "solar_kw": 0.0,
    "connection": "domestic", "feeder": "Feeder-1",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_ac_hours_above_25_returns_422(client):
    r = client.post("/predict", json={**VALID, "ac_hours_per_day": 26})
    assert r.status_code == 422


def test_temp_below_4_returns_422(client):
    r = client.post("/predict", json={**VALID, "avg_temp_c": 3})
    assert r.status_code == 422