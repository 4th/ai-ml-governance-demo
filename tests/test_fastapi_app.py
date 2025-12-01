# tests
# tests/test_fastapi_app.py
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.fastapi_app import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
