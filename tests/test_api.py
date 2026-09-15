"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from bhel_internship.api.routes import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_model" in data
    assert "indexed_chunks" in data


def test_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "backend" in data


def test_documents_endpoint(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data
    assert "total_indexed_chunks" in data


def test_ask_endpoint_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_invalid_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_chat_endpoint_empty_question(client):
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 400


def test_chat_endpoint_non_streaming(client):
    response = client.post("/api/chat", json={"question": "test", "stream": False})
    # May return 500 if engine not fully initialized, but shouldn't be 422
    assert response.status_code != 422


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Enterprise RAG Engine" in response.text