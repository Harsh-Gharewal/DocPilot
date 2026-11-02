"""Unit and integration tests for DocPilot FastAPI REST service."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import settings

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": settings.api_key}


def test_root_endpoint():
    """Verify root discovery endpoint returns service info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "DocPilot RAG Engine"
    assert "/docs" in data["docs_url"]


def test_health_endpoint():
    """Verify health diagnostics endpoint returns system statuses."""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["ok", "degraded"]
    assert "chromadb" in data
    assert "bm25" in data
    assert "embedding_provider" in data
    assert "llm_provider" in data
    assert isinstance(data["strategies_available"], list)


def test_ask_endpoint_auth_required():
    """Verify /v1/ask rejects requests without API key."""
    resp = client.post("/v1/ask", json={"question": "What is the policy?"})
    assert resp.status_code == 401
    assert "Missing required X-API-Key header" in resp.json()["detail"]


def test_ask_endpoint_auth_invalid():
    """Verify /v1/ask rejects requests with an invalid API key."""
    resp = client.post(
        "/v1/ask",
        json={"question": "What is the policy?"},
        headers={"X-API-Key": "wrong-secret-key"},
    )
    assert resp.status_code == 403
    assert "Invalid or unauthorized API key" in resp.json()["detail"]


def test_ask_endpoint_success():
    """Verify /v1/ask produces structured AnswerResponse with valid credentials."""
    resp = client.post(
        "/v1/ask",
        json={
            "question": "What is the salary limit?",
            "strategy": "structure_aware",
            "mode": "sparse_only",
            "top_k": 3,
        },
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "confidence" in data
    assert "latency_by_stage" in data
    assert "retrieval_trace" in data


def test_list_documents_endpoint():
    """Verify /v1/documents returns registered document collection."""
    resp = client.get("/v1/documents", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "total_count" in data
    assert isinstance(data["documents"], list)


def test_ingest_endpoint_and_status():
    """Verify /v1/ingest queues asynchronous processing and reports job status."""
    resp = client.post(
        "/v1/ingest?folder_path=sample_docs&strategy=fixed_size",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in ["pending", "processing", "completed"]

    job_id = data["job_id"]
    status_resp = client.get(f"/v1/ingest/{job_id}", headers=AUTH_HEADERS)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id

