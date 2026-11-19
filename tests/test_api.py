"""HTTP tests. Requires FastAPI + httpx (pip install -r requirements-dev.txt)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag.main import create_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(create_app())


def test_health(client):
    body = client.get("/health").json()
    assert body["chunks"] > 0 and body["llm"] is False


def test_search(client):
    hits = client.get("/search", params={"q": "lahar detection"}).json()
    assert hits[0]["doc"] == "lahars"


def test_ask(client):
    body = client.post("/ask", json={"question": "Where is CVO located?"}).json()
    assert body["citations"] and body["retrieved"]
    assert client.post("/ask", json={"question": ""}).status_code == 422
