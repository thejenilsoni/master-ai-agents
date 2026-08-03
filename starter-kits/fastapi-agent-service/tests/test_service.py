"""Service tests that run with no provider account.

Every test here drives the real app through FastAPI's TestClient. The service
falls back to its built-in stub model whenever no provider key is configured, so
the whole suite runs offline and in CI without secrets.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the app package from the kit root regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient bound to a freshly configured app.

    Settings are read from the environment at construction time, so the env is
    set *before* the app factory runs. A generous rate limit keeps unrelated
    tests from tripping it; the limiter has its own test below.
    """
    for key in list(os.environ):
        if key.startswith("AGENT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_API_KEYS", API_KEY)
    monkeypatch.setenv("AGENT_RATE_LIMIT_REQUESTS", "1000")
    monkeypatch.setenv("AGENT_ENVIRONMENT", "test")
    monkeypatch.delenv("AGENT_OPENAI_API_KEY", raising=False)  # force the stub model

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()  # settings are cached; drop the memo between tests
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_healthz_needs_no_auth(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_readiness(client: TestClient) -> None:
    response = client.get("/readyz")
    # Readiness is a separate signal from liveness; it may legitimately be 200 or
    # 503, but it must answer with a JSON body either way.
    assert response.status_code in (200, 503)
    assert "status" in response.json()


def test_chat_requires_an_api_key(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 401


def test_chat_rejects_a_wrong_api_key(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_chat_returns_a_reply_with_the_stub_model(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "hello there"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["request_id"]
    assert body["latency_ms"] >= 0
    assert "input_tokens" in body["usage"]


def test_request_id_is_returned_as_a_header(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.headers.get("X-Request-ID")


def test_empty_message_is_rejected_by_validation(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""}, headers=HEADERS)
    assert response.status_code == 422


def test_oversized_message_is_rejected(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "x" * 9000}, headers=HEADERS)
    assert response.status_code == 422


def test_streaming_endpoint_emits_server_sent_events(client: TestClient) -> None:
    response = client.post("/chat/stream", json={"message": "stream please"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data:" in response.text


def test_rate_limiter_trips_and_sets_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limiter is exercised with its own app so the shared fixture stays generous."""
    for key in list(os.environ):
        if key.startswith("AGENT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_API_KEYS", API_KEY)
    monkeypatch.setenv("AGENT_RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("AGENT_RATE_LIMIT_WINDOW_S", "60")

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()) as limited:
        payload = {"message": "hi"}
        first = limited.post("/chat", json=payload, headers=HEADERS)
        second = limited.post("/chat", json=payload, headers=HEADERS)
        third = limited.post("/chat", json=payload, headers=HEADERS)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.headers.get("Retry-After")
    get_settings.cache_clear()
