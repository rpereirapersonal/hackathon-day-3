"""Response-contract tests — AC-1, AC-2, AC-3.

Covers FR-1.x and NFR-4: the shape of every response, on every path.

These drive the real app with the model gateway pointed at a closed port, which
is the honest version of the scored scenario: the organizers' harness has no
obligation to find a working gateway, and every one of these paths must still
return a valid body.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

# A port nothing listens on, so every model call fails fast.
_DEAD_GATEWAY = "http://127.0.0.1:9/v1"


@pytest.fixture
def client(monkeypatch):
    """The real app, with the model gateway deliberately unreachable."""
    for name in ("AGENT_BRAIN_BASE_URL", "DOMAIN_FT_BASE_URL"):
        monkeypatch.setenv(name, _DEAD_GATEWAY)
    for name, value in (
        ("AGENT_BRAIN_MODEL", "brain"),
        ("DOMAIN_FT_MODEL", "domain"),
        ("AGENT_BRAIN_API_KEY", "EMPTY"),
        ("DOMAIN_FT_API_KEY", "EMPTY"),
        ("DOMAIN_PREDICT_MODE", "llm"),
    ):
        monkeypatch.setenv(name, value)

    # Imported inside the fixture so the patched environment is what the module
    # reads at import time.
    import importlib

    from src import api

    importlib.reload(api)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_is_fast_and_ok_with_a_dead_gateway(client):
    """The hard gate never depends on an upstream (AC-1, NFR-4.1)."""
    started = time.monotonic()
    response = client.get("/health")
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert elapsed < 1.0


def test_health_echoes_the_predict_mode(client):
    """A submission left in `mock` must be visible at a glance (AC-12)."""
    assert response_mode(client) in {"llm", "mock"}


def response_mode(client) -> str:
    return client.get("/health").json()["domain_predict_mode"]


def test_query_returns_the_contract_shape(client):
    """AC-2: non-empty `answer`, integer `steps`, list `tool_trace`."""
    response = client.post("/query", json={"question": "What was the cash rate?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "steps", "tool_trace"}
    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert isinstance(body["steps"], int)
    assert isinstance(body["tool_trace"], list)


def test_unreachable_gateway_still_yields_a_valid_answer(client):
    """AC-3: a dead gateway degrades, it does not fail (FR-1.6, CON-5)."""
    body = client.post("/query", json={"question": "What was the cash rate?"}).json()
    assert body["answer"].strip()
    # A stated limitation, not an invented figure.
    assert "4.35" not in body["answer"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"wrong_field": "x"},
        {"question": None},
        [],
        "not json at all",
    ],
)
def test_malformed_input_never_returns_an_error_status(client, payload):
    """AC-3: a 422 is a malformed-contract response on a scored request."""
    response = (
        client.post("/query", json=payload)
        if not isinstance(payload, str)
        else client.post("/query", content=payload)
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "steps", "tool_trace"}
    assert body["answer"].strip()


def test_empty_question_is_answered_not_rejected(client):
    """An empty string is well-formed input; it must not 4xx."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 200
    assert response.json()["answer"].strip()
