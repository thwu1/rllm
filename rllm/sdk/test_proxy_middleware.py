"""Tests for MetadataRoutingMiddleware."""

from __future__ import annotations

from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from rllm.sdk.proxy.metadata_slug import encode_metadata_slug
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware


def test_middleware_injects_metadata_into_body():
    """Test that middleware extracts metadata from URL and injects into request body."""
    app = FastAPI()
    app.add_middleware(MetadataRoutingMiddleware)

    seen_payloads = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        payload = await request.json()
        seen_payloads.append(payload)
        return {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }

    client = TestClient(app)

    metadata = {"session_id": "sess-abc", "job": "nightly"}
    slug = encode_metadata_slug(metadata)

    resp = client.post(
        f"/meta/{slug}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": []},
    )

    assert resp.status_code == 200
    assert seen_payloads, "Route should observe mutated request payload"
    payload = seen_payloads[-1]

    # Verify metadata was injected into body
    assert "metadata" in payload
    assert payload["metadata"]["session_id"] == "sess-abc"
    assert payload["metadata"]["job"] == "nightly"


def test_middleware_merges_existing_metadata():
    """Test that middleware merges with existing metadata in request body."""
    app = FastAPI()
    app.add_middleware(MetadataRoutingMiddleware)

    seen_payloads = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        payload = await request.json()
        seen_payloads.append(payload)
        return {"choices": [], "usage": {}}

    client = TestClient(app)

    metadata = {"session_id": "sess-xyz"}
    slug = encode_metadata_slug(metadata)

    resp = client.post(
        f"/meta/{slug}/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [],
            "metadata": {"existing_key": "existing_value"},
        },
    )

    assert resp.status_code == 200
    payload = seen_payloads[-1]

    # Verify both existing and new metadata are present
    assert payload["metadata"]["existing_key"] == "existing_value"
    assert payload["metadata"]["session_id"] == "sess-xyz"


def test_middleware_without_metadata():
    """Test that middleware works when no metadata slug is present."""
    app = FastAPI()
    app.add_middleware(MetadataRoutingMiddleware)

    seen_payloads = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        payload = await request.json()
        seen_payloads.append(payload)
        return {"choices": []}

    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": []},
    )

    assert resp.status_code == 200
    payload = seen_payloads[-1]

    # Original payload should be unchanged
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == []
