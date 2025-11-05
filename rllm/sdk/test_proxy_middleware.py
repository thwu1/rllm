"""Tests for MetadataRoutingMiddleware."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient

from rllm.sdk.proxy.metadata_slug import encode_metadata_slug
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware


class DummyTracer:
    def __init__(self) -> None:
        self.calls = []

    def log_llm_call(self, **kwargs):
        self.calls.append(kwargs)


def test_middleware_decodes_metadata_and_logs():
    tracer = DummyTracer()
    app = FastAPI()
    app.add_middleware(MetadataRoutingMiddleware, tracer=tracer)

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
    # Consume body to trigger middleware logging
    _ = resp.json()
    assert resp.status_code == 200
    assert seen_payloads, "Route should observe mutated request payload"
    payload = seen_payloads[-1]
    assert payload.get("logprobs") is True
    assert payload.get("top_logprobs") == 5
    assert payload.get("return_token_ids") is True

    assert tracer.calls, "Tracer should receive a logged call"
    logged = tracer.calls[-1]
    assert logged["metadata"] == metadata
    assert logged["tokens"] == {"prompt": 5, "completion": 7, "total": 12}


def test_streaming_response_accumulates_and_logs():
    tracer = DummyTracer()
    app = FastAPI()
    app.add_middleware(MetadataRoutingMiddleware, tracer=tracer)

    seen_payloads = []

    @app.post("/v1/chat/completions/stream")
    async def completions_stream(request: Request):
        payload = await request.json()
        seen_payloads.append(payload)

        async def gen():
            yield b'{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}'

        return StreamingResponse(gen(), media_type="application/json")

    client = TestClient(app)

    metadata = {"session_id": "sess-stream", "job": "streaming"}
    slug = encode_metadata_slug(metadata)

    resp = client.post(
        f"/meta/{slug}/v1/chat/completions/stream",
        json={"model": "gpt-4o-mini", "messages": [], "stream": True},
    )

    # Consume the streaming body to drive the generator and flush logging
    _ = resp.content

    assert resp.status_code == 200
    assert seen_payloads, "Route should observe mutated request payload"
    payload = seen_payloads[-1]
    assert payload.get("logprobs") is True
    assert payload.get("return_token_ids") is True

    assert tracer.calls, "Tracer should log streamed responses"
    logged = tracer.calls[-1]
    assert logged["metadata"] == metadata
    assert logged["tokens"] == {"prompt": 2, "completion": 3, "total": 5}
