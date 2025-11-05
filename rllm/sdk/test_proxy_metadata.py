"""Unit tests for proxy metadata slug helpers."""

from __future__ import annotations

from rllm.sdk.proxy.metadata_slug import (
    assemble_routing_metadata,
    build_proxied_base_url,
    decode_metadata_slug,
    encode_metadata_slug,
    extract_metadata_from_path,
)
from rllm.sdk.session import SessionContext


def test_encode_decode_roundtrip():
    payload = {"session_id": "sess-123", "job": "nightly", "split": "validation"}
    slug = encode_metadata_slug(payload)
    assert slug.startswith("rllm1:")
    decoded = decode_metadata_slug(slug)
    assert decoded == payload


def test_build_proxied_base_url_inserts_slug():
    payload = {"session_id": "sess-xyz", "job": "batch"}
    base = "http://proxy-host:4000/v1"
    proxied = build_proxied_base_url(base, payload)
    assert proxied.startswith("http://proxy-host:4000/meta/")
    assert proxied.endswith("/v1")
    slug = proxied.split("/meta/")[1].rsplit("/v1", 1)[0]
    assert decode_metadata_slug(slug) == payload


def test_extract_metadata_from_path_returns_clean_path():
    payload = {"session_id": "sess-456", "job": "adhoc"}
    slug = encode_metadata_slug(payload)
    path = f"/meta/{slug}/v1/chat/completions"
    clean, metadata = extract_metadata_from_path(path)
    assert clean == "/v1/chat/completions"
    assert metadata == payload


def test_assemble_routing_metadata_uses_session_context():
    with SessionContext("sess-nested", job="nightly", run="001"):
        metadata = assemble_routing_metadata({"loc": "worker-1"})
    assert metadata["session_id"] == "sess-nested"
    assert metadata["job"] == "nightly"
    assert metadata["run"] == "001"
    assert metadata["loc"] == "worker-1"
