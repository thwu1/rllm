"""OpenTelemetry-aware OpenAI chat client that relies on proxy tracing."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.proxy.metadata_slug import (
    assemble_routing_metadata,
    build_proxied_base_url,
)
from rllm.sdk.session.opentelemetry import OpenTelemetrySession, get_current_otel_session


def _merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if args:
        if len(args) == 1 and isinstance(args[0], Mapping):
            merged = dict(args[0])
            merged.update(kwargs)
            return merged
        raise TypeError("Positional arguments are not supported; pass keyword arguments.")
    return dict(kwargs)


def _current_otel_session() -> OpenTelemetrySession | None:
    return get_current_otel_session()


def _read_baggage_if_no_session(metadata: dict[str, Any]) -> dict[str, Any]:
    """Read session context from OpenTelemetry baggage if no active session exists."""
    try:
        import json

        from opentelemetry import baggage as otel_baggage
        from opentelemetry import trace as otel_trace

        if otel_baggage is None:
            return metadata

        baggage_val = otel_baggage.get_baggage("rllm-session")
        if not baggage_val:
            return metadata

        ctx = json.loads(baggage_val)
        session_uid_chain = ctx.get("session_uid_chain", [])
        session_metadata = ctx.get("metadata", {})

        if session_uid_chain:
            metadata["session_uids"] = session_uid_chain
        # Merge session metadata directly into top-level (includes session_name)
        if session_metadata:
            for key, value in session_metadata.items():
                metadata.setdefault(key, value)

        # Also extract OpenTelemetry trace ID from current span context (propagated via trace headers)
        if otel_trace is not None:
            span = otel_trace.get_current_span()
            if span is not None and span.is_recording():
                span_context = span.get_span_context()
                if span_context.is_valid:
                    trace_id = f"{span_context.trace_id:032x}"
                    metadata.setdefault("otel_trace_id", trace_id)

        return metadata
    except Exception:
        return metadata


def _build_routing_metadata(user_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = assemble_routing_metadata(user_metadata)
    session = _current_otel_session()

    if session is None:
        # No active session - try to read from baggage (for cross-process propagation)
        return _read_baggage_if_no_session(metadata)

    # Active session exists - use its context
    payload = session.to_context_payload()
    metadata.setdefault("session_uid", payload["session_uid"])

    # Add session UIDs for trace association (proxy expects this list)
    # Include all UIDs in the chain for proper hierarchy tracking
    session_uid_chain = payload.get("session_uid_chain", [])
    if session_uid_chain:
        # Overwrite any session_uids from ContextVarSession with our OTel chain
        metadata["session_uids"] = session_uid_chain

    # Merge session metadata directly into top-level (includes session_name)
    if payload.get("metadata"):
        for key, value in payload["metadata"].items():
            metadata.setdefault(key, value)
    if payload.get("trace_id"):
        metadata["otel_trace_id"] = payload["trace_id"]
    if payload.get("span_id"):
        metadata.setdefault("otel_span_id", payload["span_id"])
    return metadata


class _ScopedClientMixin:
    """Shared helpers for sync/async variants."""

    _base_headers: dict[str, str]
    base_url: str | None
    use_proxy: bool

    def _scoped_client(self, metadata: Mapping[str, Any] | None):
        client = self._client

        if self.use_proxy and metadata:
            if not self.base_url:
                raise RuntimeError("base_url must be set when use_proxy=True.")
            proxied_url = build_proxied_base_url(self.base_url, metadata)
            client = client.with_options(base_url=proxied_url)

        if self._base_headers:
            client = client.with_options(extra_headers=self._base_headers)

        return client


@dataclass
class _OTelChatCompletions:
    parent: OpenTelemetryTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = _merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = scoped_client.chat.completions.create(**call_kwargs)
        _ = (time.perf_counter() - start) * 1000  # Latency measured for debugging if needed
        return response


@dataclass
class _OTelChatNamespace:
    parent: OpenTelemetryTrackedChatClient

    @property
    def completions(self) -> _OTelChatCompletions:
        return _OTelChatCompletions(self.parent)


@dataclass
class _OTelCompletions:
    parent: OpenTelemetryTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = _merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = scoped_client.completions.create(**call_kwargs)
        _ = (time.perf_counter() - start) * 1000
        return response


class OpenTelemetryTrackedChatClient(_ScopedClientMixin):
    """OpenAI client wrapper that forwards OTel session metadata to the proxy."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
        use_proxy: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = OpenAI(**init_kwargs)

        resolved_base_url = base_url or getattr(self._client, "base_url", None)
        self.default_model = default_model
        self.base_url = resolved_base_url
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})

        self.chat = _OTelChatNamespace(self)
        self.completions = _OTelCompletions(self)


@dataclass
class _OTelAsyncChatCompletions:
    parent: OpenTelemetryTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = _merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = await scoped_client.chat.completions.create(**call_kwargs)
        _ = (time.perf_counter() - start) * 1000
        return response


@dataclass
class _OTelAsyncChatNamespace:
    parent: OpenTelemetryTrackedAsyncChatClient

    @property
    def completions(self) -> _OTelAsyncChatCompletions:
        return _OTelAsyncChatCompletions(self.parent)


@dataclass
class _OTelAsyncCompletions:
    parent: OpenTelemetryTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = _merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = await scoped_client.completions.create(**call_kwargs)
        _ = (time.perf_counter() - start) * 1000
        return response


class OpenTelemetryTrackedAsyncChatClient(_ScopedClientMixin):
    """Async variant that mirrors `OpenTelemetryTrackedChatClient`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: AsyncOpenAI | None = None,
        use_proxy: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**init_kwargs)

        resolved_base_url = base_url or getattr(self._client, "base_url", None)
        self.default_model = default_model
        self.base_url = resolved_base_url
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})

        self.chat = _OTelAsyncChatNamespace(self)
        self.completions = _OTelAsyncCompletions(self)


# Backwards-compatible shorthand names used in docs/design discussions
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = OpenTelemetryTrackedAsyncChatClient
