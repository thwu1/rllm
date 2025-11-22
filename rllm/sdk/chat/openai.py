"""Unified OpenAI chat clients with session tracking and optional proxy support.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    TrackedChatClient                            │
    │  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
    │  │  use_proxy    │  │ enable_local_    │  │    tracer       │  │
    │  │  (default: T) │  │ tracing (def: T) │  │  (optional)     │  │
    │  └───────────────┘  └──────────────────┘  └─────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          ┌──────────────┐        ┌──────────────┐
          │   Default    │        │  OTel Mode   │
          │  (proxy+log) │        │ (proxy only) │
          └──────────────┘        └──────────────┘
           use_proxy=True          use_proxy=True
           local_trace=True        local_trace=False

Aliases (backward compatible):
- OpenTelemetryTrackedChatClient = TrackedChatClient(enable_local_tracing=False)
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.chat.util import extract_completion_tokens, extract_usage_tokens, merge_args
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer

# Shared tracer instance for all clients (when no custom tracer provided)
_SHARED_TRACER = InMemorySessionTracer()


def _get_scoped_client(client, base_url: str | None, metadata: dict | None, headers: dict | None):
    """Apply proxy URL rewriting and extra headers to client."""
    if base_url and metadata:
        client = client.with_options(base_url=build_proxied_base_url(base_url, metadata))
    if headers:
        client = client.with_options(extra_headers=headers)
    return client


def _log_trace(
    tracer,
    *,
    model: str,
    messages: list[dict],
    response: dict,
    token_ids: list[int] | None,
    metadata: dict,
    latency_ms: float,
) -> None:
    """Log LLM call to tracer."""
    if not tracer:
        return

    ctx_metadata = {**get_current_metadata(), **metadata}
    if token_ids:
        ctx_metadata["token_ids"] = {"prompt": [], "completion": token_ids}

    tracer.log_llm_call(
        name="chat.completions.create",
        model=model,
        input={"messages": messages},
        output=response,
        metadata=ctx_metadata,
        latency_ms=latency_ms,
        tokens=extract_usage_tokens(response),
        trace_id=response.get("id"),
        session_name=get_current_session_name(),
        session_uids=get_active_session_uids(),
        sessions=get_active_cv_sessions(),
    )


# =============================================================================
# Sync Implementation
# =============================================================================


@dataclass
class _ChatCompletions:
    """Namespace for chat.completions.create()"""

    parent: "TrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        # Get client (with proxy URL if enabled)
        if p.use_proxy:
            client = _get_scoped_client(
                p._client, p.base_url, assemble_routing_metadata(metadata), p._headers
            )
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        # Log trace (if enabled)
        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _Completions:
    """Namespace for completions.create()"""

    parent: "TrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        if p.use_proxy:
            client = _get_scoped_client(
                p._client, p.base_url, assemble_routing_metadata(metadata), p._headers
            )
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _ChatNamespace:
    parent: "TrackedChatClient"

    @property
    def completions(self) -> _ChatCompletions:
        return _ChatCompletions(self.parent)


class TrackedChatClient:
    """OpenAI client wrapper with proxy support and tracing.

    Args:
        client: Pre-configured OpenAI client (if None, creates one with **kwargs)
        use_proxy: Inject metadata into proxy URL (default: True)
        enable_local_tracing: Log traces locally (default: True)
        tracer: Custom tracer (default: shared in-memory tracer)
        **kwargs: Passed directly to OpenAI() if client not provided
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        **kwargs: Any,
    ) -> None:
        # Use provided client or create one (let OpenAI handle all its own args)
        self._client = client if client is not None else OpenAI(**kwargs)

        # Resolve base_url for proxy routing from the client
        # Skip default OpenAI URL (would break if we tried to proxy-rewrite it)
        client_url = getattr(self._client, "base_url", None)
        if client_url and str(client_url).rstrip("/") != "https://api.openai.com/v1":
            self.base_url = str(client_url)
        else:
            self.base_url = None

        self.use_proxy = use_proxy
        self.enable_local_tracing = enable_local_tracing
        self._tracer = tracer
        self._headers: dict[str, str] = {}

        self.chat = _ChatNamespace(self)
        self.completions = _Completions(self)


# =============================================================================
# Async Implementation
# =============================================================================


@dataclass
class _AsyncChatCompletions:
    parent: "TrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        if p.use_proxy:
            client = _get_scoped_client(
                p._client, p.base_url, assemble_routing_metadata(metadata), p._headers
            )
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = await client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=messages,
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _AsyncCompletions:
    parent: "TrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        p = self.parent
        call_kwargs = merge_args(args, kwargs)

        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt required")

        model = call_kwargs.get("model")  # Let OpenAI client handle model validation
        metadata = call_kwargs.pop("metadata", None) or {}

        if p.use_proxy:
            client = _get_scoped_client(
                p._client, p.base_url, assemble_routing_metadata(metadata), p._headers
            )
        else:
            client = _get_scoped_client(p._client, None, None, p._headers)

        start = time.perf_counter()
        response = await client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        if p.enable_local_tracing:
            resp_dict = response.model_dump()
            _log_trace(
                p._tracer or _SHARED_TRACER,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response=resp_dict,
                token_ids=extract_completion_tokens(resp_dict),
                metadata=metadata,
                latency_ms=latency_ms,
            )

        return response


@dataclass
class _AsyncChatNamespace:
    parent: "TrackedAsyncChatClient"

    @property
    def completions(self) -> _AsyncChatCompletions:
        return _AsyncChatCompletions(self.parent)


class TrackedAsyncChatClient:
    """Async OpenAI client wrapper with proxy support and tracing."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        use_proxy: bool = True,
        enable_local_tracing: bool = True,
        tracer: Any = None,
        **kwargs: Any,
    ) -> None:
        # Use provided client or create one (let AsyncOpenAI handle all its own args)
        self._client = client if client is not None else AsyncOpenAI(**kwargs)

        # Resolve base_url for proxy routing from the client
        # Skip default OpenAI URL (would break if we tried to proxy-rewrite it)
        client_url = getattr(self._client, "base_url", None)
        if client_url and str(client_url).rstrip("/") != "https://api.openai.com/v1":
            self.base_url = str(client_url)
        else:
            self.base_url = None

        self.use_proxy = use_proxy
        self.enable_local_tracing = enable_local_tracing
        self._tracer = tracer
        self._headers: dict[str, str] = {}

        self.chat = _AsyncChatNamespace(self)
        self.completions = _AsyncCompletions(self)


# =============================================================================
# Backward-compatible Aliases
# =============================================================================
# These are simple subclasses that set sensible defaults for common use cases.
# No logic changes - just preset configurations.


class ProxyTrackedChatClient(TrackedChatClient):
    """Alias: TrackedChatClient with defaults (use_proxy=True, local_tracing=True)"""

    pass


class ProxyTrackedAsyncChatClient(TrackedAsyncChatClient):
    """Alias: TrackedAsyncChatClient with defaults"""

    pass


class OpenTelemetryTrackedChatClient(TrackedChatClient):
    """Alias: TrackedChatClient with enable_local_tracing=False (OTel mode)"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_local_tracing=False, **kwargs)


class OpenTelemetryTrackedAsyncChatClient(TrackedAsyncChatClient):
    """Alias: TrackedAsyncChatClient with enable_local_tracing=False"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(enable_local_tracing=False, **kwargs)


# Legacy shorthand names
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = OpenTelemetryTrackedAsyncChatClient
