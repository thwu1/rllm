"""Unified proxy-aware OpenAI chat clients that inject metadata slugs per request.

This client injects session metadata into the proxy URL for server-side tracing.
It supports optional local in-memory tracing for immediate access to LLM calls.

When `enable_local_tracing=True` (default), traces are logged to an in-memory store
for access via session.llm_calls. When `enable_local_tracing=False`, the client
relies entirely on the proxy/backend for tracing (suitable for OTel-based setups).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.chat.util import (
    extract_completion_tokens,
    extract_usage_tokens,
    merge_args,
)
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer


class _ScopedClientMixin:
    """Shared helpers for proxy client scoping and in-memory tracing."""

    # Shared in-memory session tracer instance for all proxy clients
    _memory_tracer = InMemorySessionTracer()

    _base_headers: dict[str, str]
    base_url: str | None
    use_proxy: bool
    enable_local_tracing: bool

    def _scoped_client(self, metadata: dict[str, Any] | None):
        client = self._client

        base_url = getattr(self, "base_url", None)
        if self.use_proxy and base_url and metadata:
            proxied_base = build_proxied_base_url(base_url, metadata)
            client = client.with_options(base_url=proxied_base)

        if self._base_headers:
            client = client.with_options(extra_headers=self._base_headers)

        return client

    def _log_trace(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_payload: dict[str, Any],
        completion_token_ids: list[int] | None,
        metadata_overrides: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Log trace to in-memory session (if active and local tracing is enabled).

        In proxy mode, the proxy handles persistent tracing to the backend.
        When enable_local_tracing=True, we also log to in-memory session for
        immediate access via session.llm_calls without any I/O.

        When enable_local_tracing=False (OTel mode), we skip local tracing and
        rely entirely on the proxy/OTel backend for tracing.
        """
        if not self.enable_local_tracing:
            return

        context_metadata = get_current_metadata()
        merged_metadata = {**context_metadata, **(dict(metadata_overrides) if metadata_overrides else {})}

        session_name = get_current_session_name()
        tokens = extract_usage_tokens(response_payload)
        trace_id = response_payload.get("id")
        session_uids = get_active_session_uids()
        sessions = get_active_cv_sessions()

        self._memory_tracer.log_llm_call(
            name="proxy.chat.completions.create",
            input={"messages": messages},
            output=response_payload,
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            metadata=merged_metadata,
            trace_id=trace_id,
            session_name=session_name,
            session_uids=session_uids,
            sessions=sessions,
        )


# =============================================================================
# Sync Client Implementation
# =============================================================================


@dataclass
class _ProxyChatCompletions:
    parent: "ProxyTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


@dataclass
class _ProxyChatNamespace:
    parent: "ProxyTrackedChatClient"

    @property
    def completions(self) -> _ProxyChatCompletions:
        return _ProxyChatCompletions(self.parent)


@dataclass
class _ProxyCompletions:
    parent: "ProxyTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


class ProxyTrackedChatClient(_ScopedClientMixin):
    """OpenAI client wrapper that injects metadata slugs into the proxy base URL.

    This unified client supports both proxy-based tracing and optional local
    in-memory tracing for immediate access to LLM call data.

    Args:
        api_key: OpenAI API key
        base_url: Base URL for the API (typically the proxy URL)
        default_model: Default model to use if not specified in calls
        client: Pre-configured OpenAI client instance
        use_proxy: Whether to inject metadata into proxy URL (default: True)
        extra_headers: Additional headers to include in requests
        enable_local_tracing: Whether to log traces locally (default: True).
            Set to False for OTel-only mode where tracing is handled by backend.
        **client_kwargs: Additional arguments passed to OpenAI client
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
        use_proxy: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        enable_local_tracing: bool = True,
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

        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})
        self.enable_local_tracing = enable_local_tracing

        self.chat = _ProxyChatNamespace(self)
        self.completions = _ProxyCompletions(self)


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _ProxyAsyncChatCompletions:
    parent: "ProxyTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = await scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


@dataclass
class _ProxyAsyncChatNamespace:
    parent: "ProxyTrackedAsyncChatClient"

    @property
    def completions(self) -> _ProxyAsyncChatCompletions:
        return _ProxyAsyncChatCompletions(self.parent)


@dataclass
class _ProxyAsyncCompletions:
    parent: "ProxyTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = await scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


class ProxyTrackedAsyncChatClient(_ScopedClientMixin):
    """Async variant of the proxy-aware chat client.

    This unified client supports both proxy-based tracing and optional local
    in-memory tracing for immediate access to LLM call data.

    Args:
        api_key: OpenAI API key
        base_url: Base URL for the API (typically the proxy URL)
        default_model: Default model to use if not specified in calls
        client: Pre-configured AsyncOpenAI client instance
        use_proxy: Whether to inject metadata into proxy URL (default: True)
        extra_headers: Additional headers to include in requests
        enable_local_tracing: Whether to log traces locally (default: True).
            Set to False for OTel-only mode where tracing is handled by backend.
        **client_kwargs: Additional arguments passed to AsyncOpenAI client
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: AsyncOpenAI | None = None,
        use_proxy: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        enable_local_tracing: bool = True,
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

        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})
        self.enable_local_tracing = enable_local_tracing

        self.chat = _ProxyAsyncChatNamespace(self)
        self.completions = _ProxyAsyncCompletions(self)


# =============================================================================
# Backward-compatible aliases for OTel clients
# =============================================================================
# These aliases provide the same interface as the former OpenTelemetryTrackedChatClient
# but with enable_local_tracing=False by default (OTel mode).


class OpenTelemetryTrackedChatClient(ProxyTrackedChatClient):
    """Backward-compatible alias for OTel-mode sync client.

    This is equivalent to ProxyTrackedChatClient with enable_local_tracing=False.
    In this mode, local tracing is disabled and the client relies entirely on
    the proxy/OTel backend for tracing.
    """

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
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            use_proxy=use_proxy,
            extra_headers=extra_headers,
            enable_local_tracing=False,
            **client_kwargs,
        )


class OpenTelemetryTrackedAsyncChatClient(ProxyTrackedAsyncChatClient):
    """Backward-compatible alias for OTel-mode async client.

    This is equivalent to ProxyTrackedAsyncChatClient with enable_local_tracing=False.
    In this mode, local tracing is disabled and the client relies entirely on
    the proxy/OTel backend for tracing.
    """

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
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            use_proxy=use_proxy,
            extra_headers=extra_headers,
            enable_local_tracing=False,
            **client_kwargs,
        )


# Legacy shorthand names
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = OpenTelemetryTrackedAsyncChatClient
