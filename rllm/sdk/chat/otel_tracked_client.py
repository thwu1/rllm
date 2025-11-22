"""OpenTelemetry-aware OpenAI chat client that reads session context from baggage.

This client reads session metadata from OTel baggage for cross-process context propagation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.chat.base import (
    BaseAsyncChatClient,
    BaseChatClient,
    ChatCompletionsBase,
    CompletionsBase,
    TimedCall,
)
from rllm.sdk.proxy.metadata_slug import build_proxied_base_url
from rllm.sdk.session.opentelemetry import (
    get_active_otel_session_uids,
    get_current_otel_metadata,
    get_current_otel_session_name,
)


def _build_routing_metadata(user_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build routing metadata from baggage (single source of truth).

    This function reads directly from W3C baggage, which works both:
    - In-process: when inside an otel_session() context
    - Cross-process: when baggage has been propagated via HTTP headers

    Args:
        user_metadata: Optional user-provided metadata to merge (overrides session metadata)

    Returns:
        Dict with session_name, session_uids, and merged metadata
    """
    session_name = get_current_otel_session_name()
    session_uids = get_active_otel_session_uids()
    session_metadata = get_current_otel_metadata()

    result: dict[str, Any] = {}

    if session_metadata:
        result.update(session_metadata)
    if session_name:
        result["session_name"] = session_name
    if session_uids:
        result["session_uids"] = session_uids
    if user_metadata:
        result.update(dict(user_metadata))

    return result


class _OTelClientMixin:
    """Mixin providing OTel-specific client scoping with headers support."""

    _client: Any
    _base_headers: dict[str, str]
    base_url: str | None
    use_proxy: bool

    def _scoped_client(self, metadata: Mapping[str, Any] | None):
        """Get a client scoped to the proxied base URL with metadata slug.

        Args:
            metadata: Metadata to encode in the proxy URL slug

        Returns:
            Client instance with modified base_url and/or headers
        """
        client = self._client

        if self.use_proxy and metadata:
            if not self.base_url:
                raise RuntimeError("base_url must be set when use_proxy=True.")
            proxied_url = build_proxied_base_url(self.base_url, metadata)
            client = client.with_options(base_url=proxied_url)

        if self._base_headers:
            client = client.with_options(extra_headers=self._base_headers)

        return client


# =============================================================================
# Sync Client Implementation
# =============================================================================


@dataclass
class _OTelChatCompletions(ChatCompletionsBase):
    """Chat completions namespace for OpenTelemetryTrackedChatClient."""

    parent: "OpenTelemetryTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion with OTel context propagation."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)

        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        with TimedCall():
            response = scoped_client.chat.completions.create(**call_kwargs)

        return response


@dataclass
class _OTelChatNamespace:
    """Chat namespace for OpenTelemetryTrackedChatClient."""

    parent: "OpenTelemetryTrackedChatClient"

    @property
    def completions(self) -> _OTelChatCompletions:
        return _OTelChatCompletions(self.parent)


@dataclass
class _OTelCompletions(CompletionsBase):
    """Completions namespace for OpenTelemetryTrackedChatClient."""

    parent: "OpenTelemetryTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion with OTel context propagation."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)

        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        with TimedCall():
            response = scoped_client.completions.create(**call_kwargs)

        return response


class OpenTelemetryTrackedChatClient(_OTelClientMixin, BaseChatClient[OpenAI]):
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
        """Initialize the OTel-aware chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the proxy endpoint
            default_model: Default model to use
            client: Pre-configured OpenAI client
            use_proxy: Enable proxy metadata injection (default: True)
            extra_headers: Additional headers to include in requests
            **client_kwargs: Additional client configuration
        """
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

        # Resolve base_url from client if not provided
        if base_url is None and self._client is not None:
            resolved = getattr(self._client, "base_url", None)
            if resolved:
                self.base_url = str(resolved)

    def _create_chat_namespace(self) -> _OTelChatNamespace:
        return _OTelChatNamespace(self)

    def _create_completions_namespace(self) -> _OTelCompletions:
        return _OTelCompletions(self)


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _OTelAsyncChatCompletions(ChatCompletionsBase):
    """Async chat completions namespace for OpenTelemetryTrackedAsyncChatClient."""

    parent: "OpenTelemetryTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion with OTel context propagation."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)

        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        with TimedCall():
            response = await scoped_client.chat.completions.create(**call_kwargs)

        return response


@dataclass
class _OTelAsyncChatNamespace:
    """Async chat namespace for OpenTelemetryTrackedAsyncChatClient."""

    parent: "OpenTelemetryTrackedAsyncChatClient"

    @property
    def completions(self) -> _OTelAsyncChatCompletions:
        return _OTelAsyncChatCompletions(self.parent)


@dataclass
class _OTelAsyncCompletions(CompletionsBase):
    """Async completions namespace for OpenTelemetryTrackedAsyncChatClient."""

    parent: "OpenTelemetryTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion with OTel context propagation."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)

        routing_metadata = _build_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        with TimedCall():
            response = await scoped_client.completions.create(**call_kwargs)

        return response


class OpenTelemetryTrackedAsyncChatClient(_OTelClientMixin, BaseAsyncChatClient[AsyncOpenAI]):
    """Async variant that mirrors OpenTelemetryTrackedChatClient."""

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
        """Initialize the async OTel-aware chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the proxy endpoint
            default_model: Default model to use
            client: Pre-configured AsyncOpenAI client
            use_proxy: Enable proxy metadata injection (default: True)
            extra_headers: Additional headers to include in requests
            **client_kwargs: Additional client configuration
        """
        self.use_proxy = use_proxy
        self._base_headers = dict(extra_headers or {})

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

        # Resolve base_url from client if not provided
        if base_url is None and self._client is not None:
            resolved = getattr(self._client, "base_url", None)
            if resolved:
                self.base_url = str(resolved)

    def _create_chat_namespace(self) -> _OTelAsyncChatNamespace:
        return _OTelAsyncChatNamespace(self)

    def _create_completions_namespace(self) -> _OTelAsyncCompletions:
        return _OTelAsyncCompletions(self)


# Backwards-compatible shorthand names used in docs/design discussions
OpenAIOTelClient = OpenTelemetryTrackedChatClient
AsyncOpenAIOTelClient = OpenTelemetryTrackedAsyncChatClient
