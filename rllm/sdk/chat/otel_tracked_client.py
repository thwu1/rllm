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

from rllm.sdk.chat.util import merge_args
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url


class _ScopedClientMixin:
    """Shared helpers for sync/async OTel variants."""

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


# =============================================================================
# Sync Client Implementation
# =============================================================================


@dataclass
class _OTelChatCompletions:
    parent: "OpenTelemetryTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        response = scoped_client.chat.completions.create(**call_kwargs)
        return response


@dataclass
class _OTelChatNamespace:
    parent: "OpenTelemetryTrackedChatClient"

    @property
    def completions(self) -> _OTelChatCompletions:
        return _OTelChatCompletions(self.parent)


@dataclass
class _OTelCompletions:
    parent: "OpenTelemetryTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        response = scoped_client.completions.create(**call_kwargs)
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


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _OTelAsyncChatCompletions:
    parent: "OpenTelemetryTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        response = await scoped_client.chat.completions.create(**call_kwargs)
        return response


@dataclass
class _OTelAsyncChatNamespace:
    parent: "OpenTelemetryTrackedAsyncChatClient"

    @property
    def completions(self) -> _OTelAsyncChatCompletions:
        return _OTelAsyncChatCompletions(self.parent)


@dataclass
class _OTelAsyncCompletions:
    parent: "OpenTelemetryTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata) if self.parent.use_proxy else None
        scoped_client = self.parent._scoped_client(routing_metadata)

        response = await scoped_client.completions.create(**call_kwargs)
        return response


class OpenTelemetryTrackedAsyncChatClient(_ScopedClientMixin):
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
