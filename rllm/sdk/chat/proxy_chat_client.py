"""Proxy-aware OpenAI chat clients that inject metadata slugs per request.

This client injects session metadata into the proxy URL for server-side tracing.
"""

from __future__ import annotations

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
    extract_completion_tokens,
    extract_usage_tokens,
)
from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer


class _ProxyClientMixin:
    """Mixin providing proxy URL scoping and in-memory tracing for proxy clients."""

    # Shared in-memory session tracer instance for all proxy clients
    _memory_tracer = InMemorySessionTracer()

    _client: Any
    base_url: str | None
    use_proxy: bool

    def _scoped_client(self, metadata: dict[str, Any] | None):
        """Get a client scoped to the proxied base URL with metadata slug.

        Args:
            metadata: Metadata to encode in the proxy URL slug

        Returns:
            Client instance with modified base_url if proxy mode enabled
        """
        if not self.base_url or not metadata:
            return self._client
        proxied_base = build_proxied_base_url(self.base_url, metadata)
        return self._client.with_options(base_url=proxied_base)

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
        """Log trace to in-memory session (if active).

        In proxy mode, the proxy handles persistent tracing to the backend.
        However, we still log to in-memory session for immediate access via
        session.llm_calls without any I/O.

        Args:
            model: Model identifier used for the call
            messages: Input messages sent to the model
            response_payload: Full response dict from the API
            completion_token_ids: Token IDs from response (if available)
            metadata_overrides: Call-specific metadata to merge
            latency_ms: Latency of the API call in milliseconds
        """
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
class _ProxyChatCompletions(ChatCompletionsBase):
    """Chat completions namespace for ProxyTrackedChatClient."""

    parent: "ProxyTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion with proxy metadata injection."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        messages = call_kwargs["messages"]

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        with TimedCall() as timer:
            response = scoped_client.chat.completions.create(**call_kwargs)

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=timer.latency_ms,
        )

        return response


@dataclass
class _ProxyChatNamespace:
    """Chat namespace for ProxyTrackedChatClient."""

    parent: "ProxyTrackedChatClient"

    @property
    def completions(self) -> _ProxyChatCompletions:
        return _ProxyChatCompletions(self.parent)


@dataclass
class _ProxyCompletions(CompletionsBase):
    """Completions namespace for ProxyTrackedChatClient."""

    parent: "ProxyTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion with proxy metadata injection."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        prompt = call_kwargs["prompt"]

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        with TimedCall() as timer:
            response = scoped_client.completions.create(**call_kwargs)

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=timer.latency_ms,
        )

        return response


class ProxyTrackedChatClient(_ProxyClientMixin, BaseChatClient[OpenAI]):
    """OpenAI client wrapper that injects metadata slugs into the proxy base URL."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
        use_proxy: bool = True,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the proxy-aware chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the proxy endpoint
            tracer: Deprecated, kept for compatibility (proxy handles tracing)
            default_model: Default model to use
            client: Pre-configured OpenAI client
            use_proxy: Enable proxy metadata injection (default: True)
            **client_kwargs: Additional client configuration
        """
        self.tracer = tracer  # Kept for compatibility but not used
        self.use_proxy = use_proxy
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

    def _create_chat_namespace(self) -> _ProxyChatNamespace:
        return _ProxyChatNamespace(self)

    def _create_completions_namespace(self) -> _ProxyCompletions:
        return _ProxyCompletions(self)


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _ProxyAsyncChatCompletions(ChatCompletionsBase):
    """Async chat completions namespace for ProxyTrackedAsyncChatClient."""

    parent: "ProxyTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion with proxy metadata injection."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        messages = call_kwargs["messages"]

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        with TimedCall() as timer:
            response = await scoped_client.chat.completions.create(**call_kwargs)

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=timer.latency_ms,
        )

        return response


@dataclass
class _ProxyAsyncChatNamespace:
    """Async chat namespace for ProxyTrackedAsyncChatClient."""

    parent: "ProxyTrackedAsyncChatClient"

    @property
    def completions(self) -> _ProxyAsyncChatCompletions:
        return _ProxyAsyncChatCompletions(self.parent)


@dataclass
class _ProxyAsyncCompletions(CompletionsBase):
    """Async completions namespace for ProxyTrackedAsyncChatClient."""

    parent: "ProxyTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion with proxy metadata injection."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        prompt = call_kwargs["prompt"]

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            scoped_client = self.parent._client

        with TimedCall() as timer:
            response = await scoped_client.completions.create(**call_kwargs)

        response_dict = response.model_dump()
        completion_token_ids = extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=timer.latency_ms,
        )

        return response


class ProxyTrackedAsyncChatClient(_ProxyClientMixin, BaseAsyncChatClient[AsyncOpenAI]):
    """Async variant of the proxy-aware chat client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: AsyncOpenAI | None = None,
        use_proxy: bool = True,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the async proxy-aware chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the proxy endpoint
            tracer: Deprecated, kept for compatibility (proxy handles tracing)
            default_model: Default model to use
            client: Pre-configured AsyncOpenAI client
            use_proxy: Enable proxy metadata injection (default: True)
            **client_kwargs: Additional client configuration
        """
        self.tracer = tracer  # Kept for compatibility but not used
        self.use_proxy = use_proxy
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

    def _create_chat_namespace(self) -> _ProxyAsyncChatNamespace:
        return _ProxyAsyncChatNamespace(self)

    def _create_completions_namespace(self) -> _ProxyAsyncCompletions:
        return _ProxyAsyncCompletions(self)
