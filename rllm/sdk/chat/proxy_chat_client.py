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
    TimedCall,
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

    def _scoped_client(self, metadata: dict[str, Any] | None):
        base_url = getattr(self, "base_url", None)
        if not base_url or not metadata:
            return self._client
        proxied_base = build_proxied_base_url(base_url, metadata)
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


class ProxyTrackedChatClient(_ScopedClientMixin):
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
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = OpenAI(**init_kwargs)

        self.tracer = tracer  # Kept for compatibility but not used
        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy

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


class ProxyTrackedAsyncChatClient(_ScopedClientMixin):
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
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**init_kwargs)

        self.tracer = tracer  # Kept for compatibility but not used
        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy

        self.chat = _ProxyAsyncChatNamespace(self)
        self.completions = _ProxyAsyncCompletions(self)
