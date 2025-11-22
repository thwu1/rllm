"""Proxy-aware OpenAI chat clients that inject metadata slugs per request."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions
from rllm.sdk.tracers import InMemorySessionTracer


def _merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword arguments into a single dict."""
    if args:
        if len(args) == 1 and isinstance(args[0], Mapping):
            merged = dict(args[0])
            merged.update(kwargs)
            return merged
        raise TypeError("Positional arguments are not supported; pass keyword arguments.")
    return dict(kwargs)


def _extract_completion_tokens(response_payload: Mapping[str, Any]) -> list[int] | None:
    """Extract completion token IDs from response payload."""
    choices = response_payload.get("choices") or []
    if not choices:
        return None
    choice0 = choices[0]
    output_ids = choice0.get("output_token_ids")
    if isinstance(output_ids, list):
        return [int(tok) for tok in output_ids]
    logprobs = choice0.get("logprobs")
    if isinstance(logprobs, Mapping):
        token_ids = logprobs.get("token_ids")
        if isinstance(token_ids, list):
            return [int(tok) for tok in token_ids]
    return None


class _ScopedClientMixin:
    # Shared in-memory session tracer instance for all proxy clients
    _memory_tracer = InMemorySessionTracer()

    def _scoped_client(self, metadata: Mapping[str, Any]):
        base_url = getattr(self, "base_url", None)
        if not base_url or not metadata:
            return self._client
        proxied_base = build_proxied_base_url(base_url, metadata)
        return self._client.with_options(base_url=proxied_base)

    def _log_trace(  # type: ignore[override]
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_payload: Mapping[str, Any],
        completion_token_ids: list[int] | None,
        metadata_overrides: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        """
        Log trace to in-memory session (if active).

        In proxy mode, the proxy handles persistent tracing to the backend.
        However, we still log to in-memory session for immediate access via
        session.llm_calls without any I/O.

        Uses the shared InMemorySessionTracer to avoid code duplication.
        """
        # Get context metadata and merge with call-specific metadata
        context_metadata = get_current_metadata()
        merged_metadata = {**context_metadata, **(dict(metadata_overrides) if metadata_overrides else {})}

        # Extract session_name from context
        session_name = get_current_session_name()

        # Extract token usage for the tracer
        usage = response_payload.get("usage") or {}
        tokens = {
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "total": int(usage.get("total_tokens") or 0),
        }

        # Extract trace_id from response (e.g., "chatcmpl-xxx" from OpenAI)
        trace_id = response_payload.get("id")

        # Extract session_uids and sessions from context
        session_uids = get_active_session_uids()
        sessions = get_active_cv_sessions()

        # Use the shared memory tracer
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


@dataclass
class _ProxyChatCompletions:
    parent: ProxyTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = _merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            # Proxy mode: inject metadata slug into URL
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            # Direct mode: use client as-is (like normal OpenAI client)
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = _extract_completion_tokens(response_dict)
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
    parent: ProxyTrackedChatClient

    @property
    def completions(self) -> _ProxyChatCompletions:
        return _ProxyChatCompletions(self.parent)


@dataclass
class _ProxyCompletions:
    parent: ProxyTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = _merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            # Proxy mode: inject metadata slug into URL
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            # Direct mode: use client as-is (like normal OpenAI client)
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = _extract_completion_tokens(response_dict)
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

        self.tracer = tracer
        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy

        self.chat = _ProxyChatNamespace(self)
        self.completions = _ProxyCompletions(self)


@dataclass
class _ProxyAsyncChatCompletions:
    parent: ProxyTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = _merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            # Proxy mode: inject metadata slug into URL
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            # Direct mode: use client as-is (like normal OpenAI client)
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = await scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = _extract_completion_tokens(response_dict)
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
    parent: ProxyTrackedAsyncChatClient

    @property
    def completions(self) -> _ProxyAsyncChatCompletions:
        return _ProxyAsyncChatCompletions(self.parent)


@dataclass
class _ProxyAsyncCompletions:
    parent: ProxyTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = _merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        # Choose client based on use_proxy setting
        if self.parent.use_proxy:
            # Proxy mode: inject metadata slug into URL
            routing_metadata = assemble_routing_metadata(metadata)
            scoped_client = self.parent._scoped_client(routing_metadata)
        else:
            # Direct mode: use client as-is (like normal OpenAI client)
            scoped_client = self.parent._client

        start = time.perf_counter()
        response = await scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = _extract_completion_tokens(response_dict)
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

        self.tracer = tracer
        self.default_model = default_model
        self.base_url = base_url
        self.use_proxy = use_proxy

        self.chat = _ProxyAsyncChatNamespace(self)
        self.completions = _ProxyAsyncCompletions(self)
