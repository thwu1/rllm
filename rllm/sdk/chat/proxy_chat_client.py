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

from .simple_chat_client import _SimpleTrackedChatClientBase


class _ScopedClientMixin:
    def _scoped_client(self, metadata: Mapping[str, Any]):
        base_url = getattr(self, "_proxy_base_url", None)
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
        # Proxy mode logs inside the proxy middleware; disable SDK-side logging.
        return None


@dataclass
class _ProxyChatCompletions:
    parent: ProxyTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata)
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)
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
        call_kwargs = self.parent._merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata)
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)
        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


class ProxyTrackedChatClient(_SimpleTrackedChatClientBase, _ScopedClientMixin):
    """OpenAI client wrapper that injects metadata slugs into the proxy base URL."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
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
        self._proxy_base_url = base_url

        self.chat = _ProxyChatNamespace(self)
        self.completions = _ProxyCompletions(self)


@dataclass
class _ProxyAsyncChatCompletions:
    parent: ProxyTrackedAsyncChatClient

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata)
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = await scoped_client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)
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
        call_kwargs = self.parent._merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        routing_metadata = assemble_routing_metadata(metadata)
        scoped_client = self.parent._scoped_client(routing_metadata)

        start = time.perf_counter()
        response = await scoped_client.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)
        self.parent._log_trace(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_payload=response_dict,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )
        return response


class ProxyTrackedAsyncChatClient(_SimpleTrackedChatClientBase, _ScopedClientMixin):
    """Async variant of the proxy-aware chat client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        default_model: str | None = None,
        client: AsyncOpenAI | None = None,
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
        self._proxy_base_url = base_url

        self.chat = _ProxyAsyncChatNamespace(self)
        self.completions = _ProxyAsyncCompletions(self)
