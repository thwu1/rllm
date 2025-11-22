"""Simple chat client that logs completions via a tracer.

This client provides direct tracing without proxy involvement.
"""

from __future__ import annotations

import time
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
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions


class _SimpleTrackedChatClientBase:
    """Shared logic for sync and async simple tracked clients."""

    tracer: Any
    default_model: str | None

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
        if not self.tracer:
            return

        session_name = get_current_session_name()
        context_metadata = dict(get_current_metadata() or {})

        metadata: dict[str, Any] = dict(context_metadata)
        metadata.update(dict(metadata_overrides or {}))
        metadata["token_ids"] = {"prompt": []}
        if completion_token_ids:
            metadata["token_ids"]["completion"] = completion_token_ids

        tokens = extract_usage_tokens(response_payload)
        trace_id = response_payload.get("id")
        session_uids = get_active_session_uids()
        sessions = get_active_cv_sessions()

        self.tracer.log_llm_call(
            name="simple.chat.completions.create",
            model=model,
            input={"messages": messages},
            output=response_payload,
            session_name=session_name,
            metadata=metadata,
            latency_ms=latency_ms,
            tokens=tokens,
            trace_id=trace_id,
            session_uids=session_uids,
            sessions=sessions,
        )


# =============================================================================
# Sync Client Implementation
# =============================================================================


@dataclass
class _ChatCompletions:
    parent: "SimpleTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = self.parent._client.chat.completions.create(**call_kwargs)
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
class _ChatNamespace:
    parent: "SimpleTrackedChatClient"

    @property
    def completions(self) -> _ChatCompletions:
        return _ChatCompletions(self.parent)


@dataclass
class _Completions:
    parent: "SimpleTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = self.parent._client.completions.create(**call_kwargs)
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


class SimpleTrackedChatClient(_SimpleTrackedChatClientBase):
    """Lean wrapper around OpenAI that records chat completions via a tracer."""

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

        self.chat = _ChatNamespace(self)
        self.completions = _Completions(self)


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _AsyncChatCompletions:
    parent: "SimpleTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = await self.parent._client.chat.completions.create(**call_kwargs)
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
class _AsyncChatNamespace:
    parent: "SimpleTrackedAsyncChatClient"

    @property
    def completions(self) -> _AsyncChatCompletions:
        return _AsyncChatCompletions(self.parent)


@dataclass
class _AsyncCompletions:
    parent: "SimpleTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        start = time.perf_counter()
        response = await self.parent._client.completions.create(**call_kwargs)
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


class SimpleTrackedAsyncChatClient(_SimpleTrackedChatClientBase):
    """Async variant of the simple client that records chat completions."""

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

        self.chat = _AsyncChatNamespace(self)
        self.completions = _AsyncCompletions(self)
