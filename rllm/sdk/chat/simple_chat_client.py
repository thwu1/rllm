"""Simple chat client that logs completions via a tracer.

This client provides direct tracing without proxy involvement.
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
from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name
from rllm.sdk.session.contextvar import get_active_cv_sessions


class _TracingMixin:
    """Mixin providing trace logging functionality for simple clients."""

    tracer: Any

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
        """Log trace to the configured tracer.

        Args:
            model: Model identifier used for the call
            messages: Input messages sent to the model
            response_payload: Full response dict from the API
            completion_token_ids: Token IDs from response (if available)
            metadata_overrides: Call-specific metadata to merge
            latency_ms: Latency of the API call in milliseconds
        """
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
class _SimpleChatCompletions(ChatCompletionsBase):
    """Chat completions namespace for SimpleTrackedChatClient."""

    parent: "SimpleTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion and log the trace."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        messages = call_kwargs["messages"]

        with TimedCall() as timer:
            response = self.parent._client.chat.completions.create(**call_kwargs)

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
class _SimpleChatNamespace:
    """Chat namespace for SimpleTrackedChatClient."""

    parent: "SimpleTrackedChatClient"

    @property
    def completions(self) -> _SimpleChatCompletions:
        return _SimpleChatCompletions(self.parent)


@dataclass
class _SimpleCompletions(CompletionsBase):
    """Completions namespace for SimpleTrackedChatClient."""

    parent: "SimpleTrackedChatClient"

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion and log the trace."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        prompt = call_kwargs["prompt"]

        with TimedCall() as timer:
            response = self.parent._client.completions.create(**call_kwargs)

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


class SimpleTrackedChatClient(_TracingMixin, BaseChatClient[OpenAI]):
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
        """Initialize the simple tracked chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API endpoint
            tracer: Tracer instance for logging calls
            default_model: Default model to use
            client: Pre-configured OpenAI client
            **client_kwargs: Additional client configuration
        """
        self.tracer = tracer
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

    def _create_chat_namespace(self) -> _SimpleChatNamespace:
        return _SimpleChatNamespace(self)

    def _create_completions_namespace(self) -> _SimpleCompletions:
        return _SimpleCompletions(self)


# =============================================================================
# Async Client Implementation
# =============================================================================


@dataclass
class _SimpleAsyncChatCompletions(ChatCompletionsBase):
    """Async chat completions namespace for SimpleTrackedAsyncChatClient."""

    parent: "SimpleTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        """Create a chat completion and log the trace."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        messages = call_kwargs["messages"]

        with TimedCall() as timer:
            response = await self.parent._client.chat.completions.create(**call_kwargs)

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
class _SimpleAsyncChatNamespace:
    """Async chat namespace for SimpleTrackedAsyncChatClient."""

    parent: "SimpleTrackedAsyncChatClient"

    @property
    def completions(self) -> _SimpleAsyncChatCompletions:
        return _SimpleAsyncChatCompletions(self.parent)


@dataclass
class _SimpleAsyncCompletions(CompletionsBase):
    """Async completions namespace for SimpleTrackedAsyncChatClient."""

    parent: "SimpleTrackedAsyncChatClient"

    async def create(self, *args: Any, **kwargs: Any) -> Completion:
        """Create a completion and log the trace."""
        call_kwargs, model, metadata = self._validate_and_prepare(args, kwargs)
        prompt = call_kwargs["prompt"]

        with TimedCall() as timer:
            response = await self.parent._client.completions.create(**call_kwargs)

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


class SimpleTrackedAsyncChatClient(_TracingMixin, BaseAsyncChatClient[AsyncOpenAI]):
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
        """Initialize the async simple tracked chat client.

        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API endpoint
            tracer: Tracer instance for logging calls
            default_model: Default model to use
            client: Pre-configured AsyncOpenAI client
            **client_kwargs: Additional client configuration
        """
        self.tracer = tracer
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            client=client,
            **client_kwargs,
        )

    def _create_chat_namespace(self) -> _SimpleAsyncChatNamespace:
        return _SimpleAsyncChatNamespace(self)

    def _create_completions_namespace(self) -> _SimpleAsyncCompletions:
        return _SimpleAsyncCompletions(self)
