"""OpenAI SDK instrumentation provider.

This module patches the OpenAI Python SDK to automatically capture LLM traces
within session contexts, with the same functionality as ProxyTrackedChatClient
and OpenTelemetryTrackedChatClient.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.completion import Completion

# Track instrumentation state
_instrumented = False
_original_sync_chat_create: Callable | None = None
_original_async_chat_create: Callable | None = None
_original_sync_completions_create: Callable | None = None
_original_async_completions_create: Callable | None = None


def _extract_response_content(response: Any) -> dict[str, Any]:
    """Extract response content from OpenAI response object."""
    result: dict[str, Any] = {}

    if hasattr(response, "model_dump"):
        return response.model_dump()

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            result["role"] = getattr(choice.message, "role", None)
            result["content"] = getattr(choice.message, "content", None)
        elif hasattr(choice, "text"):
            result["text"] = choice.text

    if hasattr(response, "id"):
        result["id"] = response.id
    if hasattr(response, "model"):
        result["model"] = response.model

    return result


def _extract_tokens(response: Any) -> dict[str, int]:
    """Extract token usage from OpenAI response object."""
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion": getattr(response.usage, "completion_tokens", 0) or 0,
            "total": getattr(response.usage, "total_tokens", 0) or 0,
        }
    return {"prompt": 0, "completion": 0, "total": 0}


def _log_trace(
    name: str,
    messages: list[dict[str, Any]] | str,
    response: Any,
    model: str,
    latency_ms: float,
) -> None:
    """Log trace to active sessions using InMemorySessionTracer.

    This provides the same functionality as ProxyTrackedChatClient._log_trace.
    """
    from rllm.sdk.session import SESSION_BACKEND, get_active_session_uids, get_current_metadata, get_current_session_name
    # Import InMemorySessionTracer directly to avoid sqlite dependencies
    from rllm.sdk.tracers.memory import InMemorySessionTracer

    # Check if we're in a session context
    session_uids = get_active_session_uids()
    if not session_uids:
        return  # Not in a session, skip tracing

    # Get session info
    session_name = get_current_session_name()
    metadata = dict(get_current_metadata())

    # Extract response data
    response_payload = _extract_response_content(response)
    tokens = _extract_tokens(response)
    trace_id = response_payload.get("id")

    # Get active sessions for in-memory tracing
    if SESSION_BACKEND == "opentelemetry":
        # OpenTelemetry backend doesn't use ContextVarSession objects
        # Traces are captured via baggage propagation to proxy
        sessions = None
    else:
        from rllm.sdk.session.contextvar import get_active_cv_sessions
        sessions = get_active_cv_sessions()

    # Format input
    if isinstance(messages, str):
        input_data = {"prompt": messages}
    else:
        input_data = {"messages": messages}

    # Use shared InMemorySessionTracer (same as ProxyTrackedChatClient)
    tracer = InMemorySessionTracer()
    tracer.log_llm_call(
        name=name,
        input=input_data,
        output=response_payload,
        model=model,
        latency_ms=latency_ms,
        tokens=tokens,
        metadata=metadata,
        trace_id=trace_id,
        session_name=session_name,
        session_uids=session_uids,
        sessions=sessions,
    )


def _wrap_sync_chat_create(original: Callable) -> Callable:
    """Wrap synchronous chat.completions.create method."""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs) -> "ChatCompletion":
        messages = kwargs.get("messages", args[0] if args else None)
        model = kwargs.get("model", "unknown")

        start = time.perf_counter()
        response = original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        _log_trace(
            name="openai.chat.completions.create",
            messages=messages or [],
            response=response,
            model=model,
            latency_ms=latency_ms,
        )

        return response

    return wrapped


def _wrap_async_chat_create(original: Callable) -> Callable:
    """Wrap asynchronous chat.completions.create method."""

    @functools.wraps(original)
    async def wrapped(self, *args, **kwargs) -> "ChatCompletion":
        messages = kwargs.get("messages", args[0] if args else None)
        model = kwargs.get("model", "unknown")

        start = time.perf_counter()
        response = await original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        _log_trace(
            name="openai.chat.completions.create",
            messages=messages or [],
            response=response,
            model=model,
            latency_ms=latency_ms,
        )

        return response

    return wrapped


def _wrap_sync_completions_create(original: Callable) -> Callable:
    """Wrap synchronous completions.create method."""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs) -> "Completion":
        prompt = kwargs.get("prompt", args[0] if args else None)
        model = kwargs.get("model", "unknown")

        start = time.perf_counter()
        response = original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        _log_trace(
            name="openai.completions.create",
            messages=prompt or "",
            response=response,
            model=model,
            latency_ms=latency_ms,
        )

        return response

    return wrapped


def _wrap_async_completions_create(original: Callable) -> Callable:
    """Wrap asynchronous completions.create method."""

    @functools.wraps(original)
    async def wrapped(self, *args, **kwargs) -> "Completion":
        prompt = kwargs.get("prompt", args[0] if args else None)
        model = kwargs.get("model", "unknown")

        start = time.perf_counter()
        response = await original(self, *args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        _log_trace(
            name="openai.completions.create",
            messages=prompt or "",
            response=response,
            model=model,
            latency_ms=latency_ms,
        )

        return response

    return wrapped


def instrument_openai() -> bool:
    """Instrument OpenAI SDK for automatic trace capture.

    Returns:
        True if instrumentation was successful, False if OpenAI is not installed
        or already instrumented.
    """
    global _instrumented
    global _original_sync_chat_create, _original_async_chat_create
    global _original_sync_completions_create, _original_async_completions_create

    if _instrumented:
        return True

    try:
        from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
        from openai.resources.chat.completions import Completions as SyncChatCompletions
        from openai.resources.completions import AsyncCompletions, Completions
    except ImportError:
        return False  # OpenAI not installed

    # Store originals
    _original_sync_chat_create = SyncChatCompletions.create
    _original_async_chat_create = AsyncChatCompletions.create
    _original_sync_completions_create = Completions.create
    _original_async_completions_create = AsyncCompletions.create

    # Apply patches
    SyncChatCompletions.create = _wrap_sync_chat_create(_original_sync_chat_create)
    AsyncChatCompletions.create = _wrap_async_chat_create(_original_async_chat_create)
    Completions.create = _wrap_sync_completions_create(_original_sync_completions_create)
    AsyncCompletions.create = _wrap_async_completions_create(_original_async_completions_create)

    _instrumented = True
    return True


def uninstrument_openai() -> None:
    """Remove OpenAI SDK instrumentation."""
    global _instrumented
    global _original_sync_chat_create, _original_async_chat_create
    global _original_sync_completions_create, _original_async_completions_create

    if not _instrumented:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
        from openai.resources.chat.completions import Completions as SyncChatCompletions
        from openai.resources.completions import AsyncCompletions, Completions
    except ImportError:
        return

    # Restore originals
    if _original_sync_chat_create is not None:
        SyncChatCompletions.create = _original_sync_chat_create
    if _original_async_chat_create is not None:
        AsyncChatCompletions.create = _original_async_chat_create
    if _original_sync_completions_create is not None:
        Completions.create = _original_sync_completions_create
    if _original_async_completions_create is not None:
        AsyncCompletions.create = _original_async_completions_create

    _instrumented = False
    _original_sync_chat_create = None
    _original_async_chat_create = None
    _original_sync_completions_create = None
    _original_async_completions_create = None


def is_instrumented() -> bool:
    """Check if OpenAI SDK is currently instrumented."""
    return _instrumented
