"""OpenAI SDK instrumentation provider.

Patches OpenAI Python SDK to automatically capture LLM traces within session contexts.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

# Instrumentation state
_instrumented = False
_originals: dict[str, Callable] = {}


def _extract_response(response: Any) -> tuple[dict, dict[str, int]]:
    """Extract response content and token usage from OpenAI response."""
    content = response.model_dump() if hasattr(response, "model_dump") else {}

    usage = getattr(response, "usage", None)
    tokens = {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    } if usage else {"prompt": 0, "completion": 0, "total": 0}

    return content, tokens


def _log_trace(name: str, input_data: Any, response: Any, model: str, latency_ms: float) -> None:
    """Log trace to active sessions if within a session context."""
    from rllm.sdk.session import SESSION_BACKEND, get_active_session_uids, get_current_metadata, get_current_session_name
    from rllm.sdk.tracers.memory import InMemorySessionTracer

    session_uids = get_active_session_uids()
    if not session_uids:
        return

    response_payload, tokens = _extract_response(response)

    # Get sessions for in-memory tracing (ContextVar backend only)
    sessions = None
    if SESSION_BACKEND != "opentelemetry":
        from rllm.sdk.session.contextvar import get_active_cv_sessions
        sessions = get_active_cv_sessions()

    InMemorySessionTracer().log_llm_call(
        name=name,
        input={"messages": input_data} if isinstance(input_data, list) else {"prompt": input_data},
        output=response_payload,
        model=model,
        latency_ms=latency_ms,
        tokens=tokens,
        metadata=dict(get_current_metadata()),
        trace_id=response_payload.get("id"),
        session_name=get_current_session_name(),
        session_uids=session_uids,
        sessions=sessions,
    )


def _make_wrapper(original: Callable, name: str, input_key: str, is_async: bool) -> Callable:
    """Create a wrapper function for an OpenAI method."""
    if is_async:
        @functools.wraps(original)
        async def async_wrapper(self, *args, **kwargs):
            input_data = kwargs.get(input_key, args[0] if args else None) or []
            model = kwargs.get("model", "unknown")

            start = time.perf_counter()
            response = await original(self, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            _log_trace(name, input_data, response, model, latency_ms)
            return response
        return async_wrapper
    else:
        @functools.wraps(original)
        def sync_wrapper(self, *args, **kwargs):
            input_data = kwargs.get(input_key, args[0] if args else None) or []
            model = kwargs.get("model", "unknown")

            start = time.perf_counter()
            response = original(self, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            _log_trace(name, input_data, response, model, latency_ms)
            return response
        return sync_wrapper


def instrument_openai() -> bool:
    """Instrument OpenAI SDK. Returns True if successful, False if OpenAI not installed."""
    global _instrumented

    if _instrumented:
        return True

    # Import OpenAI classes (fails gracefully if not installed)
    try:
        from openai.resources.chat.completions import AsyncCompletions as AsyncChat, Completions as SyncChat
        from openai.resources.completions import AsyncCompletions as AsyncComp, Completions as SyncComp
    except ImportError:
        return False

    # Define patches: (class, method_name, trace_name, input_key, is_async)
    patches = [
        (SyncChat, "create", "openai.chat.completions.create", "messages", False),
        (AsyncChat, "create", "openai.chat.completions.create", "messages", True),
        (SyncComp, "create", "openai.completions.create", "prompt", False),
        (AsyncComp, "create", "openai.completions.create", "prompt", True),
    ]

    for cls, method, name, input_key, is_async in patches:
        key = f"{cls.__module__}.{cls.__name__}.{method}"
        original = getattr(cls, method)
        _originals[key] = original
        setattr(cls, method, _make_wrapper(original, name, input_key, is_async))

    _instrumented = True
    return True


def uninstrument_openai() -> None:
    """Remove OpenAI SDK instrumentation."""
    global _instrumented

    if not _instrumented:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions as AsyncChat, Completions as SyncChat
        from openai.resources.completions import AsyncCompletions as AsyncComp, Completions as SyncComp
    except ImportError:
        return

    classes = [SyncChat, AsyncChat, SyncComp, AsyncComp]
    for cls in classes:
        key = f"{cls.__module__}.{cls.__name__}.create"
        if key in _originals:
            setattr(cls, "create", _originals[key])

    _originals.clear()
    _instrumented = False


def is_instrumented() -> bool:
    """Check if OpenAI SDK is currently instrumented."""
    return _instrumented
