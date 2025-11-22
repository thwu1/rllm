"""Base protocol and runtime helpers for session implementations."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rllm.sdk.session import SESSION_BACKEND


@runtime_checkable
class SessionProtocol(Protocol):
    """
    Common interface for all session implementations.

    All session types (ContextVar-based, OpenTelemetry-based, etc.)
    must implement this protocol.
    """

    name: str
    metadata: dict[str, Any]

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        """Get all LLM calls made within this session."""
        ...

    def clear_calls(self) -> None:
        """Clear all stored calls for this session."""
        ...

    def __enter__(self) -> "SessionProtocol":
        """Enter session context."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit session context."""
        ...


# Runtime helpers for configuring session-aware agent functions


def _ensure_tracer_initialized(service_name: str | None = None) -> None:
    """Configure OpenTelemetry tracer provider once per process (idempotent)."""
    from rllm.sdk.session import SESSION_BACKEND

    if SESSION_BACKEND != "opentelemetry":
        return
    from rllm.sdk.session.opentelemetry import configure_default_tracer

    configure_default_tracer(service_name=service_name)


def wrap_with_session_context(agent_func, *, tracer_service_name: str | None = None):
    """
    Wrap an agent function so each invocation runs inside a session context.

    Args:
        agent_func: Original agent_run_func provided to AgentSdkEngine.
        tracer_service_name: Optional service name for OT tracer configuration.

    Returns:
        Callable that injects a session context and returns (output, session_uid).
    """
    from rllm.sdk.shortcuts import _session_with_name

    if inspect.iscoroutinefunction(agent_func):

        async def wrapped(metadata, *args, **kwargs):
            _ensure_tracer_initialized(service_name=tracer_service_name)
            session_name = metadata.pop("session_name", None)
            session_kwargs = metadata.copy()
            with _session_with_name(name=session_name, **session_kwargs) as session:
                output = await agent_func(*args, **kwargs)
            return output, session._uid

        return wrapped

    def wrapped_sync(metadata, *args, **kwargs):
        _ensure_tracer_initialized(service_name=tracer_service_name)
        session_name = metadata.pop("session_name", None)
        session_kwargs = metadata.copy()
        with _session_with_name(name=session_name, **session_kwargs) as session:
            output = agent_func(*args, **kwargs)
        return output, session._uid

    return wrapped_sync


__all__ = ["SessionProtocol", "wrap_with_session_context"]
