"""Base protocol and utilities for tracer implementations."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TracerProtocol(Protocol):
    """
    Common interface for all tracer implementations.

    All tracer types (in-memory, persistent, etc.) must implement this protocol.
    This allows chat clients to work with any tracer through a uniform API.

    Example:
        >>> from rllm.sdk.tracers import InMemorySessionTracer, EpisodicTracer
        >>>
        >>> # Use in-memory tracer
        >>> tracer = InMemorySessionTracer()
        >>> llm = get_chat_client(tracer=tracer, ...)
        >>>
        >>> # Or use persistent tracer
        >>> tracer = EpisodicTracer(context_store=cs, project="my-app")
        >>> llm = get_chat_client(tracer=tracer, ...)
        >>>
        >>> # Chat client calls tracer.log_llm_call() - works with any tracer
    """

    def log_llm_call(
        self,
        name: str,
        input: str | list | dict,
        output: str | dict,
        model: str,
        latency_ms: float,
        tokens: dict[str, int],
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        cost: float | None = None,
        environment: str | None = None,
        tools: list[dict] | None = None,
        contexts: list[str | dict] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Log an LLM call trace.

        Args:
            name: Identifier for the call (e.g., "chat.completions.create")
            input: Input data (messages, prompt, etc.)
            output: Output data (response, completion, etc.)
            model: Model identifier (e.g., "gpt-4")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            session_id: Session ID (optional, may be inferred from context)
            metadata: Additional metadata dict
            trace_id: Unique trace ID (auto-generated if None)
            parent_trace_id: Parent trace ID for nested calls
            cost: Cost in USD (optional)
            environment: Environment name (e.g., "production", "dev")
            tools: List of tool definitions used
            contexts: List of context IDs or dicts
            tags: List of tags for categorization
        """
        ...

    def flush(self, timeout: float = 30.0) -> bool | None:
        """
        Flush all pending traces (blocking).

        This method should block until all traces are persisted or
        the timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if flush succeeded, False if it failed/timed out,
            or None for backward compatibility (treated as success)
        """
        ...

    async def close(self, timeout: float = 30.0) -> None:
        """
        Close tracer and flush pending traces.

        This method should clean up resources and ensure all traces
        are persisted before returning.

        Args:
            timeout: Maximum time to wait in seconds
        """
        ...


class CompositeTracer:
    """
    Tracer that delegates to multiple child tracers.

    Use this when you want to log to multiple backends simultaneously,
    e.g., both in-memory (for immediate access) and persistent storage.

    Example:
        >>> from rllm.sdk.tracers import InMemorySessionTracer, EpisodicTracer, CompositeTracer
        >>>
        >>> # Create individual tracers
        >>> memory = InMemorySessionTracer()
        >>> episodic = EpisodicTracer(context_store=cs, project="my-app")
        >>>
        >>> # Combine them
        >>> tracer = CompositeTracer([memory, episodic])
        >>>
        >>> # Use with chat client - logs to both tracers
        >>> llm = get_chat_client(tracer=tracer, ...)
        >>> llm.chat.completions.create(...)  # Logged to memory AND episodic
    """

    def __init__(self, tracers: list[TracerProtocol]):
        """
        Initialize composite tracer.

        Args:
            tracers: List of tracer instances to delegate to
        """
        self.tracers = tracers

    def log_llm_call(
        self,
        name: str,
        input: str | list | dict,
        output: str | dict,
        model: str,
        latency_ms: float,
        tokens: dict[str, int],
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        parent_trace_id: str | None = None,
        cost: float | None = None,
        environment: str | None = None,
        tools: list[dict] | None = None,
        contexts: list[str | dict] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Log trace to all child tracers.

        Each tracer receives the same trace data. If a tracer fails,
        the error is logged but other tracers continue to receive the trace.

        Args:
            See TracerProtocol.log_llm_call() for parameter descriptions.
        """
        for tracer in self.tracers:
            try:
                tracer.log_llm_call(
                    name=name,
                    input=input,
                    output=output,
                    model=model,
                    latency_ms=latency_ms,
                    tokens=tokens,
                    session_id=session_id,
                    metadata=metadata,
                    trace_id=trace_id,
                    parent_trace_id=parent_trace_id,
                    cost=cost,
                    environment=environment,
                    tools=tools,
                    contexts=contexts,
                    tags=tags,
                )
            except Exception as e:
                logger.exception(f"Tracer {tracer.__class__.__name__} failed to log trace: {e}")

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Flush all child tracers.

        Args:
            timeout: Maximum time to wait in seconds (applied per tracer)

        Returns:
            True if all tracers flushed successfully, False if any failed
        """
        all_succeeded = True
        for tracer in self.tracers:
            try:
                result = tracer.flush(timeout=timeout)
                # Treat None as success for backward compatibility
                if result is False:
                    all_succeeded = False
                    logger.warning(f"Tracer {tracer.__class__.__name__} flush returned False")
            except Exception as e:
                all_succeeded = False
                logger.exception(f"Tracer {tracer.__class__.__name__} failed to flush: {e}")
        return all_succeeded

    async def close(self, timeout: float = 30.0) -> None:
        """
        Close all child tracers.

        Args:
            timeout: Maximum time to wait in seconds (applied per tracer)
        """
        for tracer in self.tracers:
            try:
                await tracer.close(timeout=timeout)
            except Exception as e:
                logger.exception(f"Tracer {tracer.__class__.__name__} failed to close: {e}")

    def __repr__(self):
        return f"CompositeTracer(tracers={[t.__class__.__name__ for t in self.tracers]})"
