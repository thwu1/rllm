"""OpenTelemetry-based session implementation for distributed tracing.

This module provides OTelSession, an alternative to ContextVarSession that uses
OpenTelemetry baggage for automatic context propagation across HTTP/gRPC boundaries.

Key features:
- Automatic HTTP/gRPC context propagation (with instrumentation)
- Manual Ray/multiprocessing propagation (same as ContextVarSession)
- Same storage backends (InMemoryStorage, SqliteSessionStorage)
- Zero breaking changes - opt-in alternative

Example:
    >>> from rllm.sdk.session.otel import OTelSession, init_otel_distributed_tracing
    >>>
    >>> # Initialize once at startup
    >>> init_otel_distributed_tracing()
    >>>
    >>> # Use like ContextVarSession
    >>> with OTelSession(experiment="v1") as session:
    ...     llm.chat.completions.create(...)
    ...     print(len(session.llm_calls))
"""

import contextvars
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rllm.sdk.protocol import StepView, Trace, trace_to_step_view
from rllm.sdk.session.storage import SqliteSessionStorage

if TYPE_CHECKING:
    from rllm.sdk.session.storage import SessionStorage

logger = logging.getLogger(__name__)

# Module-level flags for HTTP instrumentation
_http_instrumentation_enabled = False
_auto_init_warned = False

# OTelSession-specific context variables (separate from ContextVarSession)
_current_otel_session: contextvars.ContextVar["OTelSession | None"] = contextvars.ContextVar(
    "current_otel_session", default=None
)
_otel_sessions_stack: contextvars.ContextVar[list["OTelSession"] | None] = contextvars.ContextVar(
    "otel_sessions_stack", default=None
)


def get_current_otel_session() -> "OTelSession | None":
    """Get the current OTelSession instance from context.

    Returns:
        Current OTelSession or None if not in an OTel session context.
    """
    return _current_otel_session.get()


def get_otel_session_name() -> str | None:
    """Get current OTel session name from context.

    Returns:
        Session name or None if not in an OTel session context.
    """
    session = _current_otel_session.get()
    return session.name if session else None


def get_otel_metadata() -> dict[str, Any]:
    """Get current OTel session metadata from context.

    Returns metadata from OTel baggage if available, otherwise returns empty dict.
    This works even without an active OTelSession instance, as it reads directly
    from OTel baggage (useful in HTTP handlers that receive baggage).

    Returns:
        Dictionary of metadata from OTel baggage, or empty dict.
    """
    try:
        from opentelemetry import baggage, context

        ctx = context.get_current()
        metadata = {}

        # Get metadata keys list
        metadata_keys_str = baggage.get_baggage("rllm_metadata_keys", context=ctx) or ""
        for key in metadata_keys_str.split(","):
            if key.strip():
                value = baggage.get_baggage(f"rllm_{key}", context=ctx)
                if value is not None:
                    metadata[key] = value

        return metadata
    except ImportError:
        # OTel not installed
        return {}


def get_active_otel_sessions() -> list["OTelSession"]:
    """Get a copy of the current stack of active OTel sessions (outer → inner).

    Returns:
        List of active OTelSession instances from outermost to innermost.
    """
    stack = _otel_sessions_stack.get() or []
    return list(stack)  # Return shallow copy


def init_otel_distributed_tracing():
    """Initialize OpenTelemetry distributed tracing (HTTP/gRPC instrumentation).

    Call this once at application startup to enable automatic baggage propagation
    for HTTP/gRPC requests across microservices.

    This is the **recommended** way to enable distributed tracing. If not called
    explicitly, OTelSession will auto-initialize with a warning on first use.

    Example:
        >>> from rllm.sdk.session.otel import init_otel_distributed_tracing
        >>>
        >>> # Call once at startup
        >>> init_otel_distributed_tracing()
        >>>
        >>> # Now all HTTP calls auto-propagate OTel baggage!
        >>> import requests
        >>> with OTelSession(experiment="v1"):
        ...     requests.post("http://service/api", ...)
        ...     # ✅ baggage header automatically injected!
    """
    global _http_instrumentation_enabled

    if _http_instrumentation_enabled:
        logger.debug("OTel HTTP instrumentation already enabled, skipping")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.trace import TracerProvider

        # Set up tracer provider if not already configured
        if not isinstance(trace.get_tracer_provider(), TracerProvider):
            trace.set_tracer_provider(TracerProvider())

        # Instrument HTTP libraries
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()

        _http_instrumentation_enabled = True
        logger.info("OpenTelemetry distributed tracing initialized (HTTP/gRPC instrumentation enabled)")
    except ImportError as e:
        logger.error(
            f"Failed to initialize OTel instrumentation: {e}. "
            "Install with: pip install opentelemetry-instrumentation-requests opentelemetry-instrumentation-httpx"
        )
        raise


def _ensure_instrumentation():
    """Auto-initialize instrumentation if not already done (fallback with warning).

    Called internally by OTelSession.__enter__() to ensure instrumentation is
    enabled even if user forgot to call init_otel_distributed_tracing().
    """
    global _http_instrumentation_enabled, _auto_init_warned

    if not _http_instrumentation_enabled:
        if not _auto_init_warned:
            logger.warning(
                "OTel HTTP instrumentation not explicitly initialized. "
                "Auto-initializing now. For production, call "
                "init_otel_distributed_tracing() at startup."
            )
            _auto_init_warned = True
        try:
            init_otel_distributed_tracing()
        except ImportError:
            # If OTel instrumentation libraries aren't installed, just warn and continue
            # Core OTel API should still work for baggage propagation
            logger.warning("OTel instrumentation libraries not available - HTTP auto-propagation disabled")


class OTelSession:
    """OpenTelemetry-based session with automatic distributed context propagation.

    Uses OTel baggage for automatic HTTP/gRPC propagation. Defaults to SqliteSessionStorage
    for cross-process trace sharing. Compatible with same storage backends as ContextVarSession.

    Features:
    - Automatic HTTP/gRPC context propagation (with init_otel_distributed_tracing())
    - Manual Ray/multiprocessing propagation (via to_otel_context/from_otel_context)
    - Nested sessions with metadata inheritance
    - Same storage backends (InMemoryStorage, SqliteSessionStorage)

    Example:
        >>> # Basic usage
        >>> with OTelSession(experiment="v1") as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))
        >>>
        >>> # Ray/multiprocessing
        >>> with OTelSession(experiment="v1") as session:
        ...     ctx = session.to_otel_context()
        ...     # Pass ctx to worker, then:
        ...     # with OTelSession.from_otel_context(ctx, storage):
        ...     #     llm.chat.completions.create(...)
    """

    def __init__(
        self,
        name: str | None = None,
        storage: "SessionStorage | None" = None,
        formatter: Callable[[dict], dict] | None = None,
        persistent_tracers: list | None = None,
        _session_uid_chain: list[str] | None = None,
        **metadata,
    ):
        """Initialize OTelSession with optional name and metadata.

        Args:
            name: Session name (auto-generated if None). Inherits from parent if available.
            storage: Storage backend. Defaults to SqliteSessionStorage() for distributed scenarios.
                    Pass InMemoryStorage() for single-process usage.
            formatter: Optional formatter to transform trace data (deprecated, kept for compatibility)
            persistent_tracers: Optional list of persistent tracers (deprecated, kept for compatibility)
            _session_uid_chain: Internal parameter for context restoration (do not use directly)
            **metadata: Session metadata (propagated via OTel baggage)
        """
        # If name is not explicitly provided, check if there's one in OTel baggage
        if name is None:
            try:
                from opentelemetry import baggage, context

                ctx = context.get_current()
                baggage_name = baggage.get_baggage("rllm_session_name", context=ctx)
                if baggage_name:
                    name = baggage_name
            except ImportError:
                pass

        # Generate new name only if none was provided and none exists in baggage
        self.name = name or f"sess_{uuid.uuid4().hex[:16]}"

        # Internal unique ID for this session context instance
        self._uid = f"otel_{uuid.uuid4().hex[:16]}"

        # Build session UID chain for tree hierarchy support
        if _session_uid_chain is not None:
            # Restoring from serialized context (distributed case)
            self._session_uid_chain = _session_uid_chain + [self._uid]
        else:
            # Check for parent session in current context (nested local case)
            parent_session = get_current_otel_session()
            if parent_session is not None:
                # Inherit parent's chain and append our UID
                self._session_uid_chain = parent_session._session_uid_chain + [self._uid]
            else:
                # Root session - start new chain
                self._session_uid_chain = [self._uid]

        self.metadata = metadata
        self.formatter = formatter or (lambda x: x)

        # Storage backend (defaults to SqliteSessionStorage for distributed scenarios)
        if storage is None:
            storage = SqliteSessionStorage()
        self.storage = storage

        # Optional persistent tracers (kept for backward compatibility)
        self._persistent_tracers = persistent_tracers or []

        # Context tokens for cleanup
        self._session_token = None
        self._stack_token = None
        self._otel_token = None
        self._span = None
        self._tracer = None

    @property
    def llm_calls(self) -> list[Trace]:
        """Get all LLM traces from this session and nested child sessions.

        Parent sessions automatically see traces from nested children via session UID hierarchy.
        For multi-process scenarios, use to_otel_context()/from_otel_context() for hierarchy propagation.
        """
        return self.storage.get_traces(self._uid, self.name)

    @property
    def steps(self) -> list[StepView]:
        """Get all steps within this session."""
        return [trace_to_step_view(trace) for trace in self.llm_calls]

    def clear_calls(self) -> None:
        """Clear all traces for this session (InMemoryStorage only)."""
        if hasattr(self.storage, "clear"):
            self.storage.clear(self._uid, self.name)

    def __enter__(self):
        """Enter session context - set up OTel baggage and context variables."""
        # Ensure HTTP instrumentation is enabled (auto-init if needed)
        _ensure_instrumentation()

        try:
            from opentelemetry import baggage, context, trace

            # Get tracer
            self._tracer = trace.get_tracer(__name__)

            # Start OTel span for tracing
            self._span = self._tracer.start_span(f"session:{self.name}")
            self._span.__enter__()

            # Get current OTel context
            ctx = context.get_current()

            # Set session metadata in baggage
            ctx = baggage.set_baggage("rllm_session_name", self.name, context=ctx)
            ctx = baggage.set_baggage("rllm_session_uid", self._uid, context=ctx)

            # Inherit parent metadata from baggage
            parent_metadata = get_otel_metadata()

            # Merge parent metadata with current metadata
            merged_metadata = {**parent_metadata, **self.metadata}

            # Set all metadata fields (with rllm_ prefix)
            for key, value in merged_metadata.items():
                ctx = baggage.set_baggage(f"rllm_{key}", str(value), context=ctx)

            # Store list of metadata keys for retrieval
            metadata_keys = ",".join(merged_metadata.keys())
            ctx = baggage.set_baggage("rllm_metadata_keys", metadata_keys, context=ctx)

            # Attach OTel context
            self._otel_token = context.attach(ctx)

        except ImportError:
            logger.warning("OpenTelemetry not installed - context propagation will be limited to process-local")

        # Also set in process-local contextvars for fast lookup
        self._session_token = _current_otel_session.set(self)

        # Push onto sessions stack (outer → inner order)
        current_stack = _otel_sessions_stack.get() or []
        new_stack = list(current_stack)
        new_stack.append(self)
        self._stack_token = _otel_sessions_stack.set(new_stack)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - restore previous context."""
        # End OTel span
        if self._span is not None:
            try:
                self._span.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug(f"Error ending OTel span: {e}")

        # Detach OTel context
        if self._otel_token is not None:
            try:
                from opentelemetry import context

                context.detach(self._otel_token)
            except Exception as e:
                logger.debug(f"Error detaching OTel context: {e}")

        # Restore process-local contextvars
        if self._session_token is not None:
            _current_otel_session.reset(self._session_token)

        if self._stack_token is not None:
            _otel_sessions_stack.reset(self._stack_token)

        return False

    def __len__(self) -> int:
        """Return number of calls in this session."""
        return len(self.llm_calls)

    def to_otel_context(self) -> dict:
        """Serialize session context for cross-process propagation.

        Returns dict with name, session_uid_chain (for hierarchy), and metadata.
        Use with from_otel_context() to restore session in Ray workers or multiprocessing.

        Example:
            >>> with OTelSession(experiment="v1") as session:
            ...     ctx = session.to_otel_context()
            ...     # Pass ctx to Ray worker
            >>>
            >>> # In Ray worker:
            >>> with OTelSession.from_otel_context(ctx, storage):
            ...     llm.call()  # Has parent context!
        """
        return {
            "name": self.name,
            "session_uid_chain": self._session_uid_chain[:-1],  # Exclude current UID
            "metadata": self.metadata,
        }

    @classmethod
    def from_otel_context(
        cls,
        context: dict,
        storage: "SessionStorage | None" = None,
    ) -> "OTelSession":
        """Restore session from serialized context (for cross-process tracing).

        Creates new session that continues parent hierarchy via inherited UID chain.

        Args:
            context: Dict from to_otel_context()
            storage: Storage backend (should match parent session's storage)

        Returns:
            OTelSession that continues parent hierarchy

        Example:
            >>> # In parent process:
            >>> with OTelSession(experiment="v1") as session:
            ...     ctx = session.to_otel_context()
            >>>
            >>> # In worker process:
            >>> storage = SqliteSessionStorage("traces.db")
            >>> with OTelSession.from_otel_context(ctx, storage):
            ...     llm.call()  # Inherits parent's experiment=v1
        """
        return cls(
            name=context["name"],
            _session_uid_chain=context["session_uid_chain"],
            storage=storage,
            **context.get("metadata", {}),
        )

    def __repr__(self):
        return (
            f"OTelSession(name={self.name!r}, _uid={self._uid!r}, "
            f"chain_depth={len(self._session_uid_chain)}, storage={self.storage!r})"
        )


# Ray integration helper decorator
def ray_entrypoint(func):
    """Decorator to enable OTel context restoration in Ray workers.

    Automatically restores OTelSession context if _otel_ctx kwarg is passed.

    Usage:
        >>> @ray.remote
        >>> @ray_entrypoint
        >>> def train_episode(task):
        ...     # Context automatically restored if _otel_ctx passed
        ...     llm.chat.completions.create(...)
        >>>
        >>> with OTelSession(experiment="v1") as session:
        ...     ctx = session.to_otel_context()
        ...     ray.get([train_episode.remote(task, _otel_ctx=ctx) for task in tasks])
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        otel_ctx = kwargs.pop("_otel_ctx", None)

        if otel_ctx:
            storage = SqliteSessionStorage()
            with OTelSession.from_otel_context(otel_ctx, storage=storage):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
