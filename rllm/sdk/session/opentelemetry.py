"""OpenTelemetry-backed session implementation using W3C baggage for distributed context."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

try:  # pragma: no cover - optional dependency
    from opentelemetry import baggage as otel_baggage
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
except ImportError:  # pragma: no cover
    otel_trace = None
    otel_baggage = None
    otel_context = None

from rllm.sdk.protocol import Trace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.trace import Span as _OtelSpan
    from opentelemetry.trace import Tracer as _OtelTracer

    from rllm.sdk.store import SqliteTraceStore
else:  # pragma: no cover - fallback for runtime when otel not installed
    _OtelTracer = Any  # type: ignore
    _OtelSpan = Any  # type: ignore

# Context key for storing the current OpenTelemetrySession in OTel context
_SESSION_CONTEXT_KEY = "rllm.session"


def get_current_otel_metadata() -> dict[str, Any]:
    """Get current metadata from the active OpenTelemetrySession.

    Returns:
        Session metadata dict, or empty dict if no session is active.
    """
    session = get_current_otel_session()
    if session is None:
        return {}
    return dict(session.metadata)


def get_current_otel_session_name() -> str | None:
    """Get current session name from the active OpenTelemetrySession.

    Returns:
        Session name, or None if no session is active.
    """
    session = get_current_otel_session()
    if session is None:
        return None
    return session.name


def get_active_otel_sessions() -> list[OpenTelemetrySession]:
    """Get list of active OpenTelemetry sessions.

    Note: Unlike ContextVarSession, OpenTelemetrySession doesn't support
    nested session stacks. This returns a list with the current session
    for compatibility with the session API.

    Returns:
        List containing current session, or empty list if no session is active.
    """
    session = get_current_otel_session()
    if session is None:
        return []
    return [session]


def otel_session(**kwargs: Any) -> OpenTelemetrySession:
    """Convenience factory for creating OpenTelemetry sessions.

    Example:
        >>> with otel_session(name="my-task", metadata={"env": "prod"}) as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))
    """
    return OpenTelemetrySession(**kwargs)


class OpenTelemetrySession:
    """Session implementation that uses OpenTelemetry spans and W3C baggage for distributed context.

    Unlike ContextVarSession, this session propagates hierarchy automatically across HTTP
    boundaries via W3C baggage, enabling true distributed tracing without manual context passing.

    Example:
        >>> with otel_session(name="client") as client_session:
        ...     # HTTP call automatically carries session context via baggage
        ...     httpx.post("http://server/api")
        ...
        >>> # On server:
        >>> with otel_session(name="handler") as server_session:
        ...     llm.chat.completions.create(...)
        ...     # server_session automatically inherits client's UID chain
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        tracer: _OtelTracer | None = None,
        tracer_name: str = "rllm.sdk.session",
        store: SqliteTraceStore | None = None,
        store_db_path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> None:
        """
        Initialize an OpenTelemetry-backed session.

        Args:
            name: Session name (auto-generated if None)
            tracer: OpenTelemetry tracer instance (uses global tracer if None)
            tracer_name: Tracer name for span creation
            store: SqliteTraceStore instance for querying LLM calls
            store_db_path: Path to SQLite database (used if store is None)
            metadata: Session metadata
            **extra_metadata: Additional metadata as keyword arguments
        """
        self.name = name or f"sess_{uuid.uuid4().hex[:16]}"
        self._uid = f"ctx_{uuid.uuid4().hex[:16]}"  # Will be replaced by span ID
        self._session_uid_chain: list[str] = []

        self.metadata = dict(metadata or {})
        self.metadata.update(extra_metadata)

        self._otel_tracer: _OtelTracer | None = tracer
        self._tracer_name = tracer_name
        self._span_scope = None
        self._span: _OtelSpan | None = None
        self._trace_id: str | None = None
        self._span_id: str | None = None

        self._store = store
        self._store_db_path = store_db_path
        self._baggage_token = None
        self._session_context_token = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        """Enter session context: restore baggage, start span, propagate downstream."""
        # 1. Start OpenTelemetry span first (this restores propagated context including baggage)
        tracer = self._ensure_tracer()
        span_name = f"session:{self.name}" if self.name else "session"
        self._span_scope = tracer.start_as_current_span(span_name)
        self._span = self._span_scope.__enter__()

        # 2. Now read parent session context from incoming baggage (after context is restored)
        parent_chain, parent_meta = self._read_from_baggage()

        # 3. Extract span context and use span ID as session UID
        self._cache_span_context()
        if self._span_id:
            self._uid = self._span_id

        # 4. Build UID chain: inherit parent + append our UID
        if parent_chain:
            self._session_uid_chain = parent_chain + [self._uid]
            # Merge parent metadata
            self.metadata = {**parent_meta, **self.metadata}
        else:
            self._session_uid_chain = [self._uid]

        # 5. Decorate span with session attributes
        self._decorate_span()

        # 6. Write our context into baggage for downstream services
        self._write_to_baggage()

        # 7. Register this session in OTel context so LLM clients can retrieve it
        self._register_in_context()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context: clean up baggage and close span."""
        self._unregister_from_context()
        self._clear_baggage()
        self._close_span(exc_type, exc_val, exc_tb)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def llm_calls(self) -> list[Trace]:
        """Fetch traces associated with this session UID from the SQLite store.

        Returns traces from this session and all nested child sessions, including
        those created in downstream services (thanks to baggage propagation).
        """
        store = self._ensure_store()
        trace_contexts = async_to_sync(store.get_by_session_uid)(self._uid)
        traces: list[Trace] = []
        for ctx in trace_contexts:
            data = dict(ctx.data)
            data.setdefault("trace_id", ctx.id)
            traces.append(Trace(**data))
        return traces

    @property
    def trace_id(self) -> str | None:
        """Return the hex trace ID of the active span."""
        return self._trace_id

    @property
    def span_id(self) -> str | None:
        """Return the hex span ID of the active span."""
        return self._span_id

    def to_context_payload(self) -> dict[str, Any]:
        """Return metadata for manual context forwarding (rarely needed with baggage)."""
        return {
            "session_name": self.name,
            "session_uid": self._uid,
            "session_uid_chain": list(self._session_uid_chain),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:
        """Return number of LLM calls in this session."""
        return len(self.llm_calls)

    def __repr__(self):
        return f"OpenTelemetrySession(name={self.name!r}, _uid={self._uid!r}, chain_depth={len(self._session_uid_chain)}, trace_id={self.trace_id})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_tracer(self) -> _OtelTracer:
        if self._otel_tracer is not None:
            return self._otel_tracer
        if otel_trace is None:  # pragma: no cover - import guarded
            raise RuntimeError("OpenTelemetrySession requires the 'opentelemetry-api' package. Install it via `pip install opentelemetry-api` or pass a tracer explicitly.")
        self._otel_tracer = otel_trace.get_tracer(self._tracer_name)
        return self._otel_tracer

    def _ensure_store(self):
        if self._store is not None:
            return self._store
        from rllm.sdk.store import SqliteTraceStore  # Lazy import to avoid circular dependency

        self._store = SqliteTraceStore(db_path=self._store_db_path)
        return self._store

    def _cache_span_context(self) -> None:
        """Extract trace and span IDs from the active OpenTelemetry span."""
        if self._span is None:
            return
        span_context = self._span.get_span_context()
        self._trace_id = f"{span_context.trace_id:032x}"
        self._span_id = f"{span_context.span_id:016x}"

    def _decorate_span(self) -> None:
        """Attach session metadata as span attributes."""
        if self._span is None:
            return
        attributes: dict[str, Any] = {
            "session.name": self.name,
            "session.uid": self._uid,
            "session.uid_chain": list(self._session_uid_chain),
        }
        if self.metadata:
            attributes["session.metadata"] = json.dumps(self.metadata, sort_keys=True, default=str)
        for key, value in attributes.items():
            try:
                self._span.set_attribute(key, value)
            except Exception:  # pragma: no cover - defensive
                continue

    def _close_span(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """Close the OpenTelemetry span."""
        if self._span_scope is not None:
            self._span_scope.__exit__(exc_type, exc_val, exc_tb)
            self._span_scope = None
            self._span = None
        self._trace_id = None
        self._span_id = None

    def _read_from_baggage(self) -> tuple[list[str] | None, dict[str, Any]]:
        """Read parent session context from W3C baggage.

        Returns:
            Tuple of (parent_uid_chain, parent_metadata)
        """
        if otel_baggage is None:
            return None, {}

        baggage_val = otel_baggage.get_baggage("rllm-session")
        if not baggage_val:
            return None, {}

        try:
            ctx = json.loads(baggage_val)
            parent_chain = ctx.get("session_uid_chain", [])
            parent_meta = ctx.get("metadata", {})
            return parent_chain if parent_chain else None, parent_meta
        except Exception:
            # If baggage is malformed, proceed without inheritance
            return None, {}

    def _write_to_baggage(self) -> None:
        """Write current session context into W3C baggage for downstream propagation."""
        if otel_baggage is None or otel_context is None:
            return

        payload = {
            "session_name": self.name,
            "session_uid_chain": list(self._session_uid_chain),
            "metadata": dict(self.metadata),
        }
        baggage_str = json.dumps(payload, sort_keys=True, default=str)

        # Set baggage in a new context and attach it
        ctx = otel_baggage.set_baggage("rllm-session", baggage_str)
        self._baggage_token = otel_context.attach(ctx)

    def _clear_baggage(self) -> None:
        """Remove session baggage on exit."""
        if otel_context is None or self._baggage_token is None:
            return

        try:
            otel_context.detach(self._baggage_token)
        except Exception:
            pass
        self._baggage_token = None

    def _register_in_context(self) -> None:
        """Register this session in OpenTelemetry context."""
        if otel_context is None:
            return

        # Store session reference in current OTel context
        ctx = otel_context.get_current()
        new_ctx = otel_context.set_value(_SESSION_CONTEXT_KEY, self, ctx)
        self._session_context_token = otel_context.attach(new_ctx)

    def _unregister_from_context(self) -> None:
        """Remove this session from OpenTelemetry context."""
        if otel_context is None:
            return

        if hasattr(self, "_session_context_token") and self._session_context_token is not None:
            try:
                otel_context.detach(self._session_context_token)
            except Exception:
                pass
            self._session_context_token = None


def get_current_otel_session() -> OpenTelemetrySession | None:
    """Retrieve the current OpenTelemetrySession from OTel context.

    This uses OpenTelemetry's context propagation to find the active session,
    enabling distributed session tracking across process boundaries.

    Returns:
        Current OpenTelemetrySession if one is active, None otherwise.
    """
    if otel_context is None:
        return None

    try:
        return otel_context.get_value(_SESSION_CONTEXT_KEY)
    except Exception:
        return None
