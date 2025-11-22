"""OpenTelemetry-backed session implementation using W3C baggage as single source of truth.

Design:
- Baggage is the ONLY source of truth for session state
- Session object is a thin wrapper that manages baggage lifecycle
- All getters read directly from baggage (works in-process AND cross-process)
- Span attributes are write-only copies for observability tools
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync  # type: ignore[import]
from opentelemetry import baggage as otel_baggage  # type: ignore[import]
from opentelemetry import context as otel_context  # type: ignore[import]
from opentelemetry import propagate as otel_propagate  # type: ignore[import]
from opentelemetry import trace as otel_trace  # type: ignore[import]
from opentelemetry.baggage.propagation import W3CBaggagePropagator  # type: ignore[import]
from opentelemetry.propagators.composite import CompositePropagator  # type: ignore[import]
from opentelemetry.sdk.resources import Resource  # type: ignore[import]
from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator  # type: ignore[import]

from rllm.sdk.protocol import Trace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opentelemetry.trace import Span as _OtelSpan  # type: ignore[import]
    from opentelemetry.trace import Tracer as _OtelTracer  # type: ignore[import]

    from rllm.sdk.store import SqliteTraceStore
else:  # pragma: no cover - fallback for runtime when otel not installed
    _OtelTracer = Any  # type: ignore
    _OtelSpan = Any  # type: ignore

_OTEL_RUNTIME_CONFIGURED = False
_BAGGAGE_KEY = "rllm-session"


def configure_default_tracer(service_name: str = "rllm-worker") -> None:
    """Install a default tracer provider + baggage propagation once per process."""
    global _OTEL_RUNTIME_CONFIGURED
    if _OTEL_RUNTIME_CONFIGURED:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    otel_trace.set_tracer_provider(provider)

    otel_propagate.set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )

    _OTEL_RUNTIME_CONFIGURED = True


# ---------------------------------------------------------------------------
# Baggage helpers - single source of truth
# ---------------------------------------------------------------------------
def _read_baggage() -> dict[str, Any]:
    """Read session context from W3C baggage (the single source of truth).

    Returns:
        Dict with keys: session_uid_chain, metadata (includes session_name)
        Empty dict if no baggage is set.
    """
    baggage_val = otel_baggage.get_baggage(_BAGGAGE_KEY)
    if not isinstance(baggage_val, str):
        return {}

    text = baggage_val.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _write_baggage(payload: dict[str, Any]) -> object:
    """Write session context to W3C baggage.

    Args:
        payload: Dict with session_uid_chain, metadata

    Returns:
        Context token for later detachment
    """
    baggage_str = json.dumps(payload, sort_keys=True, default=str)
    ctx = otel_baggage.set_baggage(_BAGGAGE_KEY, baggage_str)
    return otel_context.attach(ctx)


def _detach_baggage(token: object | None) -> None:
    """Detach baggage context."""
    if token is not None:
        otel_context.detach(token)


# ---------------------------------------------------------------------------
# Public getters - all read from baggage
# ---------------------------------------------------------------------------
def get_current_otel_session_name() -> str | None:
    """Get current session name from baggage.

    Returns:
        Session name, or None if no session is active.
    """
    ctx = _read_baggage()
    metadata = ctx.get("metadata", {})
    name = metadata.get("session_name")
    return name if isinstance(name, str) else None


def get_current_otel_metadata() -> dict[str, Any]:
    """Get current metadata from baggage.

    Returns:
        Session metadata dict, or empty dict if no session is active.
    """
    ctx = _read_baggage()
    metadata = ctx.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def get_active_otel_session_uids() -> list[str]:
    """Get active session UID chain from baggage.

    Returns:
        List of session UIDs from root to current, or empty list.
    """
    ctx = _read_baggage()
    chain = ctx.get("session_uid_chain", [])
    if isinstance(chain, list):
        return [str(x) for x in chain]
    return []


def get_current_otel_session() -> OpenTelemetrySession | None:
    """Check if there's an active session (based on baggage).

    Note: This returns None since we don't store session objects anymore.
    Use get_current_otel_metadata() or get_current_otel_session_name() instead.

    Returns:
        None (session objects are not stored in context)
    """
    # We no longer store session objects in context.
    # Check if baggage has session data instead.
    ctx = _read_baggage()
    if ctx.get("session_uid_chain"):
        # There's session data in baggage, but we don't have the object
        # Return None - callers should use the getter functions instead
        return None
    return None


def otel_session(**kwargs: Any) -> OpenTelemetrySession:
    """Convenience factory for creating OpenTelemetry sessions.

    Example:
        >>> with otel_session(name="my-task", env="prod") as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))
    """
    return OpenTelemetrySession(**kwargs)


class OpenTelemetrySession:
    """Session implementation using W3C baggage as single source of truth.

    Design principles:
    - Baggage is the ONLY authoritative source for session state
    - Session object manages baggage lifecycle (enter/exit)
    - Properties read from baggage (always current, works cross-process)
    - Span attributes are write-only copies for observability

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
            name: Session name (inherited from parent if not provided)
            tracer: OpenTelemetry tracer instance (uses global tracer if None)
            tracer_name: Tracer name for span creation
            store: SqliteTraceStore instance for querying LLM calls
            store_db_path: Path to SQLite database (used if store is None)
            metadata: Session metadata
            **extra_metadata: Additional metadata as keyword arguments
        """
        # Store initial values - will be merged with parent in __enter__
        self._init_name = name
        self._init_metadata = dict(metadata or {})
        self._init_metadata.update(extra_metadata)

        # These will be set in __enter__
        self._uid: str = ""
        self._session_uid_chain: list[str] = []
        self._merged_metadata: dict[str, Any] = {}

        # OTel tracer
        self._otel_tracer: _OtelTracer | None = tracer
        self._tracer_name = tracer_name
        self._span_scope = None
        self._span: _OtelSpan | None = None
        self._trace_id: str | None = None
        self._span_id: str | None = None

        # Storage
        self._store = store
        self._store_db_path = store_db_path

        # Context tokens for cleanup
        self._baggage_token: object | None = None

    # ------------------------------------------------------------------
    # Properties - read from instance (which mirrors baggage)
    # ------------------------------------------------------------------
    @property
    def name(self) -> str | None:
        """Session name."""
        return self._merged_metadata.get("session_name")

    @property
    def metadata(self) -> dict[str, Any]:
        """Session metadata (includes session_name)."""
        return dict(self._merged_metadata)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        """Enter session context: read parent baggage, merge, write new baggage."""
        # 1. Read parent context from baggage
        parent_ctx = _read_baggage()
        parent_chain = parent_ctx.get("session_uid_chain", [])
        parent_meta = parent_ctx.get("metadata", {})

        # 2. Start OpenTelemetry span (before getting UID)
        tracer = self._ensure_tracer()
        span_name = f"session:{self._init_name}" if self._init_name else "session"
        self._span_scope = tracer.start_as_current_span(span_name)
        self._span = self._span_scope.__enter__()

        # 3. Extract span context and use span ID as session UID
        self._cache_span_context()
        self._uid = self._span_id or f"ctx_{uuid.uuid4().hex[:16]}"

        # 4. Build UID chain: parent chain + our UID
        if isinstance(parent_chain, list) and parent_chain:
            self._session_uid_chain = [str(x) for x in parent_chain] + [self._uid]
        else:
            self._session_uid_chain = [self._uid]

        # 5. Merge metadata: parent values inherited, child values override
        self._merged_metadata = {}
        if isinstance(parent_meta, dict):
            self._merged_metadata.update(parent_meta)
        self._merged_metadata.update(self._init_metadata)
        if self._init_name is not None:
            self._merged_metadata["session_name"] = self._init_name

        # 6. Write to baggage (SINGLE SOURCE OF TRUTH)
        self._baggage_token = _write_baggage({
            "session_uid_chain": self._session_uid_chain,
            "metadata": self._merged_metadata,
        })

        # 7. Decorate span with session attributes (write-only, for observability)
        self._decorate_span()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context: restore parent baggage and close span."""
        # Detach our baggage (restores parent context)
        _detach_baggage(self._baggage_token)
        self._baggage_token = None

        # Close span
        self._close_span(exc_type, exc_val, exc_tb)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def llm_calls(self) -> list[Trace]:
        """Fetch traces associated with this session UID from the SQLite store.

        Returns traces from this session and all nested child sessions.
        """
        store = self._ensure_store()
        trace_contexts = async_to_sync(store.get_by_session_uid)(self._uid)
        traces: list[Trace] = []
        for ctx in trace_contexts:
            data = dict(ctx.data)
            data.setdefault("trace_id", ctx.id)
            traces.append(Trace(**data))
        return traces

    async def llm_calls_async(self) -> list[Trace]:
        """Async variant of llm_calls for use within running event loops."""
        store = self._ensure_store()
        trace_contexts = await store.get_by_session_uid(self._uid)
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
            "session_uid": self._uid,
            "session_uid_chain": list(self._session_uid_chain),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": dict(self._merged_metadata),
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
        """Attach session metadata as span attributes (write-only, for observability)."""
        if self._span is None:
            return
        attributes: dict[str, Any] = {
            "session.name": self.name,
            "session.uid": self._uid,
            "session.uid_chain": list(self._session_uid_chain),
        }
        if self._merged_metadata:
            attributes["session.metadata"] = json.dumps(self._merged_metadata, sort_keys=True, default=str)
        for key, value in attributes.items():
            self._span.set_attribute(key, value)

    def _close_span(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """Close the OpenTelemetry span."""
        if self._span_scope is not None:
            self._span_scope.__exit__(exc_type, exc_val, exc_tb)
            self._span_scope = None
            self._span = None
        self._trace_id = None
        self._span_id = None
