"""Session management for RLLM SDK."""

from rllm.sdk.session.base import SessionProtocol
from rllm.sdk.session.contextvar import (
    ContextVarSession,
    get_active_sessions,
    get_current_metadata,
    get_current_session,
    get_current_session_name,
)
from rllm.sdk.session.otel import (
    OTelSession,
    get_active_otel_sessions,
    get_current_otel_session,
    get_otel_metadata,
    get_otel_session_name,
    init_otel_distributed_tracing,
    ray_entrypoint,
)
from rllm.sdk.session.storage import (
    InMemoryStorage,
    SessionStorage,
    SqliteSessionStorage,
)

# Default session type (alias for convenience)
SessionContext = ContextVarSession

__all__ = [
    # Protocol
    "SessionProtocol",
    # Implementations
    "ContextVarSession",
    "OTelSession",  # OpenTelemetry-based session for distributed tracing
    "SessionContext",  # Default alias (ContextVarSession)
    # Context helpers (ContextVarSession)
    "get_current_session",
    "get_current_session_name",
    "get_current_metadata",
    "get_active_sessions",
    # Context helpers (OTelSession)
    "get_current_otel_session",
    "get_otel_session_name",
    "get_otel_metadata",
    "get_active_otel_sessions",
    # OTel setup
    "init_otel_distributed_tracing",
    # Ray helpers
    "ray_entrypoint",
    # Storage
    "SessionStorage",  # Protocol
    "InMemoryStorage",  # Default in-memory storage
    "SqliteSessionStorage",  # SQLite-backed storage
]
