"""RLLM SDK for automatic LLM trace collection and RL training."""

from rllm.sdk.reward import set_reward, set_reward_async
from rllm.sdk.session import (
    ContextVarSession,
    InMemoryStorage,
    SessionContext,
    SessionStorage,
    SqliteSessionStorage,
    get_current_metadata,
    get_current_session,
    get_current_session_id,
)
from rllm.sdk.shortcuts import get_chat_client, get_chat_client_async, session
from rllm.sdk.tracers import (
    CompositeTracer,
    ContextStoreProtocol,
    EpisodicTracer,
    InMemorySessionTracer,
    SqliteTracer,
    TracerProtocol,
)

__all__ = [
    # Sessions
    "SessionContext",  # Default (alias for ContextVarSession)
    "ContextVarSession",  # Explicit contextvars-based session
    "get_current_session",  # Get current session instance
    "get_current_session_id",  # Get current session ID
    "get_current_metadata",  # Get current metadata
    # Session Storage
    "SessionStorage",  # Storage protocol
    "InMemoryStorage",  # Default in-memory storage
    "SqliteSessionStorage",  # SQLite-backed storage
    # Shortcuts
    "session",
    "get_chat_client",
    "get_chat_client_async",
    # Tracers
    "TracerProtocol",  # Tracer interface
    "InMemorySessionTracer",  # In-memory tracer for immediate access
    "EpisodicTracer",  # Persistent tracer with Episodic backend
    "SqliteTracer",  # SQLite-based persistent tracer
    "CompositeTracer",  # Combine multiple tracers
    "ContextStoreProtocol",  # Context store protocol for episodic tracer
    # Rewards
    "set_reward",
    "set_reward_async",
]


# Lazy import for OpenTelemetry session (optional dependency)
def __getattr__(name):
    if name == "OTelSession":
        from rllm.sdk.session import OTelSession

        return OTelSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
