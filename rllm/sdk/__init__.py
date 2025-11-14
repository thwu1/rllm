"""RLLM SDK for automatic LLM trace collection and RL training."""

from rllm.sdk.reward import set_reward, set_reward_async
from rllm.sdk.session import (
    ContextVarSession,
    SessionContext,
    get_current_metadata,
    get_current_session,
    get_current_session_id,
)
from rllm.sdk.shortcuts import get_chat_client, get_chat_client_async, session
from rllm.sdk.tracers import (
    ContextStoreProtocol,
    EpisodicTracer,
    InMemorySessionTracer,
    TracerProtocol,
)

__all__ = [
    # Sessions
    "SessionContext",  # Default (alias for ContextVarSession)
    "ContextVarSession",  # Explicit contextvars-based session
    "get_current_session",  # Get current session instance
    "get_current_session_id",  # Get current session ID
    "get_current_metadata",  # Get current metadata
    # Shortcuts
    "session",
    "get_chat_client",
    "get_chat_client_async",
    # Tracers
    "TracerProtocol",  # Tracer interface
    "InMemorySessionTracer",  # In-memory tracer for immediate access
    "EpisodicTracer",  # Persistent tracer with Episodic backend
    "ContextStoreProtocol",  # Context store protocol for episodic tracer
    # Rewards
    "set_reward",
    "set_reward_async",
]
