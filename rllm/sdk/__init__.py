"""RLLM SDK for automatic LLM trace collection and RL training."""

from rllm.sdk.decorators import (
    StepContext,
    TrajectoryContext,
    step,
    step_context,
    trajectory,
    trajectory_context,
)
from rllm.sdk.protocol import StepView, Trace, TrajectoryView
from rllm.sdk.reward import set_reward, set_reward_async
from rllm.sdk.session import (
    ContextVarSession,
    InMemoryStorage,
    SessionContext,
    SessionStorage,
    SqliteSessionStorage,
    get_current_metadata,
    get_current_session,
    get_current_session_name,
)
from rllm.sdk.shortcuts import get_chat_client, get_chat_client_async, session
from rllm.sdk.tracers import (
    InMemorySessionTracer,
    SqliteTracer,
    TracerProtocol,
)

__all__ = [
    # Protocol / Data Models
    "Trace",  # Low-level LLM call trace
    "StepView",  # High-level step view (semantic unit)
    "TrajectoryView",  # Collection of steps forming a workflow
    # Decorators
    "step",  # Decorator to mark function as a step (returns StepView)
    "trajectory",  # Decorator to mark function as trajectory (returns TrajectoryView)
    "step_context",  # Context manager for step (doesn't change return)
    "trajectory_context",  # Context manager for trajectory (doesn't change return)
    "StepContext",  # Step context manager class
    "TrajectoryContext",  # Trajectory context manager class
    # Sessions
    "SessionContext",  # Default (alias for ContextVarSession)
    "ContextVarSession",  # Explicit contextvars-based session
    "get_current_session",  # Get current session instance
    "get_current_session_name",  # Get current session name
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
    "SqliteTracer",  # SQLite-based persistent tracer
    # Rewards
    "set_reward",
    "set_reward_async",
]
