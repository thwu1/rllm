"""Tracer implementations for RLLM SDK."""

from rllm.sdk.tracers.base import TracerProtocol
from rllm.sdk.tracers.episodic import ContextStoreProtocol, EpisodicTracer
from rllm.sdk.tracers.memory import InMemorySessionTracer
from rllm.sdk.tracers.sqlite import SqliteTracer

__all__ = [
    # Protocol
    "TracerProtocol",
    # Implementations
    "InMemorySessionTracer",  # In-memory tracer for immediate access
    "EpisodicTracer",  # Persistent tracer with Episodic Context Store backend
    "SqliteTracer",  # Persistent tracer with SQLite backend
    # Context Store Protocol (for episodic tracer)
    "ContextStoreProtocol",
]
