"""Tracer implementations for RLLM SDK."""

from rllm.sdk.tracers.base import CompositeTracer, TracerProtocol
from rllm.sdk.tracers.episodic import ContextStoreProtocol, EpisodicTracer
from rllm.sdk.tracers.memory import InMemorySessionTracer

__all__ = [
    # Protocol
    "TracerProtocol",
    # Implementations
    "InMemorySessionTracer",  # In-memory tracer for immediate access
    "EpisodicTracer",  # Persistent tracer with Episodic Context Store backend
    # Utilities
    "CompositeTracer",  # Combine multiple tracers
    # Context Store Protocol (for episodic tracer)
    "ContextStoreProtocol",
]
