"""Ephemeral (in-memory) trace storage for session-scoped data.

This module provides non-persistent, session-scoped trace storage.
Traces are held in memory for the duration of the session and discarded afterward.

For persistent storage across sessions/processes, see rllm.sdk.store (SqliteTraceStore).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Protocol, runtime_checkable

from rllm.sdk.protocol import Trace


@runtime_checkable
class SessionStorage(Protocol):
    """Protocol for session trace storage backends.

    Implementations: InMemoryStorage (single-process).
    Uses session_uid_chain for hierarchy support - parent sessions see all descendant traces.
    """

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """Add trace to storage under session hierarchy."""
        ...

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Retrieve all traces for session (includes descendants)."""
        ...


class InMemoryStorage:
    """Thread-safe in-memory trace storage (default, single-process only).

    Fast ephemeral storage for single-process scenarios.
    """

    def __init__(self):
        """Initialize thread-safe in-memory storage."""
        self._traces: dict[str, list[Trace]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_trace(self, session_uid_chain: list[str], session_name: str, trace: Trace) -> None:
        """
        Add trace to in-memory storage (thread-safe).

        Stores the trace under ALL session UIDs in the chain, enabling
        parent sessions to query all descendant traces.

        Args:
            session_uid_chain: List of session UIDs from root to current
            session_name: User-visible session name (for logging/debugging)
            trace: Trace object to store
        """
        with self._lock:
            # Store trace under all session UIDs in the chain
            # This enables tree queries: parent sees all descendant traces
            for uid in session_uid_chain:
                self._traces[uid].append(trace)

    def get_traces(self, session_uid: str, session_name: str) -> list[Trace]:
        """Get all traces for session UID (thread-safe)."""
        with self._lock:
            return self._traces[session_uid].copy()

    def clear(self, session_uid: str, session_name: str) -> None:
        """Clear all traces for session UID (thread-safe)."""
        with self._lock:
            if session_uid in self._traces:
                self._traces[session_uid].clear()

    def __repr__(self):
        with self._lock:
            total_traces = sum(len(traces) for traces in self._traces.values())
            return f"InMemoryStorage(sessions={len(self._traces)}, total_traces={total_traces})"
