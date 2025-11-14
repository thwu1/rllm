"""Storage protocols and implementations for session trace storage."""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from rllm.sdk.protocol import Trace


@runtime_checkable
class SessionStorage(Protocol):
    """
    Protocol for session storage backends.

    Separates the concern of storing/retrieving traces from session context propagation.
    Different implementations can provide different storage strategies:
    - InMemoryStorage: Thread-safe, single-process, ephemeral (keys by instance _uid)
    - SqliteSessionStorage: Durable, multi-process via shared SQLite file (keys by session_id)
    - PostgresStorage: Distributed, scalable

    Thread-safety requirement:
    - Implementations SHOULD be thread-safe if they might be shared across threads
    - InMemoryStorage uses locks to ensure thread-safety
    - SqliteSessionStorage uses SQLite's built-in locking

    Storage backends decide which identifier to use:
    - Instance-based (InMemoryStorage): Uses session_uid for isolation per instance
    - Session-based (SqliteSessionStorage): Uses session_id for sharing across processes
    """

    def add_trace(self, session_uid: str, session_id: str, trace: Trace) -> None:
        """
        Add a trace to storage associated with a session.

        Args:
            session_uid: Unique ID of the session context instance (_uid field)
            session_id: User-visible session ID (session_id field)
            trace: Trace object to store
        """
        ...

    def get_traces(self, session_uid: str, session_id: str) -> list[Trace]:
        """
        Retrieve all traces for a session.

        Args:
            session_uid: Unique ID of the session context instance (_uid field)
            session_id: User-visible session ID (session_id field)

        Returns:
            List of Trace objects for this session
        """
        ...


class InMemoryStorage:
    """
    Thread-safe in-memory storage for session traces.

    Features:
    - Thread-safe: Uses locks to prevent race conditions
    - Fast: In-memory storage with minimal overhead
    - Simple: No external dependencies

    Limitations:
    - Only works within a single process (not multiprocessing-safe)
    - All data lost when process exits
    - For multiprocessing scenarios, use SqliteSessionStorage

    This is the default storage when no explicit storage is provided,
    maintaining backward compatibility with the original behavior.

    Example:
        >>> from rllm.sdk.session import ContextVarSession
        >>> from rllm.sdk.session.storage import InMemoryStorage
        >>>
        >>> storage = InMemoryStorage()
        >>> with ContextVarSession(storage=storage) as session:
        ...     # All traces stored in memory
        ...     llm.chat.completions.create(...)
        ...     print(session.llm_calls)  # Immediate access
    """

    def __init__(self):
        """Initialize thread-safe in-memory storage."""
        self._traces: dict[str, list[Trace]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_trace(self, session_uid: str, session_id: str, trace: Trace) -> None:
        """
        Add trace to in-memory storage (thread-safe).

        Uses session_uid for instance-level isolation (nested sessions get separate storage).

        Args:
            session_uid: Unique ID of the session context (used as storage key)
            session_id: User-visible session ID (ignored, for protocol compatibility)
            trace: Trace object to store
        """
        with self._lock:
            self._traces[session_uid].append(trace)

    def get_traces(self, session_uid: str, session_id: str) -> list[Trace]:
        """
        Get all traces for a session UID (thread-safe).

        Uses session_uid for instance-level isolation.

        Args:
            session_uid: Unique ID of the session context (used as storage key)
            session_id: User-visible session ID (ignored, for protocol compatibility)

        Returns:
            Copy of the trace list for this session
        """
        with self._lock:
            return self._traces[session_uid].copy()

    def clear(self, session_uid: str, session_id: str) -> None:
        """
        Clear all traces for a session UID (thread-safe).

        Args:
            session_uid: Unique ID of the session context
            session_id: User-visible session ID (ignored, for protocol compatibility)
        """
        with self._lock:
            if session_uid in self._traces:
                self._traces[session_uid].clear()

    def __repr__(self):
        with self._lock:
            total_traces = sum(len(traces) for traces in self._traces.values())
            return f"InMemoryStorage(sessions={len(self._traces)}, total_traces={total_traces})"


class SqliteSessionStorage:
    """
    SQLite-backed storage for session traces.

    Uses SqliteTraceStore for durable, multi-process-safe trace storage.
    Traces are stored asynchronously but retrieval is synchronous and uses
    an asyncio helper to run the async query.

    Features:
    - Durable storage (survives process restarts)
    - Multi-process safe (via shared SQLite file)
    - Fast queries by session_uid (indexed junction table)
    - Works across threads and processes

    Example:
        >>> from rllm.sdk.session import ContextVarSession
        >>> from rllm.sdk.session.storage import SqliteSessionStorage
        >>>
        >>> storage = SqliteSessionStorage("traces.db")
        >>> with ContextVarSession(storage=storage) as session:
        ...     llm.chat.completions.create(...)
        ...
        ...     # Query from SQLite
        ...     print(session.llm_calls)
        >>>
        >>> # In another process with same storage
        >>> storage2 = SqliteSessionStorage("traces.db")
        >>> # Can retrieve traces from first process!
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize SQLite session storage.

        Args:
            db_path: Path to SQLite database file. If None, uses default location
                    (~/.rllm/traces.db or temp directory)
        """
        from rllm.sdk.store import SqliteTraceStore

        self.store = SqliteTraceStore(db_path=db_path)

    def add_trace(self, session_uid: str, session_id: str, trace: Trace) -> None:
        """
        Add trace to SQLite storage (async operation, returns immediately).

        Uses session_id for cross-process sharing. All sessions with the same
        session_id will see the same traces, regardless of process boundaries.

        Note: This method queues the trace for async storage. The actual
        storage happens in a background task. Use flush() or await close()
        to ensure all traces are written.

        Args:
            session_uid: Unique instance ID (ignored, for protocol compatibility)
            session_id: User-visible session ID (used as storage key)
            trace: Trace object to store
        """
        # Create an async task to store the trace
        # We need to run this in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the coroutine
            loop.create_task(self._async_add_trace(session_id, trace))
        except RuntimeError:
            # No running loop, use asyncio.run() in a thread to avoid blocking
            import threading

            def _run_async():
                asyncio.run(self._async_add_trace(session_id, trace))

            thread = threading.Thread(target=_run_async, daemon=True)
            thread.start()

    async def _async_add_trace(self, session_id: str, trace: Trace) -> None:
        """Async helper to store trace to SQLite.

        Args:
            session_id: User-visible session ID used as storage key
            trace: Trace object to store
        """
        try:
            await self.store.store(
                trace_id=trace.trace_id,
                data=trace.model_dump(),
                namespace="default",
                context_type="llm_trace",
                metadata={"session_id": trace.session_id},
                session_uids=[session_id],  # Use session_id for cross-process sharing
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception(f"Failed to store trace {trace.trace_id}: {e}")

    def get_traces(self, session_uid: str, session_id: str) -> list[Trace]:
        """
        Retrieve all traces for a session from SQLite.

        Uses session_id for cross-process sharing. All sessions with the same
        session_id will see the same traces, regardless of process boundaries.

        This method runs the async query synchronously using asyncio.run().

        Args:
            session_uid: Unique instance ID (ignored, for protocol compatibility)
            session_id: User-visible session ID (used as storage key)

        Returns:
            List of Trace objects for this session
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # We're in an async context, but this is a sync method being called
            # We need to run in a separate thread to avoid blocking
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._async_get_traces(session_id))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._async_get_traces(session_id))

    async def _async_get_traces(self, session_id: str) -> list[Trace]:
        """Async helper to retrieve traces from SQLite.

        Args:
            session_id: User-visible session ID used as storage key

        Returns:
            List of Trace objects for this session
        """
        try:
            trace_contexts = await self.store.get_by_session_uid(session_id)

            # Convert TraceContext objects to Trace protocol objects
            traces = []
            for tc in trace_contexts:
                # tc.data is already a dict with all the trace fields
                trace = Trace(**tc.data)
                traces.append(trace)

            return traces
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception(f"Failed to retrieve traces for session {session_id}: {e}")
            return []

    def __repr__(self):
        return f"SqliteSessionStorage(db_path={self.store.db_path!r})"
