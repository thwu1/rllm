"""Storage protocols and implementations for session trace storage."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from rllm.sdk.protocol import Trace


@runtime_checkable
class SessionStorage(Protocol):
    """
    Protocol for session storage backends.

    Separates the concern of storing/retrieving traces from session context propagation.
    Different implementations can provide different storage strategies:
    - InMemoryStorage: Fast, single-process, ephemeral
    - SqliteSessionStorage: Durable, multi-process via shared SQLite file
    - PostgresStorage: Distributed, scalable

    All storage implementations must support querying by session_uid (the internal
    unique ID of each session context instance).
    """

    def add_trace(self, session_uid: str, trace: Trace) -> None:
        """
        Add a trace to storage associated with a session UID.

        Args:
            session_uid: Unique ID of the session context (_uid field)
            trace: Trace object to store
        """
        ...

    def get_traces(self, session_uid: str) -> list[Trace]:
        """
        Retrieve all traces for a session UID.

        Args:
            session_uid: Unique ID of the session context (_uid field)

        Returns:
            List of Trace objects associated with this session UID
        """
        ...


class InMemoryStorage:
    """
    In-memory storage for session traces.

    Fast and simple, but:
    - Only works within a single process
    - All data lost when process exits
    - Not suitable for multiprocessing scenarios

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
        """Initialize in-memory storage."""
        self._traces: dict[str, list[Trace]] = defaultdict(list)

    def add_trace(self, session_uid: str, trace: Trace) -> None:
        """Add trace to in-memory storage."""
        self._traces[session_uid].append(trace)

    def get_traces(self, session_uid: str) -> list[Trace]:
        """Get all traces for a session UID."""
        return self._traces[session_uid].copy()

    def clear(self, session_uid: str) -> None:
        """Clear all traces for a session UID."""
        if session_uid in self._traces:
            self._traces[session_uid].clear()

    def __repr__(self):
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

    def add_trace(self, session_uid: str, trace: Trace) -> None:
        """
        Add trace to SQLite storage (async operation, returns immediately).

        Note: This method queues the trace for async storage. The actual
        storage happens in a background task. Use flush() or await close()
        to ensure all traces are written.

        Args:
            session_uid: Session UID to associate with this trace
            trace: Trace object to store
        """
        # Create an async task to store the trace
        # We need to run this in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the coroutine
            loop.create_task(self._async_add_trace(session_uid, trace))
        except RuntimeError:
            # No running loop, use asyncio.run() in a thread to avoid blocking
            import threading

            def _run_async():
                asyncio.run(self._async_add_trace(session_uid, trace))

            thread = threading.Thread(target=_run_async, daemon=True)
            thread.start()

    async def _async_add_trace(self, session_uid: str, trace: Trace) -> None:
        """Async helper to store trace to SQLite."""
        try:
            await self.store.store(
                trace_id=trace.trace_id,
                data=trace.model_dump(),
                namespace="default",
                context_type="llm_trace",
                metadata={"session_id": trace.session_id},
                session_uids=[session_uid],
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception(f"Failed to store trace {trace.trace_id}: {e}")

    def get_traces(self, session_uid: str) -> list[Trace]:
        """
        Retrieve all traces for a session UID from SQLite.

        This method runs the async query synchronously using asyncio.run().

        Args:
            session_uid: Session UID to query

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
                future = executor.submit(asyncio.run, self._async_get_traces(session_uid))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._async_get_traces(session_uid))

    async def _async_get_traces(self, session_uid: str) -> list[Trace]:
        """Async helper to retrieve traces from SQLite."""
        try:
            trace_contexts = await self.store.get_by_session_uid(session_uid)

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
            logger.exception(f"Failed to retrieve traces for session {session_uid}: {e}")
            return []

    def __repr__(self):
        return f"SqliteSessionStorage(db_path={self.store.db_path!r})"
