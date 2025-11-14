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
    - InMemoryStorage: Thread-safe, single-process, ephemeral storage
    - SqliteSessionStorage: Durable, multi-process via shared SQLite file
    - PostgresStorage: Distributed, scalable

    Thread-safety requirement:
    - Implementations SHOULD be thread-safe if they might be shared across threads
    - InMemoryStorage uses locks to ensure thread-safety
    - SqliteSessionStorage uses SQLite's built-in locking

    Tree hierarchy support:
    Storage backends use session_uid_chain to enable parent sessions to
    see all descendant traces:
    - InMemoryStorage: Stores trace under all UIDs in chain
    - SqliteSessionStorage: Uses junction table with all UIDs in chain

    Example session hierarchy:
    - Root session: ["ctx_aaa"]
    - Child session: ["ctx_aaa", "ctx_bbb"]
    - Grandchild: ["ctx_aaa", "ctx_bbb", "ctx_ccc"]

    Querying "ctx_aaa" returns traces from all three sessions!
    """

    def add_trace(self, session_uid_chain: list[str], session_id: str, trace: Trace) -> None:
        """
        Add a trace to storage associated with a session hierarchy.

        Args:
            session_uid_chain: List of session UIDs from root to current (e.g., ["ctx_root", "ctx_child"])
            session_id: User-visible session ID (for logging/debugging)
            trace: Trace object to store
        """
        ...

    def get_traces(self, session_uid: str, session_id: str) -> list[Trace]:
        """
        Retrieve all traces for a session (includes all descendant sessions).

        Args:
            session_uid: Session UID to query (returns this session + all descendants)
            session_id: User-visible session ID (for logging/debugging)

        Returns:
            List of Trace objects for this session and all descendants
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

    def add_trace(self, session_uid_chain: list[str], session_id: str, trace: Trace) -> None:
        """
        Add trace to in-memory storage (thread-safe).

        Stores the trace under ALL session UIDs in the chain, enabling
        parent sessions to query all descendant traces.

        Args:
            session_uid_chain: List of session UIDs from root to current
            session_id: User-visible session ID (for logging/debugging)
            trace: Trace object to store
        """
        with self._lock:
            # Store trace under all session UIDs in the chain
            # This enables tree queries: parent sees all descendant traces
            for uid in session_uid_chain:
                self._traces[uid].append(trace)

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

    def add_trace(self, session_uid_chain: list[str], session_id: str, trace: Trace) -> None:
        """
        Add trace to SQLite storage (async operation, returns immediately).

        Uses the session UID chain to store traces in the junction table,
        enabling tree queries across process boundaries.

        Note: This method queues the trace for async storage. The actual
        storage happens in a background task. Use flush() or await close()
        to ensure all traces are written.

        Args:
            session_uid_chain: List of session UIDs from root to current
            session_id: User-visible session ID (for logging/debugging)
            trace: Trace object to store
        """
        # Create an async task to store the trace
        # We need to run this in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the coroutine
            loop.create_task(self._async_add_trace(session_uid_chain, trace))
        except RuntimeError:
            # No running loop, use asyncio.run() in a thread to avoid blocking
            import threading

            def _run_async():
                asyncio.run(self._async_add_trace(session_uid_chain, trace))

            thread = threading.Thread(target=_run_async, daemon=True)
            thread.start()

    async def _async_add_trace(self, session_uid_chain: list[str], trace: Trace) -> None:
        """Async helper to store trace to SQLite.

        Args:
            session_uid_chain: List of session UIDs from root to current
            trace: Trace object to store
        """
        try:
            await self.store.store(
                trace_id=trace.trace_id,
                data=trace.model_dump(),
                namespace="default",
                context_type="llm_trace",
                metadata={"session_id": trace.session_id},
                session_uids=session_uid_chain,  # Pass full chain to junction table!
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception(f"Failed to store trace {trace.trace_id}: {e}")

    def get_traces(self, session_uid: str, session_id: str) -> list[Trace]:
        """
        Retrieve all traces for a session from SQLite.

        Uses session_uid to query the junction table, returning all traces
        stored under this UID (including descendant sessions in the tree).

        This method runs the async query synchronously using asyncio.run().

        Args:
            session_uid: Unique session context UID (used as storage key)
            session_id: User-visible session ID (for logging/debugging)

        Returns:
            List of Trace objects for this session and all descendants
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
        """Async helper to retrieve traces from SQLite.

        Args:
            session_uid: Session context UID used as storage key

        Returns:
            List of Trace objects for this session and all descendants
        """
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
            logger.exception(f"Failed to retrieve traces for session UID {session_uid}: {e}")
            return []

    def __repr__(self):
        return f"SqliteSessionStorage(db_path={self.store.db_path!r})"
