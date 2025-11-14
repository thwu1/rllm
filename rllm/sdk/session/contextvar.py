"""ContextVar-based session implementation (default)."""

import contextvars
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rllm.sdk.protocol import StepProto, Trace, trace_to_step_proto

if TYPE_CHECKING:
    from rllm.sdk.session.storage import SessionStorage

# Session-specific context variables
_current_session: contextvars.ContextVar["ContextVarSession | None"] = contextvars.ContextVar("current_session", default=None)
_session_name: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_name", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
# Stack of active sessions (outer → inner). Use None default to avoid shared list instances.
_sessions_stack: contextvars.ContextVar[list["ContextVarSession"] | None] = contextvars.ContextVar("sessions_stack", default=None)


def get_current_session() -> "ContextVarSession | None":
    """Get the current session instance from context."""
    return _current_session.get()


def get_current_session_name() -> str | None:
    """Get current session name from context."""
    return _session_name.get()


def get_current_metadata() -> dict[str, Any]:
    """Get current metadata from context."""
    metadata = _metadata.get()
    return metadata if metadata is not None else {}


def get_active_sessions() -> list["ContextVarSession"]:
    """Get a copy of the current stack of active sessions (outer → inner)."""
    stack = _sessions_stack.get() or []
    # Return a shallow copy to prevent accidental mutation by callers
    return list(stack)


class ContextVarSession:
    """
    Session implementation using Python contextvars with pluggable storage.

    This session implementation separates context propagation (session_id, metadata)
    from trace storage. The storage backend can be plugged in to support different
    deployment scenarios:

    - **InMemoryStorage** (default): Fast, single-process, ephemeral
    - **SqliteSessionStorage**: Durable, multi-process via shared SQLite file
    - **Custom storage**: Implement SessionStorage protocol

    Features:
    - Thread-safe and async-safe context propagation
    - Automatic nested session isolation with metadata inheritance
    - Pluggable storage backends
    - Backward compatible with existing code

    Example (in-memory, default):
        >>> from rllm.sdk.session import ContextVarSession
        >>> from rllm.sdk import get_chat_client
        >>>
        >>> llm = get_chat_client(api_key="...", model="gpt-4")
        >>>
        >>> # Default: in-memory storage
        >>> with ContextVarSession() as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))  # Immediate access

    Example (SQLite, multi-process):
        >>> from rllm.sdk.session import ContextVarSession
        >>> from rllm.sdk.session.storage import SqliteSessionStorage
        >>>
        >>> storage = SqliteSessionStorage("traces.db")
        >>>
        >>> # Process 1
        >>> with ContextVarSession(storage=storage, name="task-123") as session:
        ...     llm.chat.completions.create(...)
        >>>
        >>> # Process 2 (can access same traces!)
        >>> with ContextVarSession(storage=storage, name="task-123") as session:
        ...     llm.chat.completions.create(...)
        ...     print(session.llm_calls)  # Sees traces from both processes!
    """

    def __init__(
        self,
        name: str | None = None,
        storage: "SessionStorage | None" = None,
        formatter: Callable[[dict], dict] | None = None,
        persistent_tracers: list | None = None,
        _session_uid_chain: list[str] | None = None,
        **metadata,
    ):
        """
        Initialize contextvars-based session.

        Args:
            name: Session name (auto-generated if None). If None and there's an
                  existing session name in the context (from a parent session),
                  that will be inherited instead of generating a new one.
            storage: Storage backend for traces. If None, uses InMemoryStorage (default).
                    Pass SqliteSessionStorage for multi-process scenarios.
            formatter: Optional formatter to transform trace data (deprecated, kept for compatibility)
            persistent_tracers: Optional list of persistent tracers (deprecated, kept for compatibility)
            _session_uid_chain: Internal parameter for context restoration (do not use directly)
            **metadata: Session metadata
        """
        # If name is not explicitly provided, check if there's one in the context
        # (set by a parent _session_with_name). This allows internal code to control
        # the session name while keeping it hidden from users via the session() shortcut.
        if name is None:
            existing_name = _session_name.get()
            if existing_name is not None:
                name = existing_name

        # Generate new name only if none was provided and none exists in context
        self.name = name or f"sess_{uuid.uuid4().hex[:16]}"

        # Internal unique ID for this session context instance (different from name which can be inherited)
        # This allows tracking each unique session context even when name is shared
        self._uid = f"ctx_{uuid.uuid4().hex[:16]}"

        # Build session UID chain for tree hierarchy support
        if _session_uid_chain is not None:
            # Restoring from serialized context (distributed case)
            self._session_uid_chain = _session_uid_chain + [self._uid]
        else:
            # Check for parent session in current context (nested local case)
            parent_session = get_current_session()
            if parent_session is not None:
                # Inherit parent's chain and append our UID
                self._session_uid_chain = parent_session._session_uid_chain + [self._uid]
            else:
                # Root session - start new chain
                self._session_uid_chain = [self._uid]

        self.metadata = metadata
        self.formatter = formatter or (lambda x: x)

        # Storage backend (defaults to InMemoryStorage for backward compatibility)
        if storage is None:
            from rllm.sdk.session.storage import InMemoryStorage

            storage = InMemoryStorage()
        self.storage = storage

        # Optional persistent tracers (kept for backward compatibility)
        self._persistent_tracers = persistent_tracers or []

        # Context tokens for cleanup
        self._session_token = None
        self._session_name_token = None
        self._metadata_token = None
        self._stack_token = None

    @property
    def llm_calls(self) -> list[Trace]:
        """
        Get all LLM calls made within this session and descendant sessions.

        Queries the storage backend using the session UID, which enables
        tree hierarchy: parent sessions automatically see traces from all
        nested child sessions.

        For multi-process scenarios with SqliteSessionStorage, use
        to_context()/from_context() to propagate the session hierarchy:

        Example:
            # Process 1 - parent session
            storage = SqliteSessionStorage("traces.db")
            with ContextVarSession(storage=storage) as parent:
                llm.call()  # One trace

                # Nested local session
                with ContextVarSession() as child:
                    llm.call()  # Another trace

                # Parent sees both!
                assert len(parent.llm_calls) == 2

                # Send context to another process
                context = parent.to_context()
                send_to_process2(context)

            # Process 2 - restore session hierarchy
            context = receive_from_process1()
            with ContextVarSession.from_context(context, storage) as remote:
                llm.call()  # Parent in Process 1 will see this too!

        Returns:
            List of Trace objects for this session and all descendants
        """
        return self.storage.get_traces(self._uid, self.name)

    @property
    def steps(self) -> list[StepProto]:
        """Get all steps within this session."""
        return [trace_to_step_proto(trace) for trace in self.llm_calls]

    def clear_calls(self) -> None:
        """
        Clear all stored calls for this session.

        Only works with storage backends that support clearing (e.g., InMemoryStorage).
        SQLite and other persistent storage may not support this operation.
        """
        if hasattr(self.storage, "clear"):
            self.storage.clear(self._uid, self.name)

    def __enter__(self):
        """Enter session context - set up context variables."""
        # Set this session instance in context
        self._session_token = _current_session.set(self)

        # Set session name in context
        self._session_name_token = _session_name.set(self.name)

        # Merge and set metadata (inherits from parent)
        parent_meta = _metadata.get() or {}
        merged = {**parent_meta, **self.metadata}
        self._metadata_token = _metadata.set(merged)

        # Push onto sessions stack (outer → inner order)
        current_stack = _sessions_stack.get() or []
        # Create a new list to avoid mutating shared references
        new_stack = list(current_stack)
        new_stack.append(self)
        self._stack_token = _sessions_stack.set(new_stack)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - restore previous context."""
        if self._session_token is not None:
            _current_session.reset(self._session_token)
        if self._session_name_token is not None:
            _session_name.reset(self._session_name_token)
        if self._metadata_token is not None:
            _metadata.reset(self._metadata_token)

        # Restore previous sessions stack
        if self._stack_token is not None:
            _sessions_stack.reset(self._stack_token)

        return False

    def __len__(self) -> int:
        """Return number of calls in this session."""
        return len(self.llm_calls)

    def to_context(self) -> dict:
        """
        Serialize session context for distributed propagation.

        Returns a dictionary containing the session state needed to restore
        the session in another process. This enables distributed tracing
        across process boundaries.

        Returns:
            Dictionary with name, session_uid_chain (excluding current),
            and metadata

        Example:
            >>> # Process 1
            >>> with ContextVarSession(name="task-123") as session:
            ...     context = session.to_context()
            ...     send_to_remote_process(context)
        """
        return {
            "name": self.name,
            "session_uid_chain": self._session_uid_chain[:-1],  # Exclude current UID
            "metadata": self.metadata,
        }

    @classmethod
    def from_context(
        cls,
        context: dict,
        storage: "SessionStorage | None" = None,
    ) -> "ContextVarSession":
        """
        Restore session from serialized context.

        Creates a new session instance that continues the session hierarchy
        from a parent process. The new session will inherit the parent's
        session UID chain, enabling distributed tracing.

        Args:
            context: Dictionary from to_context() containing session state
            storage: Storage backend to use (required for cross-process tracing)

        Returns:
            New ContextVarSession instance that continues the parent hierarchy

        Example:
            >>> # Process 2
            >>> context = receive_from_process1()
            >>> storage = SqliteSessionStorage("traces.db")
            >>> with ContextVarSession.from_context(context, storage=storage) as session:
            ...     llm.call()  # Traces visible to parent session in Process 1!
        """
        return cls(
            name=context["name"],
            _session_uid_chain=context["session_uid_chain"],
            storage=storage,
            **context.get("metadata", {}),
        )

    def __repr__(self):
        return f"ContextVarSession(name={self.name!r}, _uid={self._uid!r}, chain_depth={len(self._session_uid_chain)}, storage={self.storage!r})"
