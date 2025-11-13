"""ContextVar-based session implementation (default)."""

import contextvars
import uuid
from collections.abc import Callable
from typing import Any

# Session-specific context variables
_current_session: contextvars.ContextVar["ContextVarSession | None"] = contextvars.ContextVar("current_session", default=None)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
# Stack of active sessions (outer → inner). Use None default to avoid shared list instances.
_sessions_stack: contextvars.ContextVar[list["ContextVarSession"] | None] = contextvars.ContextVar("sessions_stack", default=None)


def get_current_session() -> "ContextVarSession | None":
    """Get the current session instance from context."""
    return _current_session.get()


def get_current_session_id() -> str | None:
    """Get current session_id from context."""
    return _session_id.get()


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
    Session implementation using Python contextvars.

    This is the default session implementation that uses contextvars
    for thread-safe and async-safe context propagation.

    Features:
    - Thread-safe and async-safe
    - Automatic nested session isolation
    - In-memory call storage
    - Zero external dependencies

    Example:
        >>> from rllm.sdk.session import ContextVarSession
        >>> from rllm.sdk import get_chat_client
        >>>
        >>> llm = get_chat_client(api_key="...", model="gpt-4")
        >>>
        >>> with ContextVarSession() as session:
        ...     llm.chat.completions.create(...)
        ...     print(len(session.llm_calls))  # Immediate access
    """

    def __init__(self, session_id: str | None = None, formatter: Callable[[dict], dict] | None = None, persistent_tracers: list | None = None, **metadata):
        """
        Initialize contextvars-based session.

        Args:
            session_id: Session ID (auto-generated if None). If None and there's an
                       existing session_id in the context (from a parent session),
                       that will be inherited instead of generating a new one.
            formatter: Optional formatter to transform trace data
            persistent_tracers: Optional list of persistent tracers
            **metadata: Session metadata
        """
        # If session_id is not explicitly provided, check if there's one in the context
        # (set by a parent _session_with_id). This allows internal code to control
        # the session_id while keeping it hidden from users via the session() shortcut.
        if session_id is None:
            existing_session_id = _session_id.get()
            if existing_session_id is not None:
                session_id = existing_session_id

        # Generate new session_id only if none was provided and none exists in context
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        # Internal unique ID for this session context instance (different from session_id which can be inherited)
        # This allows tracking each unique session context even when session_id is shared
        self._uid = f"ctx_{uuid.uuid4().hex[:16]}"
        self.metadata = metadata
        self.formatter = formatter or (lambda x: x)

        # In-memory storage for THIS session's calls
        self._calls: list[dict[str, Any]] = []

        # Optional persistent tracers
        self._persistent_tracers = persistent_tracers or []

        # Context tokens for cleanup
        self._session_token = None
        self._session_id_token = None
        self._metadata_token = None
        self._stack_token = None

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        """Get all LLM calls made within this session."""
        return self._calls.copy()

    def clear_calls(self) -> None:
        """Clear all stored calls for this session."""
        self._calls.clear()

    def __enter__(self):
        """Enter session context - set up context variables."""
        # Set this session instance in context
        self._session_token = _current_session.set(self)

        # Set session_id in context
        self._session_id_token = _session_id.set(self.session_id)

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
        if self._session_id_token is not None:
            _session_id.reset(self._session_id_token)
        if self._metadata_token is not None:
            _metadata.reset(self._metadata_token)

        # Restore previous sessions stack
        if self._stack_token is not None:
            _sessions_stack.reset(self._stack_token)

        return False

    def __len__(self) -> int:
        """Return number of calls in this session."""
        return len(self._calls)

    def __repr__(self):
        return f"ContextVarSession(session_id={self.session_id!r}, calls={len(self._calls)})"
