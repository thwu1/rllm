"""Session context manager for automatic trace tracking."""

import uuid

from .context import _metadata, _session_id


class SessionContext:
    """Context manager for session-based tracing with automatic metadata tracking.

    Examples:
        Simple session:
        >>> with SessionContext("my-session"):
        ...     # All traces get session_id="my-session"

        Session with metadata:
        >>> with SessionContext("my-session", experiment="v1", user="alice"):
        ...     # All traces get session_id + custom metadata

        Auto-generated session_id:
        >>> with SessionContext(experiment="v1"):
        ...     # session_id auto-generated, metadata preserved

        Nested sessions (metadata inheritance):
        >>> with SessionContext("outer", experiment="v1"):
        ...     with SessionContext("inner", task="math"):
        ...         # Inherits experiment="v1", adds task="math"
    """

    def __init__(self, session_id: str | None = None, **metadata):
        """Initialize session context.

        Args:
            session_id: Session ID (auto-generated if None)
            **metadata: Arbitrary metadata to attach to all traces in this session
        """
        self.session_id = session_id or f"session-{uuid.uuid4()}"
        self.metadata = metadata
        self.s_token = None
        self.m_token = None

    def __enter__(self):
        """Enter session context - set session_id and merge metadata."""
        # Get parent metadata (for nested contexts)
        parent_meta = _metadata.get()
        if parent_meta is None:
            parent_meta = {}

        # Merge: parent metadata + current metadata (current wins)
        merged = {**parent_meta, **self.metadata}

        # Set context variables
        self.s_token = _session_id.set(self.session_id)
        self.m_token = _metadata.set(merged)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - restore previous context."""
        # Reset to parent context
        _session_id.reset(self.s_token)
        _metadata.reset(self.m_token)
        return False

    def __repr__(self):
        return f"SessionContext(session_id={self.session_id!r}, metadata={self.metadata!r})"
