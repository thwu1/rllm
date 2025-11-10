"""Session context manager for automatic trace tracking."""

from rllm.sdk.context import _metadata, _session_id


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
            session_id: Session ID (if None, context variables won't be set)
            **metadata: Arbitrary metadata to attach to all traces in this session
        """
        self.session_id = session_id
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

        # Set session_id context only if provided
        if self.session_id is not None:
            self.s_token = _session_id.set(self.session_id)

        # Always set metadata context (even if empty, to maintain nesting)
        self.m_token = _metadata.set(merged)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - restore previous context."""
        # Only reset if we actually set the context variables
        if self.s_token is not None:
            _session_id.reset(self.s_token)
        if self.m_token is not None:
            _metadata.reset(self.m_token)
        return False

    def __repr__(self):
        return f"SessionContext(session_id={self.session_id!r}, metadata={self.metadata!r})"
