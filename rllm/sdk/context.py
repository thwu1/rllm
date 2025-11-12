"""Context variables for automatic session and metadata propagation."""

import contextvars
from typing import Any

# Context variables (thread-safe, async-safe)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
# Avoid mutable default; callers should treat missing metadata as empty dict.
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)


def get_current_session() -> str | None:
    """Get current session_id from context.

    Returns:
        Current session_id or None if not in a session context.
    """
    return _session_id.get()


def get_current_metadata() -> dict[str, Any]:
    """Get current metadata from context.

    Returns:
        Current metadata dict (empty dict if not in a session context).
    """
    metadata = _metadata.get()
    return metadata if metadata is not None else {}
