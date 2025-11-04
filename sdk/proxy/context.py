"""
Session context management for RLLM proxy.

Handles context propagation through HTTP headers for episodic tracing.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional
import json
import uuid


# Context variables for request-level state
_session_context: ContextVar[Optional["SessionContext"]] = ContextVar(
    "_session_context", default=None
)


@dataclass
class SessionContext:
    """
    Context for tracking LLM calls within an episode/trajectory.

    Attributes:
        session_id: Unique identifier for the episode or trajectory
        metadata: Additional context (task_id, agent_name, step_idx, etc.)
        request_id: Unique ID for this specific LLM request
    """
    session_id: str
    metadata: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_headers(cls, headers: dict) -> Optional["SessionContext"]:
        """
        Extract session context from HTTP headers.

        Expected headers:
            X-RLLM-Session: Session/episode identifier
            X-RLLM-Metadata: JSON-encoded metadata

        Args:
            headers: HTTP headers dictionary

        Returns:
            SessionContext if headers present, None otherwise
        """
        session_id = headers.get("x-rllm-session")
        if not session_id:
            return None

        metadata_str = headers.get("x-rllm-metadata", "{}")
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}

        return cls(session_id=session_id, metadata=metadata)

    def to_headers(self) -> dict:
        """
        Convert context to HTTP headers for propagation.

        Returns:
            Dictionary of headers
        """
        return {
            "X-RLLM-Session": self.session_id,
            "X-RLLM-Metadata": json.dumps(self.metadata),
            "X-RLLM-Request-ID": self.request_id,
        }


def get_session_context() -> Optional[SessionContext]:
    """Get the current session context."""
    return _session_context.get()


def set_session_context(context: Optional[SessionContext]) -> None:
    """Set the current session context."""
    _session_context.set(context)
