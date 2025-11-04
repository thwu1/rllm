"""
Unit tests for session context management.
"""

import pytest
import json

from sdk.proxy.context import (
    SessionContext,
    get_session_context,
    set_session_context,
)


class TestSessionContext:
    """Tests for SessionContext class."""

    def test_from_headers_with_valid_headers(self):
        """Test creating context from valid headers."""
        headers = {
            "x-rllm-session": "episode-123",
            "x-rllm-metadata": json.dumps({"task_id": "task-456"}),
        }

        context = SessionContext.from_headers(headers)

        assert context is not None
        assert context.session_id == "episode-123"
        assert context.metadata == {"task_id": "task-456"}
        assert context.request_id is not None

    def test_from_headers_without_session_id(self):
        """Test creating context without session ID returns None."""
        headers = {"x-rllm-metadata": json.dumps({"task_id": "task-456"})}

        context = SessionContext.from_headers(headers)

        assert context is None

    def test_from_headers_with_invalid_metadata(self):
        """Test creating context with invalid JSON metadata."""
        headers = {
            "x-rllm-session": "episode-123",
            "x-rllm-metadata": "invalid-json",
        }

        context = SessionContext.from_headers(headers)

        assert context is not None
        assert context.session_id == "episode-123"
        assert context.metadata == {}

    def test_to_headers(self):
        """Test converting context to headers."""
        context = SessionContext(
            session_id="episode-123",
            metadata={"task_id": "task-456"},
            request_id="req-789",
        )

        headers = context.to_headers()

        assert headers["X-RLLM-Session"] == "episode-123"
        assert headers["X-RLLM-Request-ID"] == "req-789"
        assert json.loads(headers["X-RLLM-Metadata"]) == {"task_id": "task-456"}

    def test_context_vars(self):
        """Test context variable get/set."""
        context = SessionContext(session_id="episode-123")

        set_session_context(context)
        retrieved = get_session_context()

        assert retrieved == context
        assert retrieved.session_id == "episode-123"

        # Clear context
        set_session_context(None)
        assert get_session_context() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
