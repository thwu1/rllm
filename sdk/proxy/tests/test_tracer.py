"""
Unit tests for LLM call tracing.
"""

import pytest
import tempfile
import json
from pathlib import Path

from sdk.proxy.tracer import LLMTracer, LLMCallTrace, get_tracer, set_tracer
from sdk.proxy.context import SessionContext


class TestLLMCallTrace:
    """Tests for LLMCallTrace dataclass."""

    def test_trace_creation(self):
        """Test creating a trace."""
        trace = LLMCallTrace(
            request_id="req-123",
            session_id="episode-456",
            timestamp="2024-01-01T12:00:00",
            model="gpt-3.5-turbo",
        )

        assert trace.request_id == "req-123"
        assert trace.session_id == "episode-456"
        assert trace.model == "gpt-3.5-turbo"
        assert trace.metadata == {}

    def test_trace_to_dict(self):
        """Test converting trace to dict."""
        trace = LLMCallTrace(
            request_id="req-123",
            session_id="episode-456",
            timestamp="2024-01-01T12:00:00",
            model="gpt-3.5-turbo",
            response_text="Hello!",
        )

        trace_dict = trace.to_dict()

        assert trace_dict["request_id"] == "req-123"
        assert trace_dict["response_text"] == "Hello!"
        assert "metadata" in trace_dict


class TestLLMTracer:
    """Tests for LLMTracer class."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tracer(self, temp_log_dir):
        """Create tracer with temp directory."""
        return LLMTracer(log_dir=temp_log_dir)

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="episode-123",
            metadata={"task_id": "task-456"},
            request_id="req-789",
        )

    def test_start_trace(self, tracer, session_context):
        """Test starting a trace."""
        trace = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            provider="openai",
        )

        assert trace.request_id == "req-789"
        assert trace.session_id == "episode-123"
        assert trace.model == "gpt-3.5-turbo"
        assert trace.provider == "openai"
        assert trace.messages == [{"role": "user", "content": "Hello"}]

    def test_complete_trace(self, tracer, session_context):
        """Test completing a trace."""
        trace = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            provider="openai",
        )

        tracer.complete_trace(
            trace,
            response_text="Hi there!",
            prompt_length=10,
            completion_length=5,
            latency_ms=123.45,
            finish_reason="stop",
        )

        assert trace.response_text == "Hi there!"
        assert trace.prompt_length == 10
        assert trace.completion_length == 5
        assert trace.latency_ms == 123.45
        assert trace.finish_reason == "stop"

    def test_trace_persistence(self, tracer, session_context, temp_log_dir):
        """Test that traces are written to disk."""
        trace = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            provider="openai",
        )

        tracer.complete_trace(
            trace,
            response_text="Test response",
            latency_ms=100.0,
        )

        # Check that trace file was created
        trace_file = temp_log_dir / "episode-123" / f"{trace.request_id}.json"
        assert trace_file.exists()

        # Verify contents
        with open(trace_file) as f:
            saved_trace = json.load(f)

        assert saved_trace["request_id"] == trace.request_id
        assert saved_trace["response_text"] == "Test response"

    def test_get_traces(self, tracer, session_context):
        """Test retrieving traces for a session."""
        # Create multiple traces
        trace1 = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            provider="openai",
        )
        tracer.complete_trace(trace1, response_text="Response 1")

        context2 = SessionContext(
            session_id="episode-123",
            request_id="req-999",
        )
        trace2 = tracer.start_trace(
            context=context2,
            model="gpt-4",
            provider="openai",
        )
        tracer.complete_trace(trace2, response_text="Response 2")

        # Get all traces for session
        traces = tracer.get_traces("episode-123")

        assert len(traces) == 2
        assert traces[0].response_text == "Response 1"
        assert traces[1].response_text == "Response 2"

    def test_clear_session(self, tracer, session_context):
        """Test clearing session traces."""
        trace = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            provider="openai",
        )
        tracer.complete_trace(trace, response_text="Test")

        # Verify traces exist
        assert len(tracer.get_traces("episode-123")) == 1

        # Clear session
        tracer.clear_session("episode-123")

        # Verify traces cleared
        assert len(tracer.get_traces("episode-123")) == 0

    def test_trace_with_error(self, tracer, session_context):
        """Test tracing with error."""
        trace = tracer.start_trace(
            context=session_context,
            model="gpt-3.5-turbo",
            provider="openai",
        )

        tracer.complete_trace(
            trace,
            error="Connection timeout",
            latency_ms=5000.0,
        )

        assert trace.error == "Connection timeout"
        assert trace.response_text is None


class TestTracerGlobal:
    """Tests for global tracer functions."""

    def test_get_global_tracer(self):
        """Test getting global tracer instance."""
        tracer = get_tracer()
        assert tracer is not None
        assert isinstance(tracer, LLMTracer)

        # Should return same instance
        tracer2 = get_tracer()
        assert tracer is tracer2

    def test_set_global_tracer(self):
        """Test setting global tracer."""
        custom_tracer = LLMTracer(log_dir=Path("/tmp/custom"))
        set_tracer(custom_tracer)

        retrieved = get_tracer()
        assert retrieved is custom_tracer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
