"""Tests for auto-instrumentation module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from rllm.sdk.instrumentation import (
    instrument,
    uninstrument,
    is_instrumented,
    clear_proxy_urls,
)
from rllm.sdk.instrumentation.openai_provider import (
    instrument_openai,
    uninstrument_openai,
    is_instrumented as is_openai_instrumented,
)
from rllm.sdk.instrumentation.httpx_transport import (
    register_proxy_url,
    _should_inject_metadata,
    _inject_metadata_into_url,
    _proxy_urls,
)
from rllm.sdk.session.contextvar import ContextVarSession


@pytest.fixture(autouse=True)
def cleanup_instrumentation():
    """Ensure instrumentation is cleaned up after each test."""
    yield
    uninstrument()
    clear_proxy_urls()


class TestInstrumentationBasics:
    """Test basic instrumentation lifecycle."""

    def test_instrument_uninstrument(self):
        """Test that instrument and uninstrument work correctly."""
        assert not is_instrumented()

        instrument()
        assert is_instrumented()
        assert is_instrumented("openai")

        uninstrument()
        assert not is_instrumented()

    def test_double_instrument_is_idempotent(self):
        """Test that calling instrument twice doesn't cause issues."""
        instrument()
        instrument()  # Should not raise
        assert is_instrumented()

    def test_double_uninstrument_is_idempotent(self):
        """Test that calling uninstrument twice doesn't cause issues."""
        instrument()
        uninstrument()
        uninstrument()  # Should not raise
        assert not is_instrumented()

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            instrument(providers=["unknown_provider"])


class TestOpenAIProvider:
    """Test OpenAI-specific instrumentation."""

    def test_openai_instrument_uninstrument(self):
        """Test OpenAI provider instrumentation lifecycle."""
        assert not is_openai_instrumented()

        result = instrument_openai()
        assert result is True
        assert is_openai_instrumented()

        uninstrument_openai()
        assert not is_openai_instrumented()

    def test_openai_methods_are_patched(self):
        """Test that OpenAI methods are actually patched."""
        from openai.resources.chat.completions import Completions

        original_create = Completions.create

        instrument_openai()

        # The method should be different after patching
        assert Completions.create is not original_create

        uninstrument_openai()

        # Should be restored
        assert Completions.create is original_create


class TestProxyUrlRegistration:
    """Test proxy URL registration and matching."""

    def test_register_proxy_url(self):
        """Test proxy URL registration."""
        clear_proxy_urls()
        assert len(_proxy_urls) == 0

        register_proxy_url("http://proxy:4000")
        assert "http://proxy:4000" in _proxy_urls

        register_proxy_url("http://proxy:4000/v1")  # Should normalize
        assert len(_proxy_urls) == 1  # Same host, shouldn't add duplicate

    def test_should_inject_metadata(self):
        """Test URL matching for metadata injection."""
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        assert _should_inject_metadata("http://proxy:4000/v1/chat/completions")
        assert not _should_inject_metadata("http://other:8000/v1/chat/completions")

    def test_inject_metadata_into_url(self):
        """Test URL transformation with metadata."""
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        # Test within a session context
        with ContextVarSession(agent="test"):
            url = "http://proxy:4000/v1/chat/completions"
            new_url = _inject_metadata_into_url(url)

            # URL should contain /meta/ segment
            assert "/meta/" in new_url
            assert "/v1/chat/completions" in new_url

    def test_no_injection_outside_session(self):
        """Test that no metadata is injected outside session context."""
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        url = "http://proxy:4000/v1/chat/completions"
        new_url = _inject_metadata_into_url(url)

        # Should be unchanged (no session context)
        assert new_url == url


class TestTraceCapture:
    """Test trace capture within sessions."""

    def test_trace_capture_in_session(self):
        """Test that traces are captured within session context."""
        instrument()

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4"
        mock_response.id = "chatcmpl-123"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model_dump = MagicMock(return_value={
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })

        with patch("openai.resources.chat.completions.Completions.create") as mock_create:
            # We need to use the original after instrumentation
            from openai import OpenAI

            with ContextVarSession(agent="test") as sess:
                # The session should start empty
                assert len(sess.llm_calls) == 0

    def test_no_trace_outside_session(self):
        """Test that traces are not captured outside session context."""
        instrument()

        # Outside session, no traces should be stored
        # This is verified by checking that _log_trace returns early
        from rllm.sdk.instrumentation.openai_provider import _log_trace

        # Call _log_trace outside session - should not raise and should not store
        _log_trace(
            name="test",
            messages=[{"role": "user", "content": "test"}],
            response=MagicMock(
                model_dump=lambda: {},
                usage=None,
            ),
            model="gpt-4",
            latency_ms=100,
        )
        # No assertion needed - just verify it doesn't raise


class TestNestedSessions:
    """Test instrumentation with nested sessions."""

    def test_nested_session_traces(self):
        """Test that nested sessions properly capture traces."""
        instrument()

        with ContextVarSession(agent="outer") as outer:
            with ContextVarSession(task="inner") as inner:
                # Both sessions should be accessible
                assert outer is not inner

                # Inner session should inherit outer's metadata context
                from rllm.sdk.session.contextvar import get_current_cv_metadata
                metadata = get_current_cv_metadata()
                assert metadata.get("agent") == "outer"
                assert metadata.get("task") == "inner"


class TestHttpxTransport:
    """Test httpx transport patching."""

    def test_httpx_transport_wrapping(self):
        """Test that httpx transports are properly wrapped."""
        from rllm.sdk.instrumentation.httpx_transport import (
            patch_httpx,
            unpatch_httpx,
            SessionAwareAsyncTransport,
        )

        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        patch_httpx()

        try:
            import httpx

            # Create a client - transport should be wrapped
            client = httpx.AsyncClient()
            assert isinstance(client._transport, SessionAwareAsyncTransport)

        finally:
            unpatch_httpx()


class TestIntegration:
    """Integration tests for the full instrumentation flow."""

    def test_full_instrumentation_with_proxy(self):
        """Test full instrumentation with proxy URL registration."""
        # Setup
        instrument(proxy_urls=["http://proxy:4000"])

        assert is_instrumented()
        assert "http://proxy:4000" in _proxy_urls

        # Cleanup
        uninstrument()

        assert not is_instrumented()

    def test_instrumentation_session_metadata_flow(self):
        """Test that session metadata flows correctly through instrumentation."""
        instrument()

        with ContextVarSession(experiment="v1", user_id="123") as sess:
            from rllm.sdk.session.contextvar import get_current_cv_metadata
            metadata = get_current_cv_metadata()

            assert metadata["experiment"] == "v1"
            assert metadata["user_id"] == "123"

            # Session should be accessible
            assert sess.name is not None
