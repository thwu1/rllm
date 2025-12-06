"""Tests for auto-instrumentation module."""

import pytest
from unittest.mock import MagicMock, patch

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
    _should_inject,
    _inject_metadata,
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
        assert not is_instrumented()
        instrument()
        assert is_instrumented()
        assert is_instrumented("openai")
        uninstrument()
        assert not is_instrumented()

    def test_double_instrument_is_idempotent(self):
        instrument()
        instrument()  # Should not raise
        assert is_instrumented()

    def test_double_uninstrument_is_idempotent(self):
        instrument()
        uninstrument()
        uninstrument()  # Should not raise
        assert not is_instrumented()

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            instrument(providers=["unknown_provider"])


class TestOpenAIProvider:
    """Test OpenAI-specific instrumentation."""

    def test_openai_instrument_uninstrument(self):
        assert not is_openai_instrumented()
        result = instrument_openai()
        assert result is True
        assert is_openai_instrumented()
        uninstrument_openai()
        assert not is_openai_instrumented()

    def test_openai_methods_are_patched(self):
        from openai.resources.chat.completions import Completions

        original_create = Completions.create
        instrument_openai()
        assert Completions.create is not original_create
        uninstrument_openai()
        assert Completions.create is original_create


class TestProxyUrlRegistration:
    """Test proxy URL registration and matching."""

    def test_register_proxy_url(self):
        clear_proxy_urls()
        assert len(_proxy_urls) == 0

        register_proxy_url("http://proxy:4000")
        assert "http://proxy:4000" in _proxy_urls

        register_proxy_url("http://proxy:4000/v1")  # Should normalize
        assert len(_proxy_urls) == 1  # Same host, shouldn't add duplicate

    def test_should_inject(self):
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        assert _should_inject("http://proxy:4000/v1/chat/completions")
        assert not _should_inject("http://other:8000/v1/chat/completions")

    def test_inject_metadata(self):
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        with ContextVarSession(agent="test"):
            url = "http://proxy:4000/v1/chat/completions"
            new_url = _inject_metadata(url)
            assert "/meta/" in new_url
            assert "/v1/chat/completions" in new_url

    def test_no_injection_outside_session(self):
        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")

        url = "http://proxy:4000/v1/chat/completions"
        new_url = _inject_metadata(url)
        assert new_url == url  # Unchanged outside session


class TestTraceCapture:
    """Test trace capture within sessions."""

    def test_trace_capture_in_session(self):
        instrument()

        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })

        with patch("openai.resources.chat.completions.Completions.create"):
            with ContextVarSession(agent="test") as sess:
                assert len(sess.llm_calls) == 0

    def test_no_trace_outside_session(self):
        instrument()
        from rllm.sdk.instrumentation.openai_provider import _log_trace

        # Call _log_trace outside session - should not raise
        _log_trace(
            name="test",
            input_data=[{"role": "user", "content": "test"}],
            response=MagicMock(model_dump=lambda: {}, usage=None),
            model="gpt-4",
            latency_ms=100,
        )


class TestNestedSessions:
    """Test instrumentation with nested sessions."""

    def test_nested_session_traces(self):
        instrument()

        with ContextVarSession(agent="outer") as outer:
            with ContextVarSession(task="inner") as inner:
                assert outer is not inner

                from rllm.sdk.session.contextvar import get_current_cv_metadata
                metadata = get_current_cv_metadata()
                assert metadata.get("agent") == "outer"
                assert metadata.get("task") == "inner"


class TestHttpxTransport:
    """Test httpx transport patching."""

    def test_httpx_transport_wrapping(self):
        from rllm.sdk.instrumentation.httpx_transport import (
            patch_httpx,
            unpatch_httpx,
            AsyncTransportWrapper,
        )

        clear_proxy_urls()
        register_proxy_url("http://proxy:4000")
        patch_httpx()

        try:
            import httpx
            client = httpx.AsyncClient()
            assert isinstance(client._transport, AsyncTransportWrapper)
        finally:
            unpatch_httpx()


class TestIntegration:
    """Integration tests for the full instrumentation flow."""

    def test_full_instrumentation_with_proxy(self):
        instrument(proxy_urls=["http://proxy:4000"])
        assert is_instrumented()
        assert "http://proxy:4000" in _proxy_urls
        uninstrument()
        assert not is_instrumented()

    def test_instrumentation_session_metadata_flow(self):
        instrument()

        with ContextVarSession(experiment="v1", user_id="123") as sess:
            from rllm.sdk.session.contextvar import get_current_cv_metadata
            metadata = get_current_cv_metadata()
            assert metadata["experiment"] == "v1"
            assert metadata["user_id"] == "123"
            assert sess.name is not None
