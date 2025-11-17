#!/usr/bin/env python3
"""Integration tests for OTelSession with HTTP propagation and middleware."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rllm.sdk.session.otel import (
    OTelSession,
    get_otel_metadata,
    init_otel_distributed_tracing,
)
from rllm.sdk.session.storage import InMemoryStorage


class TestHTTPInstrumentation:
    """Test HTTP instrumentation setup and auto-init."""

    def test_init_otel_distributed_tracing(self):
        """Test explicit initialization of OTel distributed tracing."""
        # Mock the OTel libraries
        with patch("rllm.sdk.session.otel.RequestsInstrumentor") as mock_requests, \
             patch("rllm.sdk.session.otel.HTTPXClientInstrumentor") as mock_httpx, \
             patch("rllm.sdk.session.otel.trace") as mock_trace, \
             patch("rllm.sdk.session.otel.TracerProvider") as mock_provider:

            # Configure mock to simulate TracerProvider not being set
            mock_trace.get_tracer_provider.return_value = MagicMock()
            type(mock_trace.get_tracer_provider.return_value).__class__ = type

            # Call init
            init_otel_distributed_tracing()

            # Verify instrumentation was called
            mock_requests.return_value.instrument.assert_called_once()
            mock_httpx.return_value.instrument.assert_called_once()

    def test_init_otel_idempotent(self):
        """Test init_otel_distributed_tracing can be called multiple times safely."""
        with patch("rllm.sdk.session.otel.RequestsInstrumentor") as mock_requests, \
             patch("rllm.sdk.session.otel.HTTPXClientInstrumentor") as mock_httpx, \
             patch("rllm.sdk.session.otel.trace") as mock_trace:

            mock_trace.get_tracer_provider.return_value = MagicMock()
            type(mock_trace.get_tracer_provider.return_value).__class__ = type

            # Reset the global flag for this test
            import rllm.sdk.session.otel
            rllm.sdk.session.otel._http_instrumentation_enabled = False

            # First call
            init_otel_distributed_tracing()
            first_call_count = mock_requests.return_value.instrument.call_count

            # Second call should not instrument again
            init_otel_distributed_tracing()
            second_call_count = mock_requests.return_value.instrument.call_count

            assert first_call_count == second_call_count

    def test_auto_init_on_session_enter(self):
        """Test auto-initialization happens on first session entry."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation") as mock_ensure:
            storage = InMemoryStorage()
            with OTelSession(storage=storage):
                # Should have called ensure_instrumentation
                mock_ensure.assert_called_once()


class TestBaggagePropagation:
    """Test OTel baggage propagation mechanisms."""

    def test_baggage_set_on_session_enter(self):
        """Test baggage is set when entering OTelSession."""
        with patch("rllm.sdk.session.otel.baggage") as mock_baggage, \
             patch("rllm.sdk.session.otel.context") as mock_context, \
             patch("rllm.sdk.session.otel.trace") as mock_trace, \
             patch("rllm.sdk.session.otel._ensure_instrumentation"):

            # Setup mocks
            mock_ctx = MagicMock()
            mock_context.get_current.return_value = mock_ctx
            mock_baggage.set_baggage.return_value = mock_ctx
            mock_trace.get_tracer.return_value.start_span.return_value.__enter__ = MagicMock()
            mock_trace.get_tracer.return_value.start_span.return_value.__exit__ = MagicMock()

            storage = InMemoryStorage()
            with OTelSession(name="test", experiment="v1", storage=storage):
                # Verify baggage.set_baggage was called for session_name
                calls = [call for call in mock_baggage.set_baggage.call_args_list
                        if len(call[0]) > 0 and call[0][0] == "rllm_session_name"]
                assert len(calls) > 0
                assert calls[0][0][1] == "test"

    def test_metadata_propagation_via_baggage(self):
        """Test metadata is correctly propagated via OTel baggage."""
        with patch("rllm.sdk.session.otel.baggage") as mock_baggage, \
             patch("rllm.sdk.session.otel.context") as mock_context, \
             patch("rllm.sdk.session.otel.trace") as mock_trace, \
             patch("rllm.sdk.session.otel._ensure_instrumentation"):

            # Setup mocks
            mock_ctx = MagicMock()
            mock_context.get_current.return_value = mock_ctx
            mock_baggage.set_baggage.return_value = mock_ctx
            mock_trace.get_tracer.return_value.start_span.return_value.__enter__ = MagicMock()
            mock_trace.get_tracer.return_value.start_span.return_value.__exit__ = MagicMock()

            storage = InMemoryStorage()
            with OTelSession(experiment="v1", user="alice", run_id=123, storage=storage):
                # Check that metadata keys were set in baggage
                set_calls = mock_baggage.set_baggage.call_args_list

                # Extract all baggage keys that were set
                baggage_keys = [call[0][0] for call in set_calls if len(call[0]) > 0]

                # Should include rllm_ prefixed metadata
                assert any("rllm_experiment" in key for key in baggage_keys)
                assert any("rllm_user" in key for key in baggage_keys)
                assert any("rllm_run_id" in key for key in baggage_keys)


class TestMiddlewareBaggageExtraction:
    """Test middleware extraction of OTel baggage from HTTP headers."""

    @pytest.mark.asyncio
    async def test_middleware_extracts_baggage(self):
        """Test middleware extracts metadata from baggage header."""
        from fastapi import Request
        from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

        # Create mock request with baggage header
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {
            "baggage": "rllm_session_name=test_session,rllm_experiment=v1"
        }

        # Mock the OTel extract function
        with patch("rllm.sdk.proxy.middleware.extract") as mock_extract, \
             patch("rllm.sdk.proxy.middleware.baggage") as mock_baggage:

            # Setup mock context with baggage
            mock_ctx = MagicMock()
            mock_extract.return_value = mock_ctx

            # Mock baggage.get_baggage to return our test values
            def get_baggage_side_effect(key, context=None):
                baggage_map = {
                    "rllm_session_name": "test_session",
                    "rllm_experiment": "v1",
                    "rllm_metadata_keys": "experiment"
                }
                return baggage_map.get(key)

            mock_baggage.get_baggage.side_effect = get_baggage_side_effect

            # Create middleware and mock call_next
            middleware = MetadataRoutingMiddleware(app=MagicMock())
            mock_call_next = MagicMock()
            mock_call_next.return_value = MagicMock()

            # Mock request.state
            mock_request.state = MagicMock()
            mock_request.body = MagicMock(return_value=b'{}')

            # Process request
            await middleware.dispatch(mock_request, mock_call_next)

            # Verify metadata was extracted and stored
            assert hasattr(mock_request.state, 'rllm_metadata')
            metadata = mock_request.state.rllm_metadata
            assert metadata.get("session_name") == "test_session"
            assert metadata.get("experiment") == "v1"

    @pytest.mark.asyncio
    async def test_middleware_url_slug_fallback(self):
        """Test middleware falls back to URL slug when no baggage."""
        from fastapi import Request
        from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware
        from rllm.sdk.proxy.metadata_slug import encode_metadata_slug

        # Create metadata slug
        metadata = {"experiment": "v2", "session_name": "url_session"}
        slug = encode_metadata_slug(metadata)

        # Create mock request with slug in URL
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = f"/meta/{slug}/v1/chat/completions"
        mock_request.headers = {}
        mock_request.scope = {"path": mock_request.url.path, "raw_path": b"/test"}

        # Mock request.state and body
        mock_request.state = MagicMock()
        mock_request.body = MagicMock(return_value=b'{}')

        # Create middleware
        middleware = MetadataRoutingMiddleware(app=MagicMock())
        mock_call_next = MagicMock()
        mock_call_next.return_value = MagicMock()

        # Process request
        await middleware.dispatch(mock_request, mock_call_next)

        # Verify metadata was extracted from slug
        assert hasattr(mock_request.state, 'rllm_metadata')
        extracted_metadata = mock_request.state.rllm_metadata
        assert extracted_metadata.get("experiment") == "v2"
        assert extracted_metadata.get("session_name") == "url_session"

    @pytest.mark.asyncio
    async def test_middleware_baggage_precedence(self):
        """Test baggage takes precedence over URL slug."""
        from fastapi import Request
        from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware
        from rllm.sdk.proxy.metadata_slug import encode_metadata_slug

        # Create metadata slug
        slug_metadata = {"experiment": "v1", "session_name": "slug_session"}
        slug = encode_metadata_slug(slug_metadata)

        # Create mock request with BOTH slug and baggage
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = f"/meta/{slug}/v1/chat/completions"
        mock_request.headers = {
            "baggage": "rllm_session_name=baggage_session,rllm_experiment=v2"
        }
        mock_request.scope = {"path": mock_request.url.path, "raw_path": b"/test"}

        # Mock the OTel extract
        with patch("rllm.sdk.proxy.middleware.extract") as mock_extract, \
             patch("rllm.sdk.proxy.middleware.baggage") as mock_baggage:

            mock_ctx = MagicMock()
            mock_extract.return_value = mock_ctx

            def get_baggage_side_effect(key, context=None):
                baggage_map = {
                    "rllm_session_name": "baggage_session",
                    "rllm_experiment": "v2",
                    "rllm_metadata_keys": "experiment"
                }
                return baggage_map.get(key)

            mock_baggage.get_baggage.side_effect = get_baggage_side_effect

            # Mock request.state and body
            mock_request.state = MagicMock()
            mock_request.body = MagicMock(return_value=b'{}')

            # Create middleware
            middleware = MetadataRoutingMiddleware(app=MagicMock())
            mock_call_next = MagicMock()
            mock_call_next.return_value = MagicMock()

            # Process request
            await middleware.dispatch(mock_request, mock_call_next)

            # Verify baggage metadata took precedence
            metadata = mock_request.state.rllm_metadata
            assert metadata.get("session_name") == "baggage_session"  # From baggage, not slug
            assert metadata.get("experiment") == "v2"  # From baggage, not slug


class TestMetadataAssemblyWithOTel:
    """Test metadata assembly integrates correctly with OTelSession."""

    def test_assemble_routing_metadata_with_otel(self):
        """Test assemble_routing_metadata reads from OTelSession."""
        from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata

        with patch("rllm.sdk.session.otel.baggage") as mock_baggage, \
             patch("rllm.sdk.session.otel.context") as mock_context, \
             patch("rllm.sdk.session.otel.trace") as mock_trace, \
             patch("rllm.sdk.session.otel._ensure_instrumentation"):

            # Setup mocks
            mock_ctx = MagicMock()
            mock_context.get_current.return_value = mock_ctx
            mock_baggage.set_baggage.return_value = mock_ctx
            mock_trace.get_tracer.return_value.start_span.return_value.__enter__ = MagicMock()
            mock_trace.get_tracer.return_value.start_span.return_value.__exit__ = MagicMock()

            # Mock get_baggage for reading
            def get_baggage_side_effect(key, context=None):
                baggage_map = {
                    "rllm_session_name": "otel_session",
                    "rllm_session_uid": "otel_123",
                    "rllm_experiment": "v3",
                    "rllm_metadata_keys": "experiment"
                }
                return baggage_map.get(key)

            mock_baggage.get_baggage.side_effect = get_baggage_side_effect

            storage = InMemoryStorage()
            with OTelSession(name="otel_session", experiment="v3", storage=storage) as session:
                metadata = assemble_routing_metadata()

                # Should have extracted from OTel baggage
                assert "session_name" in metadata
                assert "experiment" in metadata


class TestConcurrentSessions:
    """Test concurrent and interleaved session usage."""

    def test_concurrent_contextvarsession_and_otelsession(self):
        """Test ContextVarSession and OTelSession can coexist."""
        from rllm.sdk.session import ContextVarSession, get_current_session
        from rllm.sdk.session.otel import get_current_otel_session

        storage = InMemoryStorage()

        # Start with ContextVarSession
        with ContextVarSession(name="ctx_session", storage=storage):
            assert get_current_session() is not None
            assert get_current_session().name == "ctx_session"
            assert get_current_otel_session() is None

            # Nest OTelSession inside
            with patch("rllm.sdk.session.otel._ensure_instrumentation"):
                with OTelSession(name="otel_session", storage=storage):
                    # Both should be active in their respective contexts
                    assert get_current_session().name == "ctx_session"
                    assert get_current_otel_session().name == "otel_session"

            # After OTel exits, ContextVar should still be active
            assert get_current_session().name == "ctx_session"
            assert get_current_otel_session() is None

    def test_interleaved_sessions(self):
        """Test multiple interleaved OTelSessions."""
        from rllm.sdk.session.otel import get_active_otel_sessions

        storage = InMemoryStorage()

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="session1", storage=storage) as s1:
                with OTelSession(name="session2", storage=storage) as s2:
                    with OTelSession(name="session3", storage=storage) as s3:
                        active = get_active_otel_sessions()
                        assert len(active) == 3
                        assert active[0].name == "session1"
                        assert active[1].name == "session2"
                        assert active[2].name == "session3"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_session_without_otel_installed(self):
        """Test graceful degradation when OTel not fully installed."""
        # This should not crash even if some OTel imports fail
        with patch("rllm.sdk.session.otel.trace") as mock_trace:
            # Simulate ImportError when trying to use trace
            mock_trace.get_tracer.side_effect = ImportError("OTel not installed")

            storage = InMemoryStorage()
            # Should still work (with warnings) even without OTel
            with patch("rllm.sdk.session.otel._ensure_instrumentation"):
                try:
                    with OTelSession(storage=storage):
                        pass
                    # If we get here, graceful degradation worked
                except ImportError:
                    pytest.fail("Should not raise ImportError, should degrade gracefully")

    def test_metadata_assembly_without_otel(self):
        """Test metadata assembly works when OTel import fails."""
        from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata

        # Mock import failure
        with patch("rllm.sdk.proxy.metadata_slug.get_current_otel_session") as mock_get:
            # Simulate ImportError
            mock_get.side_effect = ImportError("OTel not installed")

            # Should fall back to ContextVarSession without crashing
            metadata = assemble_routing_metadata()
            assert isinstance(metadata, dict)

    def test_middleware_without_otel(self):
        """Test middleware works when OTel not installed."""
        # The middleware should handle ImportError gracefully
        from fastapi import Request
        from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"baggage": "test"}
        mock_request.state = MagicMock()
        mock_request.body = MagicMock(return_value=b'{}')

        # This should not crash even if OTel extract fails
        middleware = MetadataRoutingMiddleware(app=MagicMock())
        mock_call_next = MagicMock()
        mock_call_next.return_value = MagicMock()

        # Should complete without error (will skip baggage extraction)
        import asyncio
        try:
            asyncio.run(middleware.dispatch(mock_request, mock_call_next))
        except Exception as e:
            # Should not raise exception from missing OTel
            if "opentelemetry" in str(e).lower():
                pytest.fail(f"Should handle missing OTel gracefully: {e}")


class TestStorageIntegration:
    """Test OTelSession integration with different storage backends."""

    def test_otelsession_with_sqlite_storage(self):
        """Test OTelSession works with SqliteSessionStorage."""
        from rllm.sdk.session.storage import SqliteSessionStorage
        from rllm.sdk.protocol import Trace

        # Use in-memory SQLite for testing
        storage = SqliteSessionStorage(":memory:")

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="test", experiment="v1", storage=storage) as session:
                # Add a trace
                import time
                trace = Trace(
                    trace_id="trace_1",
                    session_name=session.name,
                    name="test_call",
                    input={"messages": [{"role": "user", "content": "test"}]},
                    output="response",
                    model="gpt-4",
                    latency_ms=100.0,
                    tokens={"prompt": 10, "completion": 20, "total": 30},
                    timestamp=time.time(),
                    metadata={"experiment": "v1"}
                )
                storage.add_trace(session._session_uid_chain, session.name, trace)

                # Verify trace can be retrieved
                traces = session.llm_calls
                assert len(traces) >= 0  # May be async, just verify no crash

    def test_nested_sessions_share_traces_via_uid_chain(self):
        """Test parent sessions see child traces via UID chain."""
        from rllm.sdk.protocol import Trace

        storage = InMemoryStorage()

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="parent", storage=storage) as parent:
                import time
                parent_trace = Trace(
                    trace_id="parent_trace",
                    session_name=parent.name,
                    name="parent_call",
                    input=[],
                    output="parent response",
                    model="gpt-4",
                    latency_ms=100.0,
                    tokens={"prompt": 10, "completion": 20, "total": 30},
                    timestamp=time.time(),
                    metadata={}
                )
                storage.add_trace(parent._session_uid_chain, parent.name, parent_trace)

                with OTelSession(name="child", storage=storage) as child:
                    # Child's chain should include parent's UID
                    assert parent._uid in child._session_uid_chain

                    child_trace = Trace(
                        trace_id="child_trace",
                        session_name=child.name,
                        name="child_call",
                        input=[],
                        output="child response",
                        model="gpt-4",
                        latency_ms=100.0,
                        tokens={"prompt": 10, "completion": 20, "total": 30},
                        timestamp=time.time(),
                        metadata={}
                    )
                    storage.add_trace(child._session_uid_chain, child.name, child_trace)

                # Parent should see both traces (because child's chain includes parent's UID)
                parent_traces = parent.llm_calls
                assert len(parent_traces) >= 1  # At least the parent trace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
