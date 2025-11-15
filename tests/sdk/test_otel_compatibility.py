#!/usr/bin/env python3
"""Compatibility and real-world scenario tests for OTelSession."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from rllm.sdk.protocol import Trace
from rllm.sdk.session import ContextVarSession
from rllm.sdk.session.otel import OTelSession, init_otel_distributed_tracing
from rllm.sdk.session.storage import InMemoryStorage, SqliteSessionStorage


class TestContextVarSessionCompatibility:
    """Test OTelSession maintains compatibility with ContextVarSession patterns."""

    def test_same_api_surface(self):
        """Test OTelSession has same API as ContextVarSession."""
        storage = InMemoryStorage()

        ctx_session = ContextVarSession(name="ctx", storage=storage)
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            otel_session = OTelSession(name="otel", storage=storage)

        # Both should have same attributes
        assert hasattr(ctx_session, 'name')
        assert hasattr(otel_session, 'name')
        assert hasattr(ctx_session, 'metadata')
        assert hasattr(otel_session, 'metadata')
        assert hasattr(ctx_session, 'llm_calls')
        assert hasattr(otel_session, 'llm_calls')
        assert hasattr(ctx_session, 'clear_calls')
        assert hasattr(otel_session, 'clear_calls')
        assert hasattr(ctx_session, '__enter__')
        assert hasattr(otel_session, '__enter__')
        assert hasattr(ctx_session, '__exit__')
        assert hasattr(otel_session, '__exit__')

    def test_same_context_manager_behavior(self):
        """Test both sessions work as context managers identically."""
        storage = InMemoryStorage()

        # ContextVarSession
        with ContextVarSession(name="ctx", experiment="v1", storage=storage) as ctx_session:
            assert ctx_session.name == "ctx"
            assert ctx_session.metadata["experiment"] == "v1"

        # OTelSession
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="otel", experiment="v1", storage=storage) as otel_session:
                assert otel_session.name == "otel"
                assert otel_session.metadata["experiment"] == "v1"

    def test_both_work_with_same_storage(self):
        """Test ContextVarSession and OTelSession can use same storage."""
        storage = InMemoryStorage()

        # Use ContextVarSession first
        with ContextVarSession(name="ctx", storage=storage) as ctx:
            trace1 = Trace(
                uid="ctx_trace",
                session_uid_chain=ctx._session_uid_chain,
                session_name=ctx.name,
                messages=[],
                model="gpt-4",
                response="response1",
                metadata={}
            )
            storage.add_trace(ctx._session_uid_chain, ctx.name, trace1)

        # Use OTelSession with same storage
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="otel", storage=storage) as otel:
                trace2 = Trace(
                    uid="otel_trace",
                    session_uid_chain=otel._session_uid_chain,
                    session_name=otel.name,
                    messages=[],
                    model="gpt-4",
                    response="response2",
                    metadata={}
                )
                storage.add_trace(otel._session_uid_chain, otel.name, trace2)

        # Both types of sessions can coexist with same storage
        assert True  # No errors means success


class TestRealWorldHTTPScenarios:
    """Test real-world HTTP microservice scenarios."""

    def test_http_request_with_baggage_header(self):
        """Test simulated HTTP request with OTel baggage."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Service A: Create session and make "HTTP request"
            with OTelSession(name="service_a", experiment="v1", user="alice", storage=storage):
                from rllm.sdk.session.otel import get_otel_metadata

                # Simulate extracting baggage for HTTP header
                metadata = get_otel_metadata()
                assert metadata.get("experiment") == "v1"
                assert metadata.get("user") == "alice"

                # In real scenario, this would be injected into HTTP header by OTel instrumentation
                # Here we just verify the metadata is available

    def test_microservice_chain_simulation(self):
        """Simulate request flowing through multiple microservices."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"), \
             patch("rllm.sdk.session.otel.baggage") as mock_baggage, \
             patch("rllm.sdk.session.otel.context") as mock_context, \
             patch("rllm.sdk.session.otel.trace") as mock_trace:

            # Setup mocks
            mock_ctx = MagicMock()
            mock_context.get_current.return_value = mock_ctx
            mock_baggage.set_baggage.return_value = mock_ctx
            mock_trace.get_tracer.return_value.start_span.return_value.__enter__ = MagicMock()
            mock_trace.get_tracer.return_value.start_span.return_value.__exit__ = MagicMock()

            # Service A starts request
            storage_a = InMemoryStorage()
            with OTelSession(name="service_a", request_id="req_123", storage=storage_a) as session_a:
                ctx_a = session_a.to_otel_context()

                # Simulate passing to Service B (via HTTP, would be automatic with instrumentation)
                # Service B receives and restores context
                storage_b = InMemoryStorage()
                with OTelSession.from_otel_context(ctx_a, storage=storage_b) as session_b:
                    ctx_b = session_b.to_otel_context()

                    # Service C receives from Service B
                    storage_c = InMemoryStorage()
                    with OTelSession.from_otel_context(ctx_b, storage=storage_c) as session_c:
                        # All should have same request_id metadata
                        assert session_c.metadata.get("request_id") == "req_123"


class TestErrorHandlingComprehensive:
    """Comprehensive error handling tests."""

    def test_exception_in_session_context(self):
        """Test session handles exceptions in context properly."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            try:
                with OTelSession(name="test", storage=storage):
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected

            # Session should have cleaned up
            from rllm.sdk.session.otel import get_current_otel_session
            assert get_current_otel_session() is None

    def test_nested_exception_cleanup(self):
        """Test nested sessions clean up correctly on exception."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            from rllm.sdk.session.otel import get_active_otel_sessions

            try:
                with OTelSession(name="outer", storage=storage):
                    with OTelSession(name="middle", storage=storage):
                        with OTelSession(name="inner", storage=storage):
                            assert len(get_active_otel_sessions()) == 3
                            raise RuntimeError("Test error")
            except RuntimeError:
                pass

            # All sessions should be cleaned up
            assert len(get_active_otel_sessions()) == 0

    def test_storage_error_handling(self):
        """Test session handles storage errors gracefully."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            # Create a mock storage that raises errors
            mock_storage = MagicMock()
            mock_storage.get_traces.side_effect = Exception("Storage error")

            with OTelSession(name="test", storage=mock_storage) as session:
                # Accessing llm_calls should raise the storage error
                with pytest.raises(Exception, match="Storage error"):
                    _ = session.llm_calls

    def test_malformed_context_dict(self):
        """Test from_otel_context handles malformed data."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Missing required keys
            malformed_ctx = {"name": "test"}  # Missing metadata and session_uid_chain

            # Should handle gracefully or raise clear error
            try:
                session = OTelSession.from_otel_context(malformed_ctx, storage=storage)
                # If it doesn't raise, verify it has reasonable defaults
                assert session.name == "test"
            except (KeyError, TypeError):
                # Acceptable to raise error for malformed context
                pass


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code patterns."""

    def test_drop_in_replacement_for_contextvarsession(self):
        """Test OTelSession can replace ContextVarSession with minimal changes."""

        def user_code_with_session(SessionClass, storage):
            """Simulated user code that uses a session."""
            with SessionClass(name="my_session", experiment="test", storage=storage) as session:
                # User code that works with any session type
                assert session.name == "my_session"
                assert session.metadata["experiment"] == "test"
                assert len(session.llm_calls) >= 0
                return session._uid

        storage = InMemoryStorage()

        # Works with ContextVarSession
        uid1 = user_code_with_session(ContextVarSession, storage)

        # Works with OTelSession (just need to patch instrumentation)
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            uid2 = user_code_with_session(OTelSession, storage)

        # Both completed successfully with unique UIDs
        assert uid1 != uid2

    def test_existing_storage_patterns_still_work(self):
        """Test existing storage usage patterns work with OTelSession."""
        # Pattern 1: Default InMemoryStorage
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession() as session:
                # Should default to SqliteSessionStorage for OTelSession
                assert isinstance(session.storage, SqliteSessionStorage)

        # Pattern 2: Explicit InMemoryStorage
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            with OTelSession(storage=storage) as session:
                assert isinstance(session.storage, InMemoryStorage)

        # Pattern 3: SqliteSessionStorage
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")
            with OTelSession(storage=storage) as session:
                assert isinstance(session.storage, SqliteSessionStorage)


class TestSessionProtocolCompliance:
    """Test OTelSession complies with SessionProtocol."""

    def test_implements_required_properties(self):
        """Test OTelSession implements all required properties."""
        from rllm.sdk.session.base import SessionProtocol

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            session = OTelSession(name="test", storage=storage)

            # Check protocol compliance (will be checked at runtime)
            assert isinstance(session, SessionProtocol)

            # Required attributes
            assert hasattr(session, 'name')
            assert hasattr(session, 'metadata')

            # Required methods
            assert hasattr(session, 'llm_calls')
            assert hasattr(session, 'clear_calls')
            assert hasattr(session, '__enter__')
            assert hasattr(session, '__exit__')


class TestMetadataSlugIntegration:
    """Test integration with metadata slug encoding/decoding."""

    def test_slug_encoding_with_otel_metadata(self):
        """Test metadata from OTelSession can be encoded into slug."""
        from rllm.sdk.proxy.metadata_slug import encode_metadata_slug, decode_metadata_slug

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            with OTelSession(
                name="test",
                experiment="v1",
                dataset="train",
                run_id=123,
                storage=storage
            ) as session:
                from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata

                # Assemble metadata (should read from OTel)
                metadata = assemble_routing_metadata()

                # Encode into slug
                slug = encode_metadata_slug(metadata)

                # Decode and verify
                decoded = decode_metadata_slug(slug)
                assert "experiment" in decoded or "session_name" in decoded

    def test_url_building_with_otel_metadata(self):
        """Test building proxied URLs with OTelSession metadata."""
        from rllm.sdk.proxy.metadata_slug import build_proxied_base_url, assemble_routing_metadata

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            with OTelSession(
                name="test",
                experiment="v2",
                storage=storage
            ):
                # Assemble metadata
                metadata = assemble_routing_metadata()

                # Build URL
                base_url = "http://localhost:4000/v1"
                proxied_url = build_proxied_base_url(base_url, metadata)

                # Should have /meta/{slug} in path
                assert "/meta/" in proxied_url
                assert proxied_url.endswith("/v1")


class TestRealWorldUsagePatterns:
    """Test real-world usage patterns and workflows."""

    def test_training_loop_pattern(self):
        """Test typical RL training loop pattern."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")

            # Outer experiment session
            with OTelSession(name="experiment", experiment="ppo_v1", storage=storage) as exp:
                # Multiple episodes
                for episode in range(3):
                    with OTelSession(episode=episode, storage=storage) as ep:
                        # Multiple steps within episode
                        for step in range(5):
                            with OTelSession(step=step, storage=storage) as s:
                                # Simulate LLM call (would track automatically)
                                assert s.metadata.get("step") == step
                                assert s.metadata.get("episode") == episode
                                assert s.metadata.get("experiment") == "ppo_v1"

    def test_ab_testing_pattern(self):
        """Test A/B testing with different sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")

            # Variant A
            with OTelSession(variant="A", model="gpt-4", storage=storage) as session_a:
                a_uid = session_a._uid

            # Variant B
            with OTelSession(variant="B", model="gpt-3.5-turbo", storage=storage) as session_b:
                b_uid = session_b._uid

            # Each has unique identity
            assert a_uid != b_uid

    def test_user_session_tracking_pattern(self):
        """Test tracking different user sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")

            users = ["alice", "bob", "charlie"]
            user_sessions = {}

            for user in users:
                with OTelSession(user=user, storage=storage) as session:
                    user_sessions[user] = session._uid

            # Each user has unique session
            assert len(set(user_sessions.values())) == 3

    def test_debugging_session_pattern(self):
        """Test using sessions for debugging/replay."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            with OTelSession(
                name="debug_session",
                debug=True,
                timestamp="2024-01-15",
                storage=storage
            ) as session:
                # Add debug traces
                trace = Trace(
                    uid="debug_trace",
                    session_uid_chain=session._session_uid_chain,
                    session_name=session.name,
                    messages=[{"role": "user", "content": "debug query"}],
                    model="gpt-4",
                    response="debug response",
                    metadata=session.metadata
                )
                storage.add_trace(session._session_uid_chain, session.name, trace)

                # Retrieve for debugging
                debug_traces = session.llm_calls
                assert len(debug_traces) == 1
                assert debug_traces[0].metadata.get("debug") is True


class TestDocumentationExamples:
    """Test all code examples from documentation work correctly."""

    def test_basic_usage_example(self):
        """Test basic usage example from docs."""
        from rllm.sdk.session.otel import OTelSession

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            # Example from docs
            with OTelSession(experiment="v1", storage=storage) as session:
                # User code would call llm.chat.completions.create()
                assert session.name.startswith("sess_")
                assert len(session.llm_calls) >= 0

    def test_ray_example(self):
        """Test Ray integration example from docs."""
        from rllm.sdk.session.otel import OTelSession, ray_entrypoint

        @ray_entrypoint
        def train_episode(task):
            # Example worker function
            return f"Processed: {task}"

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")
            with OTelSession(experiment="v1", storage=storage) as session:
                ctx = session.to_otel_context()

                # Simulate Ray call
                result = train_episode("test_task", _otel_ctx=ctx)
                assert "Processed" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
