#!/usr/bin/env python3
"""Unit tests for OTelSession distributed tracing."""

import pytest

from rllm.sdk.session.otel import (
    OTelSession,
    get_active_otel_sessions,
    get_current_otel_session,
    get_otel_metadata,
    get_otel_session_name,
)
from rllm.sdk.session.storage import InMemoryStorage, SqliteSessionStorage


class TestOTelSessionBasics:
    """Test basic OTelSession functionality."""

    def test_session_creation(self):
        """Test OTelSession can be created with default name."""
        session = OTelSession()
        assert session.name.startswith("sess_")
        assert session._uid.startswith("otel_")
        assert len(session._session_uid_chain) == 1

    def test_session_with_custom_name(self):
        """Test OTelSession can be created with custom name."""
        session = OTelSession(name="my_session")
        assert session.name == "my_session"

    def test_session_with_metadata(self):
        """Test OTelSession stores metadata correctly."""
        session = OTelSession(experiment="v1", user="alice", run_id=123)
        assert session.metadata == {"experiment": "v1", "user": "alice", "run_id": 123}

    def test_session_defaults_to_sqlite_storage(self):
        """Test OTelSession defaults to SqliteSessionStorage."""
        session = OTelSession()
        assert isinstance(session.storage, SqliteSessionStorage)

    def test_session_with_custom_storage(self):
        """Test OTelSession can use custom storage."""
        storage = InMemoryStorage()
        session = OTelSession(storage=storage)
        assert session.storage is storage


class TestOTelSessionContext:
    """Test OTelSession context manager behavior."""

    def test_context_manager_basic(self):
        """Test OTelSession works as context manager."""
        with OTelSession(name="test") as session:
            assert session.name == "test"
            assert get_current_otel_session() is session
            assert get_otel_session_name() == "test"

        # After exit, context should be cleared
        assert get_current_otel_session() is None

    def test_context_manager_with_metadata(self):
        """Test metadata is accessible in context."""
        with OTelSession(experiment="v1", dataset="train") as session:
            # Metadata accessible via OTel baggage
            metadata = get_otel_metadata()
            assert "experiment" in metadata
            assert "dataset" in metadata

    def test_active_sessions_stack(self):
        """Test active sessions stack tracking."""
        assert len(get_active_otel_sessions()) == 0

        with OTelSession(name="outer"):
            active = get_active_otel_sessions()
            assert len(active) == 1
            assert active[0].name == "outer"

            with OTelSession(name="inner"):
                active = get_active_otel_sessions()
                assert len(active) == 2
                assert active[0].name == "outer"
                assert active[1].name == "inner"

            # After inner exits
            active = get_active_otel_sessions()
            assert len(active) == 1
            assert active[0].name == "outer"

        # After all exit
        assert len(get_active_otel_sessions()) == 0


class TestOTelSessionNesting:
    """Test nested OTelSession behavior."""

    def test_nested_sessions_metadata_inheritance(self):
        """Test nested sessions inherit parent metadata."""
        storage = InMemoryStorage()

        with OTelSession(experiment="v1", storage=storage) as outer:
            outer_metadata = get_otel_metadata()
            assert outer_metadata.get("experiment") == "v1"

            with OTelSession(batch="0", storage=storage) as inner:
                inner_metadata = get_otel_metadata()
                # Inner should see both experiment and batch
                assert inner_metadata.get("experiment") == "v1"
                assert inner_metadata.get("batch") == "0"

            # Back to outer - should only see experiment
            outer_metadata = get_otel_metadata()
            assert outer_metadata.get("experiment") == "v1"
            assert "batch" not in outer_metadata

    def test_nested_sessions_uid_chain(self):
        """Test nested sessions build UID chain correctly."""
        storage = InMemoryStorage()

        with OTelSession(name="outer", storage=storage) as outer:
            assert len(outer._session_uid_chain) == 1

            with OTelSession(name="inner", storage=storage) as inner:
                assert len(inner._session_uid_chain) == 2
                # Inner's chain should start with outer's UID
                assert inner._session_uid_chain[0] == outer._uid


class TestOTelSessionSerialization:
    """Test OTelSession serialization for cross-process propagation."""

    def test_to_otel_context(self):
        """Test session serialization to dict."""
        session = OTelSession(name="test", experiment="v1", run_id="123")
        ctx = session.to_otel_context()

        assert ctx["name"] == "test"
        assert ctx["metadata"] == {"experiment": "v1", "run_id": "123"}
        assert "session_uid_chain" in ctx

    def test_from_otel_context(self):
        """Test session restoration from dict."""
        storage = InMemoryStorage()
        original = OTelSession(name="test", experiment="v1", storage=storage)
        ctx = original.to_otel_context()

        # Restore in "different process"
        restored = OTelSession.from_otel_context(ctx, storage=storage)

        assert restored.name == original.name
        assert restored.metadata == original.metadata

    def test_nested_session_serialization(self):
        """Test nested session can be serialized and restored."""
        storage = InMemoryStorage()

        with OTelSession(name="parent", experiment="v1", storage=storage) as parent:
            with OTelSession(name="child", batch="0", storage=storage) as child:
                ctx = child.to_otel_context()

                # Session UID chain should exclude current UID but include parent
                assert len(ctx["session_uid_chain"]) == 1
                assert ctx["session_uid_chain"][0] == parent._uid

                # Metadata should include both parent and child metadata
                # (Note: metadata only contains child's own metadata, not inherited)
                assert ctx["metadata"] == {"batch": "0"}


class TestOTelSessionStorage:
    """Test OTelSession storage integration."""

    def test_llm_calls_property_empty(self):
        """Test llm_calls returns empty list initially."""
        storage = InMemoryStorage()
        with OTelSession(storage=storage) as session:
            assert len(session.llm_calls) == 0
            assert session.llm_calls == []

    def test_session_length(self):
        """Test __len__ returns call count."""
        storage = InMemoryStorage()
        with OTelSession(storage=storage) as session:
            assert len(session) == 0

    def test_clear_calls_with_memory_storage(self):
        """Test clear_calls works with InMemoryStorage."""
        storage = InMemoryStorage()
        with OTelSession(storage=storage) as session:
            # Add a mock trace
            from rllm.sdk.protocol import Trace
            import time

            trace = Trace(
                trace_id="test_trace",
                session_name=session.name,
                name="test_call",
                input={"messages": []},
                output="test response",
                model="gpt-4",
                latency_ms=100.0,
                tokens={"prompt": 10, "completion": 20, "total": 30},
                timestamp=time.time(),
                metadata={},
            )
            storage.add_trace(session._session_uid_chain, session.name, trace)

            assert len(session.llm_calls) == 1
            session.clear_calls()
            assert len(session.llm_calls) == 0


class TestOTelSessionHelpers:
    """Test OTelSession helper functions."""

    def test_get_current_otel_session_none(self):
        """Test get_current_otel_session returns None outside context."""
        assert get_current_otel_session() is None

    def test_get_otel_session_name_none(self):
        """Test get_otel_session_name returns None outside context."""
        assert get_otel_session_name() is None

    def test_get_otel_metadata_empty(self):
        """Test get_otel_metadata returns empty dict outside context."""
        metadata = get_otel_metadata()
        assert metadata == {}

    def test_get_active_otel_sessions_empty(self):
        """Test get_active_otel_sessions returns empty list outside context."""
        assert get_active_otel_sessions() == []


class TestOTelSessionRepr:
    """Test OTelSession string representation."""

    def test_repr(self):
        """Test __repr__ includes key information."""
        storage = InMemoryStorage()
        session = OTelSession(name="test", storage=storage)
        repr_str = repr(session)

        assert "OTelSession" in repr_str
        assert "name='test'" in repr_str
        assert "_uid='otel_" in repr_str
        assert "chain_depth=1" in repr_str
        assert "InMemoryStorage" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
