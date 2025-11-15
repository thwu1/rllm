#!/usr/bin/env python3
"""Unit tests for dual-mode metadata assembly (OTelSession vs ContextVarSession)."""

import pytest

from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata
from rllm.sdk.session import ContextVarSession, OTelSession
from rllm.sdk.session.storage import InMemoryStorage


class TestDualModeMetadataAssembly:
    """Test metadata assembly auto-detects session type correctly."""

    def test_assemble_with_no_session(self):
        """Test metadata assembly with no active session."""
        metadata = assemble_routing_metadata()
        assert metadata == {}

    def test_assemble_with_contextvarsession(self):
        """Test metadata assembly reads from ContextVarSession."""
        storage = InMemoryStorage()
        with ContextVarSession(name="test_ctx", experiment="v1", storage=storage) as session:
            metadata = assemble_routing_metadata()

            assert "session_name" in metadata
            assert metadata["session_name"] == "test_ctx"
            assert "experiment" in metadata
            assert metadata["experiment"] == "v1"
            assert "session_uids" in metadata
            assert session._uid in metadata["session_uids"]

    def test_assemble_with_otelsession(self):
        """Test metadata assembly reads from OTelSession."""
        storage = InMemoryStorage()
        with OTelSession(name="test_otel", experiment="v2", storage=storage) as session:
            metadata = assemble_routing_metadata()

            assert "session_name" in metadata
            assert metadata["session_name"] == "test_otel"
            assert "experiment" in metadata
            assert metadata["experiment"] == "v2"
            assert "session_uids" in metadata

    def test_assemble_with_extra_metadata(self):
        """Test extra metadata is merged correctly."""
        storage = InMemoryStorage()
        with ContextVarSession(name="test", storage=storage):
            metadata = assemble_routing_metadata(extra={"custom": "value", "foo": "bar"})

            assert "session_name" in metadata
            assert metadata["custom"] == "value"
            assert metadata["foo"] == "bar"

    def test_assemble_with_nested_contextvarsession(self):
        """Test metadata assembly with nested ContextVarSession."""
        storage = InMemoryStorage()
        with ContextVarSession(name="outer", experiment="v1", storage=storage) as outer:
            with ContextVarSession(name="inner", batch="0", storage=storage) as inner:
                metadata = assemble_routing_metadata()

                # Should see merged metadata from both sessions
                assert metadata["experiment"] == "v1"
                assert metadata["batch"] == "0"
                # Session UIDs should include both
                assert "session_uids" in metadata
                assert len(metadata["session_uids"]) == 2

    def test_assemble_with_nested_otelsession(self):
        """Test metadata assembly with nested OTelSession."""
        storage = InMemoryStorage()
        with OTelSession(name="outer", experiment="v1", storage=storage) as outer:
            with OTelSession(name="inner", batch="0", storage=storage) as inner:
                metadata = assemble_routing_metadata()

                # Should see merged metadata from both sessions
                assert metadata["experiment"] == "v1"
                assert metadata["batch"] == "0"
                assert "session_uids" in metadata


class TestMetadataAssemblyEdgeCases:
    """Test edge cases in metadata assembly."""

    def test_extra_overrides_session_metadata(self):
        """Test extra metadata can override session metadata."""
        storage = InMemoryStorage()
        with ContextVarSession(experiment="v1", storage=storage):
            metadata = assemble_routing_metadata(extra={"experiment": "v2"})

            # Extra should override session metadata
            assert metadata["experiment"] == "v2"

    def test_session_name_not_duplicated(self):
        """Test session_name isn't duplicated if already in metadata."""
        storage = InMemoryStorage()
        with ContextVarSession(name="test", session_name="explicit", storage=storage):
            metadata = assemble_routing_metadata()

            # Should use the explicit session_name from metadata
            assert metadata["session_name"] == "explicit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
