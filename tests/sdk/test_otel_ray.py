#!/usr/bin/env python3
"""Tests for OTelSession Ray integration and cross-process serialization."""

import pytest

from rllm.sdk.protocol import Trace
from rllm.sdk.session.otel import OTelSession, ray_entrypoint
from rllm.sdk.session.storage import InMemoryStorage, SqliteSessionStorage


class TestRayEntrypointDecorator:
    """Test the ray_entrypoint decorator for automatic context restoration."""

    def test_ray_entrypoint_with_context(self):
        """Test ray_entrypoint restores context when _otel_ctx is passed."""

        @ray_entrypoint
        def worker_function(task, result_holder):
            # Inside worker, get current session
            from rllm.sdk.session.otel import get_current_otel_session, get_otel_metadata

            session = get_current_otel_session()
            result_holder['session_name'] = session.name if session else None
            result_holder['metadata'] = get_otel_metadata()
            return f"Processed: {task}"

        from unittest.mock import patch

        # Create parent session
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")
            with OTelSession(name="parent", experiment="v1", storage=storage) as parent:
                ctx = parent.to_otel_context()

                # Simulate calling Ray worker with _otel_ctx
                result_holder = {}
                output = worker_function("test_task", result_holder, _otel_ctx=ctx)

                # Verify context was restored
                assert output == "Processed: test_task"
                assert result_holder['session_name'] == "parent"
                assert result_holder['metadata'].get('experiment') == "v1"

    def test_ray_entrypoint_without_context(self):
        """Test ray_entrypoint works normally when no _otel_ctx passed."""

        @ray_entrypoint
        def worker_function(task):
            from rllm.sdk.session.otel import get_current_otel_session

            session = get_current_otel_session()
            return f"Processed: {task}, Session: {session}"

        # Call without _otel_ctx
        output = worker_function("test_task")
        assert "Processed: test_task" in output
        assert "None" in output  # No session active

    def test_ray_entrypoint_preserves_function_metadata(self):
        """Test ray_entrypoint preserves original function name and docstring."""

        @ray_entrypoint
        def my_worker(task):
            """Worker function docstring."""
            return task

        assert my_worker.__name__ == "my_worker"
        assert my_worker.__doc__ == "Worker function docstring."


class TestCrossProcessSerialization:
    """Test serialization for cross-process context propagation."""

    def test_to_otel_context_complete(self):
        """Test to_otel_context serializes all necessary data."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            session = OTelSession(
                name="test_session",
                experiment="v1",
                user="alice",
                run_id=123,
                storage=InMemoryStorage()
            )

            ctx = session.to_otel_context()

            # Verify all required fields
            assert "name" in ctx
            assert ctx["name"] == "test_session"
            assert "metadata" in ctx
            assert ctx["metadata"]["experiment"] == "v1"
            assert ctx["metadata"]["user"] == "alice"
            assert ctx["metadata"]["run_id"] == 123
            assert "session_uid_chain" in ctx

    def test_from_otel_context_restoration(self):
        """Test from_otel_context fully restores session state."""
        from unittest.mock import patch

        storage = InMemoryStorage()

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            # Create original session
            original = OTelSession(
                name="original",
                experiment="v1",
                dataset="train",
                storage=storage
            )

            # Serialize
            ctx = original.to_otel_context()

            # Restore in "different process"
            restored = OTelSession.from_otel_context(ctx, storage=storage)

            # Verify restoration
            assert restored.name == original.name
            assert restored.metadata == original.metadata
            assert isinstance(restored.storage, type(original.storage))

    def test_nested_session_serialization_preserves_chain(self):
        """Test nested session serialization preserves UID chain."""
        from unittest.mock import patch

        storage = InMemoryStorage()

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            with OTelSession(name="grandparent", storage=storage) as gp:
                with OTelSession(name="parent", storage=storage) as p:
                    with OTelSession(name="child", storage=storage) as c:
                        # Serialize child
                        ctx = c.to_otel_context()

                        # Chain should include grandparent and parent UIDs
                        assert len(ctx["session_uid_chain"]) == 2
                        assert ctx["session_uid_chain"][0] == gp._uid
                        assert ctx["session_uid_chain"][1] == p._uid

                        # Restore and verify chain continues
                        restored = OTelSession.from_otel_context(ctx, storage=storage)
                        assert len(restored._session_uid_chain) == 3

    def test_serialization_roundtrip(self):
        """Test multiple serialization roundtrips preserve data."""
        from unittest.mock import patch

        storage = InMemoryStorage()

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            # Original
            s1 = OTelSession(name="s1", exp="v1", storage=storage)
            ctx1 = s1.to_otel_context()

            # First restore
            s2 = OTelSession.from_otel_context(ctx1, storage=storage)
            ctx2 = s2.to_otel_context()

            # Second restore
            s3 = OTelSession.from_otel_context(ctx2, storage=storage)

            # All should have same name and metadata
            assert s1.name == s2.name == s3.name
            assert s1.metadata == s2.metadata == s3.metadata


class TestRayWorkerSimulation:
    """Simulate Ray worker scenarios."""

    def test_simulated_ray_worker_with_manual_propagation(self):
        """Simulate manual context propagation to Ray worker."""
        from unittest.mock import patch

        def simulate_ray_worker(ctx_dict, task):
            """Simulates code running in a Ray worker."""
            # Restore context
            storage = SqliteSessionStorage(":memory:")
            with OTelSession.from_otel_context(ctx_dict, storage=storage) as session:
                # Verify context available
                from rllm.sdk.session.otel import get_current_otel_session, get_otel_metadata

                current = get_current_otel_session()
                assert current is not None
                assert current.name == session.name

                metadata = get_otel_metadata()
                return {
                    "task": task,
                    "session_name": session.name,
                    "metadata": metadata
                }

        # Main process
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")
            with OTelSession(name="main", experiment="v1", run_id="123", storage=storage) as main_session:
                ctx = main_session.to_otel_context()

                # Simulate Ray remote call
                result = simulate_ray_worker(ctx, "process_data")

                # Verify worker had correct context
                assert result["session_name"] == "main"
                assert result["metadata"]["experiment"] == "v1"
                assert result["metadata"]["run_id"] == "123"

    def test_multiple_workers_with_different_contexts(self):
        """Test multiple workers with different session contexts."""
        from unittest.mock import patch

        def worker(ctx_dict, worker_id):
            storage = SqliteSessionStorage(":memory:")
            with OTelSession.from_otel_context(ctx_dict, storage=storage) as session:
                from rllm.sdk.session.otel import get_otel_metadata
                metadata = get_otel_metadata()
                return {
                    "worker_id": worker_id,
                    "experiment": metadata.get("experiment"),
                    "session_name": session.name
                }

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = SqliteSessionStorage(":memory:")

            # Create two different sessions
            with OTelSession(name="exp_a", experiment="exp_a", storage=storage) as session_a:
                ctx_a = session_a.to_otel_context()

            with OTelSession(name="exp_b", experiment="exp_b", storage=storage) as session_b:
                ctx_b = session_b.to_otel_context()

            # Simulate workers with different contexts
            result_a = worker(ctx_a, 1)
            result_b = worker(ctx_b, 2)

            # Verify each worker had correct context
            assert result_a["experiment"] == "exp_a"
            assert result_b["experiment"] == "exp_b"
            assert result_a["session_name"] == "exp_a"
            assert result_b["session_name"] == "exp_b"


class TestMultiprocessingScenarios:
    """Test scenarios similar to multiprocessing."""

    def test_shared_sqlite_storage_across_sessions(self):
        """Test multiple sessions can share SQLite storage."""
        from unittest.mock import patch
        import tempfile
        import os

        # Create temporary SQLite file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            with patch("rllm.sdk.session.otel._ensure_instrumentation"):
                # Session 1 writes a trace
                storage1 = SqliteSessionStorage(db_path)
                with OTelSession(name="session1", storage=storage1) as s1:
                    trace1 = Trace(
                        uid="trace_from_s1",
                        session_uid_chain=s1._session_uid_chain,
                        session_name=s1.name,
                        messages=[],
                        model="gpt-4",
                        response="response1",
                        metadata={}
                    )
                    storage1.add_trace(s1._session_uid_chain, s1.name, trace1)
                    s1_uid = s1._uid

                # Session 2 (simulating different process) reads from same storage
                storage2 = SqliteSessionStorage(db_path)
                # Note: Would need to pass UID chain in real multiprocessing scenario
                # This just tests storage sharing

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_context_propagation_with_nested_workers(self):
        """Test context propagation through nested worker calls."""
        from unittest.mock import patch

        def level2_worker(ctx):
            storage = InMemoryStorage()
            with OTelSession.from_otel_context(ctx, storage=storage) as session:
                from rllm.sdk.session.otel import get_otel_metadata
                return get_otel_metadata()

        def level1_worker(ctx):
            storage = InMemoryStorage()
            with OTelSession.from_otel_context(ctx, storage=storage) as session:
                # Nest another session
                with OTelSession(batch="nested", storage=storage) as nested:
                    nested_ctx = nested.to_otel_context()
                    # Call another worker
                    return level2_worker(nested_ctx)

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            with OTelSession(name="root", experiment="v1", storage=storage) as root:
                ctx = root.to_otel_context()

                # Propagate through nested workers
                final_metadata = level1_worker(ctx)

                # Should see both experiment (from root) and batch (from level1)
                assert "experiment" in final_metadata
                assert "batch" in final_metadata


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_session_with_empty_metadata(self):
        """Test session works with no metadata."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            with OTelSession(name="empty", storage=storage) as session:
                ctx = session.to_otel_context()

                assert ctx["metadata"] == {}

                restored = OTelSession.from_otel_context(ctx, storage=storage)
                assert restored.metadata == {}

    def test_session_with_none_values(self):
        """Test session handles None values in metadata."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()
            # Note: None values are converted to string "None" when set in baggage
            with OTelSession(name="test", value=None, storage=storage) as session:
                assert session.metadata == {"value": None}

    def test_session_with_large_metadata(self):
        """Test session handles large metadata (baggage size limits)."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Create large metadata dict (but still reasonable)
            large_metadata = {f"key_{i}": f"value_{i}" for i in range(50)}

            # Should work (OTel has ~8KB limit total, this should be under)
            with OTelSession(name="large", storage=storage, **large_metadata) as session:
                assert len(session.metadata) == 50

                ctx = session.to_otel_context()
                assert len(ctx["metadata"]) == 50

    def test_session_with_special_characters(self):
        """Test session handles special characters in metadata."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Special characters that might cause issues
            special_metadata = {
                "path": "/home/user/data",
                "query": "name=test&value=123",
                "unicode": "测试",
                "emoji": "🚀"
            }

            with OTelSession(name="special", storage=storage, **special_metadata) as session:
                ctx = session.to_otel_context()

                # Verify serialization preserves special chars
                assert ctx["metadata"]["path"] == "/home/user/data"
                assert ctx["metadata"]["unicode"] == "测试"
                assert ctx["metadata"]["emoji"] == "🚀"

                # Verify restoration
                restored = OTelSession.from_otel_context(ctx, storage=storage)
                assert restored.metadata == special_metadata

    def test_session_uid_uniqueness(self):
        """Test each session gets unique UID."""
        from unittest.mock import patch

        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            sessions = []
            for i in range(10):
                session = OTelSession(name=f"session_{i}", storage=storage)
                sessions.append(session)

            # All UIDs should be unique
            uids = [s._uid for s in sessions]
            assert len(uids) == len(set(uids))

            # All should start with "otel_"
            assert all(uid.startswith("otel_") for uid in uids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
