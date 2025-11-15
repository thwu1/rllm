#!/usr/bin/env python3
"""Performance and stress tests for OTelSession."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

from rllm.sdk.protocol import Trace
from rllm.sdk.session.otel import OTelSession, get_active_otel_sessions, get_current_otel_session
from rllm.sdk.session.storage import InMemoryStorage, SqliteSessionStorage


class TestPerformance:
    """Test performance characteristics of OTelSession."""

    def test_session_creation_overhead(self):
        """Test overhead of creating many sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            start = time.time()
            sessions = []
            for i in range(100):
                session = OTelSession(name=f"session_{i}", storage=storage)
                sessions.append(session)
            end = time.time()

            # Should be able to create 100 sessions quickly (< 1 second)
            elapsed = end - start
            assert elapsed < 1.0, f"Creating 100 sessions took {elapsed:.2f}s, expected < 1s"

    def test_context_enter_exit_overhead(self):
        """Test overhead of entering/exiting sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            start = time.time()
            for i in range(100):
                with OTelSession(name=f"session_{i}", storage=storage):
                    pass  # Just enter and exit
            end = time.time()

            # Should be able to enter/exit 100 sessions quickly
            elapsed = end - start
            assert elapsed < 2.0, f"100 enter/exit cycles took {elapsed:.2f}s, expected < 2s"

    def test_metadata_serialization_performance(self):
        """Test performance of to_otel_context serialization."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Create session with moderate metadata
            metadata = {f"key_{i}": f"value_{i}" for i in range(20)}
            session = OTelSession(name="test", storage=storage, **metadata)

            # Serialize many times
            start = time.time()
            for _ in range(1000):
                ctx = session.to_otel_context()
            end = time.time()

            # Should serialize 1000 times quickly
            elapsed = end - start
            assert elapsed < 1.0, f"1000 serializations took {elapsed:.2f}s, expected < 1s"

    def test_nested_session_depth_performance(self):
        """Test performance with deeply nested sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Create 50 levels of nesting
            def create_nested_sessions(depth, max_depth):
                if depth >= max_depth:
                    return
                with OTelSession(name=f"level_{depth}", storage=storage):
                    create_nested_sessions(depth + 1, max_depth)

            start = time.time()
            create_nested_sessions(0, 50)
            end = time.time()

            # Should handle deep nesting
            elapsed = end - start
            assert elapsed < 2.0, f"50-level nesting took {elapsed:.2f}s, expected < 2s"


class TestConcurrency:
    """Test concurrent session usage."""

    def test_concurrent_session_creation(self):
        """Test creating sessions concurrently from multiple threads."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            def create_session(thread_id):
                with OTelSession(name=f"thread_{thread_id}", storage=storage) as session:
                    # Verify session is set correctly in this thread's context
                    current = get_current_otel_session()
                    assert current is not None
                    assert current.name == f"thread_{thread_id}"
                    return session._uid

            # Create sessions concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(create_session, i) for i in range(50)]
                results = [f.result() for f in as_completed(futures)]

            # All sessions should have completed
            assert len(results) == 50

            # All UIDs should be unique
            assert len(set(results)) == 50

    def test_concurrent_nested_sessions(self):
        """Test nested sessions from multiple threads."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            def nested_session_worker(thread_id):
                with OTelSession(name=f"outer_{thread_id}", storage=storage):
                    outer_active = len(get_active_otel_sessions())
                    with OTelSession(name=f"inner_{thread_id}", storage=storage):
                        inner_active = len(get_active_otel_sessions())
                        # Should have 2 active sessions
                        assert inner_active == 2
                    # Back to 1 after inner exits
                    final_active = len(get_active_otel_sessions())
                    assert final_active == 1
                return thread_id

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(nested_session_worker, i) for i in range(20)]
                results = [f.result() for f in as_completed(futures)]

            assert len(results) == 20

    def test_concurrent_metadata_access(self):
        """Test concurrent access to session metadata."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            def access_metadata(session_id):
                with OTelSession(
                    name=f"session_{session_id}",
                    experiment="concurrent_test",
                    iteration=session_id,
                    storage=storage
                ):
                    # Access metadata multiple times
                    from rllm.sdk.session.otel import get_otel_metadata
                    for _ in range(10):
                        metadata = get_otel_metadata()
                        assert "experiment" in metadata
                        assert "iteration" in metadata
                return session_id

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(access_metadata, i) for i in range(30)]
                results = [f.result() for f in as_completed(futures)]

            assert len(results) == 30


class TestStress:
    """Stress tests for OTelSession."""

    def test_many_sequential_sessions(self):
        """Test creating many sessions sequentially."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            session_count = 500
            for i in range(session_count):
                with OTelSession(name=f"session_{i}", iteration=i, storage=storage) as session:
                    assert session.name == f"session_{i}"

                # Verify context is cleaned up
                assert get_current_otel_session() is None

    def test_rapid_enter_exit_cycles(self):
        """Test rapid cycling of session enter/exit."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            for _ in range(200):
                with OTelSession(storage=storage):
                    with OTelSession(storage=storage):
                        with OTelSession(storage=storage):
                            # 3 levels deep
                            assert len(get_active_otel_sessions()) == 3
                        assert len(get_active_otel_sessions()) == 2
                    assert len(get_active_otel_sessions()) == 1
                assert len(get_active_otel_sessions()) == 0

    def test_large_session_stack(self):
        """Test maintaining large stack of active sessions."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Create 100 nested sessions
            session_contexts = []

            for i in range(100):
                session = OTelSession(name=f"session_{i}", storage=storage)
                session.__enter__()
                session_contexts.append(session)

            # Verify all 100 are active
            active = get_active_otel_sessions()
            assert len(active) == 100

            # Verify stack order
            for i, s in enumerate(active):
                assert s.name == f"session_{i}"

            # Exit all in reverse order
            for session in reversed(session_contexts):
                session.__exit__(None, None, None)

            # All should be cleaned up
            assert len(get_active_otel_sessions()) == 0

    def test_stress_serialization_roundtrips(self):
        """Test many serialization roundtrips."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            ctx = {"name": "initial", "session_uid_chain": [], "metadata": {"count": 0}}

            # Do 100 roundtrips
            for i in range(100):
                session = OTelSession.from_otel_context(ctx, storage=storage)
                session.metadata["count"] = i
                ctx = session.to_otel_context()

            # Final session should have correct count
            final_session = OTelSession.from_otel_context(ctx, storage=storage)
            assert final_session.metadata["count"] == 99


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_session_cleanup_releases_memory(self):
        """Test sessions properly clean up after exit."""
        import gc
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Create and destroy many sessions
            for _ in range(100):
                with OTelSession(
                    name="test",
                    large_data="x" * 1000,  # 1KB string
                    storage=storage
                ):
                    pass

            # Force garbage collection
            gc.collect()

            # Context should be clean
            assert get_current_otel_session() is None
            assert len(get_active_otel_sessions()) == 0

    def test_storage_with_many_traces(self):
        """Test storage handles many traces efficiently."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            with OTelSession(name="test", storage=storage) as session:
                # Add many traces
                for i in range(100):
                    trace = Trace(
                        uid=f"trace_{i}",
                        session_uid_chain=session._session_uid_chain,
                        session_name=session.name,
                        messages=[{"role": "user", "content": f"message {i}"}],
                        model="gpt-4",
                        response=f"response {i}",
                        metadata={"index": i}
                    )
                    storage.add_trace(session._session_uid_chain, session.name, trace)

                # Retrieve all traces
                traces = session.llm_calls
                assert len(traces) == 100


class TestScalability:
    """Test scalability with various loads."""

    def test_many_concurrent_sessions_with_storage(self):
        """Test many concurrent sessions all writing to storage."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
            db_path = f.name

        try:
            with patch("rllm.sdk.session.otel._ensure_instrumentation"):

                def session_with_traces(session_id):
                    storage = SqliteSessionStorage(db_path)
                    with OTelSession(
                        name=f"session_{session_id}",
                        experiment="scale_test",
                        storage=storage
                    ) as session:
                        # Add a few traces
                        for i in range(5):
                            trace = Trace(
                                uid=f"s{session_id}_t{i}",
                                session_uid_chain=session._session_uid_chain,
                                session_name=session.name,
                                messages=[],
                                model="gpt-4",
                                response=f"response {i}",
                                metadata={}
                            )
                            storage.add_trace(session._session_uid_chain, session.name, trace)
                    return session_id

                # Run 20 concurrent sessions
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(session_with_traces, i) for i in range(20)]
                    results = [f.result() for f in as_completed(futures)]

                assert len(results) == 20

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_deep_nesting_with_metadata_inheritance(self):
        """Test deep nesting preserves metadata correctly."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            def create_nested(depth, max_depth, accumulated_metadata):
                if depth >= max_depth:
                    from rllm.sdk.session.otel import get_otel_metadata
                    metadata = get_otel_metadata()
                    # Should see all accumulated metadata
                    for key in accumulated_metadata:
                        assert key in metadata
                    return

                metadata_key = f"level_{depth}"
                accumulated_metadata.add(metadata_key)

                with OTelSession(storage=storage, **{metadata_key: depth}):
                    create_nested(depth + 1, max_depth, accumulated_metadata)

            # 20 levels of nesting
            create_nested(0, 20, set())


class TestEdgeCasePerformance:
    """Test performance in edge case scenarios."""

    def test_session_with_no_metadata_operations(self):
        """Test sessions with minimal metadata operations are fast."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            start = time.time()
            for _ in range(500):
                with OTelSession(storage=storage):
                    pass  # No metadata operations
            end = time.time()

            elapsed = end - start
            # Should be very fast with no metadata
            assert elapsed < 2.0, f"500 minimal sessions took {elapsed:.2f}s"

    def test_session_with_complex_metadata_operations(self):
        """Test sessions with complex metadata are still performant."""
        with patch("rllm.sdk.session.otel._ensure_instrumentation"):
            storage = InMemoryStorage()

            # Complex metadata
            complex_meta = {
                f"key_{i}_{j}": f"value_{i}_{j}"
                for i in range(5)
                for j in range(5)
            }

            start = time.time()
            for i in range(50):
                with OTelSession(storage=storage, **complex_meta):
                    from rllm.sdk.session.otel import get_otel_metadata
                    metadata = get_otel_metadata()
                    # Access metadata multiple times
                    for _ in range(10):
                        _ = metadata.get(f"key_0_0")
            end = time.time()

            elapsed = end - start
            # Should still be reasonable with complex metadata
            assert elapsed < 5.0, f"50 complex sessions took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
