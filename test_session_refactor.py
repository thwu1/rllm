"""Test script for refactored session with storage separation."""

import sys
import time

# Add parent to path to import sdk directly
sys.path.insert(0, "/home/user/rllm")

from rllm.sdk.protocol import Trace
from rllm.sdk.session import ContextVarSession, InMemoryStorage, SqliteSessionStorage
from rllm.sdk.tracers import InMemorySessionTracer


def test_inmemory_storage():
    """Test session with InMemoryStorage (default)."""
    print("\n" + "=" * 60)
    print("Test 1: InMemoryStorage (default)")
    print("=" * 60)

    # Create tracer
    tracer = InMemorySessionTracer()

    # Test 1: Default storage (InMemoryStorage)
    with ContextVarSession() as session:
        print(f"Session ID: {session.session_id}")
        print(f"Session UID: {session._uid}")
        print(f"Storage: {session.storage}")

        # Manually log a trace (simulating what a chat client would do)
        trace = Trace(
            trace_id="tr_001",
            session_id=session.session_id,
            name="chat.completions.create",
            input={"messages": [{"role": "user", "content": "Hello"}]},
            output={"content": "Hi there!"},
            model="gpt-4",
            latency_ms=1234.5,
            tokens={"prompt": 10, "completion": 5, "total": 15},
            metadata={"test": "inmemory"},
            timestamp=time.time(),
        )

        # Add via tracer (which should use storage)
        tracer.log_llm_call(
            name=trace.name,
            input=trace.input,
            output=trace.output,
            model=trace.model,
            latency_ms=trace.latency_ms,
            tokens=trace.tokens,
            metadata=trace.metadata,
        )

        # Verify we can retrieve it
        calls = session.llm_calls
        print(f"✓ Stored {len(calls)} trace(s)")
        assert len(calls) == 1
        assert calls[0].trace_id is not None
        assert calls[0].model == "gpt-4"
        print(f"✓ Retrieved trace: {calls[0].trace_id}")

    print("✓ InMemoryStorage test passed!")


def test_sqlite_storage():
    """Test session with SqliteSessionStorage."""
    print("\n" + "=" * 60)
    print("Test 2: SqliteSessionStorage (persistent)")
    print("=" * 60)

    import tempfile
    import os

    # Create temp database
    db_path = os.path.join(tempfile.gettempdir(), "test_session_storage.db")
    print(f"Database: {db_path}")

    # Remove if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    storage = SqliteSessionStorage(db_path)
    tracer = InMemorySessionTracer()

    # Create session with SQLite storage
    session_id = "test-session-123"
    with ContextVarSession(session_id=session_id, storage=storage) as session:
        print(f"Session ID: {session.session_id}")
        print(f"Session UID: {session._uid}")
        print(f"Storage: {session.storage}")

        # Log some traces
        for i in range(3):
            tracer.log_llm_call(
                name=f"call_{i}",
                input=f"input {i}",
                output=f"output {i}",
                model="gpt-4",
                latency_ms=100.0 * i,
                tokens={"prompt": 10, "completion": 5, "total": 15},
            )

        # Give SQLite a moment to write (it's async)
        time.sleep(0.5)

        # Verify retrieval
        calls = session.llm_calls
        print(f"✓ Stored {len(calls)} trace(s)")
        assert len(calls) == 3
        print(f"✓ Retrieved {len(calls)} traces from SQLite")

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)

    print("✓ SqliteSessionStorage test passed!")


def test_nested_sessions():
    """Test nested sessions with storage."""
    print("\n" + "=" * 60)
    print("Test 3: Nested sessions with InMemoryStorage")
    print("=" * 60)

    tracer = InMemorySessionTracer()

    with ContextVarSession(experiment="v1") as outer:
        print(f"Outer session: {outer.session_id}")

        # Log to outer session
        tracer.log_llm_call(
            name="outer_call",
            input="outer input",
            output="outer output",
            model="gpt-4",
            latency_ms=100.0,
            tokens={"prompt": 10, "completion": 5, "total": 15},
        )

        with ContextVarSession(task="math") as inner:
            print(f"Inner session: {inner.session_id}")

            # Log to inner session
            tracer.log_llm_call(
                name="inner_call",
                input="inner input",
                output="inner output",
                model="gpt-4",
                latency_ms=200.0,
                tokens={"prompt": 10, "completion": 5, "total": 15},
            )

            # Inner session should have 2 calls (inherits outer's trace)
            inner_calls = inner.llm_calls
            print(f"✓ Inner session has {len(inner_calls)} trace(s)")
            assert len(inner_calls) == 2

        # Outer session should still have just 2 calls
        outer_calls = outer.llm_calls
        print(f"✓ Outer session has {len(outer_calls)} trace(s)")
        assert len(outer_calls) == 2

    print("✓ Nested sessions test passed!")


def test_backward_compatibility():
    """Test that old code still works (default InMemoryStorage)."""
    print("\n" + "=" * 60)
    print("Test 4: Backward compatibility (no storage param)")
    print("=" * 60)

    # Old code that doesn't specify storage
    with ContextVarSession() as session:
        # Should default to InMemoryStorage
        assert isinstance(session.storage, InMemoryStorage)
        print(f"✓ Defaults to InMemoryStorage: {session.storage}")

        # Should still work with tracers
        tracer = InMemorySessionTracer()
        tracer.log_llm_call(
            name="test_call",
            input="test",
            output="test",
            model="gpt-4",
            latency_ms=100.0,
            tokens={"total": 15},
        )

        calls = session.llm_calls
        assert len(calls) == 1
        print(f"✓ Old code still works: {len(calls)} trace(s)")

    print("✓ Backward compatibility test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Refactored Session Implementation")
    print("=" * 60)

    try:
        test_inmemory_storage()
        test_sqlite_storage()
        test_nested_sessions()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
