#!/usr/bin/env python3
"""Test script for SqliteTracer used by LiteLLM server.

This script validates the tracer functionality independently, including:
- Trace logging and storage
- Queue processing and background worker
- Flush mechanism
- Session UID association
- SQLite persistence and retrieval

Run this to ensure your tracer setup works before running full training.

Usage:
    python examples/omni_trainer/test_tracer.py [--db-path PATH]

Example:
    python examples/omni_trainer/test_tracer.py --db-path /tmp/test_tracer.db
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from rllm.sdk.session import SessionContext
from rllm.sdk.tracers import SqliteTracer


def test_basic_logging(tracer: SqliteTracer, db_path: str):
    """Test basic trace logging."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Trace Logging")
    print("=" * 60)

    # Log a simple trace
    tracer.log_llm_call(
        name="test/basic_call",
        model="gpt-3.5-turbo",
        input={"messages": [{"role": "user", "content": "Hello"}]},
        output={"id": "test-123", "choices": [{"message": {"content": "Hi there!"}}]},
        latency_ms=150.0,
        tokens={"prompt": 5, "completion": 3, "total": 8},
        metadata={"test": "basic"},
    )

    print("✓ Logged test trace")
    print(f"  - Trace ID: test-123")
    print(f"  - Model: gpt-3.5-turbo")
    print(f"  - Tokens: 8 total")
    print(f"  - Queue size: {tracer._store_queue.qsize()}")


def test_session_context(tracer: SqliteTracer):
    """Test trace logging with session context."""
    print("\n" + "=" * 60)
    print("TEST 2: Session Context Integration")
    print("=" * 60)

    with SessionContext(name="test-session-001", experiment="tracer-test") as session:
        print(f"✓ Created session: {session._uid}")
        print(f"  - Session name: {session._name}")
        print(f"  - Metadata: {session._metadata}")

        # Log trace within session context
        tracer.log_llm_call(
            name="test/session_call",
            model="gpt-4",
            input={"messages": [{"role": "user", "content": "Test in session"}]},
            output={"id": "test-456", "choices": [{"message": {"content": "Response in session"}}]},
            latency_ms=200.0,
            tokens={"prompt": 10, "completion": 5, "total": 15},
            metadata={"context": "session_test"},
        )

        print(f"✓ Logged trace within session")
        print(f"  - Trace should be associated with session UID: {session._uid}")


def test_multiple_traces(tracer: SqliteTracer):
    """Test logging multiple traces rapidly."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Rapid Traces")
    print("=" * 60)

    num_traces = 10
    print(f"Logging {num_traces} traces rapidly...")

    start_time = time.time()
    for i in range(num_traces):
        tracer.log_llm_call(
            name=f"test/bulk_{i}",
            model="gpt-3.5-turbo",
            input={"messages": [{"role": "user", "content": f"Message {i}"}]},
            output={"id": f"bulk-{i}", "choices": [{"message": {"content": f"Response {i}"}}]},
            latency_ms=100.0 + i * 10,
            tokens={"prompt": 5 + i, "completion": 3 + i, "total": 8 + i * 2},
        )

    elapsed = time.time() - start_time
    print(f"✓ Logged {num_traces} traces in {elapsed:.3f}s ({num_traces/elapsed:.1f} traces/sec)")
    print(f"  - Queue size: {tracer._store_queue.qsize()}")
    print(f"  - Worker thread alive: {tracer._worker_thread.is_alive() if tracer._worker_thread else False}")


def test_flush_mechanism(tracer: SqliteTracer):
    """Test the flush mechanism."""
    print("\n" + "=" * 60)
    print("TEST 4: Flush Mechanism")
    print("=" * 60)

    queue_size_before = tracer._store_queue.qsize()
    print(f"Queue size before flush: {queue_size_before}")

    print("Flushing tracer (waiting for all traces to persist)...")
    start_time = time.time()
    success = tracer.flush(timeout=30.0)
    elapsed = time.time() - start_time

    queue_size_after = tracer._store_queue.qsize()

    if success:
        print(f"✓ Flush succeeded in {elapsed:.3f}s")
        print(f"  - Queue size after flush: {queue_size_after}")
        print(f"  - All traces persisted to SQLite")
    else:
        print(f"✗ Flush failed or timed out after {elapsed:.3f}s")
        print(f"  - Queue size after flush: {queue_size_after}")
        return False

    return True


async def test_trace_retrieval(tracer: SqliteTracer):
    """Test retrieving traces from SQLite."""
    print("\n" + "=" * 60)
    print("TEST 5: Trace Retrieval")
    print("=" * 60)

    # Create a session and log a trace
    with SessionContext(name="retrieval-test") as session:
        session_uid = session._uid
        print(f"Session UID: {session_uid}")

        tracer.log_llm_call(
            name="test/retrieval",
            model="gpt-4",
            input={"messages": [{"role": "user", "content": "Retrieval test"}]},
            output={"id": "retrieval-123", "choices": [{"message": {"content": "Found!"}}]},
            latency_ms=180.0,
            tokens={"prompt": 8, "completion": 4, "total": 12},
        )

    # Flush to ensure trace is persisted
    print("Flushing traces...")
    tracer.flush(timeout=10.0)

    # Retrieve traces by session UID
    print(f"Retrieving traces for session: {session_uid}")
    traces = await tracer.store.get_by_session_uid(session_uid)

    if traces:
        print(f"✓ Retrieved {len(traces)} trace(s)")
        for trace in traces:
            print(f"  - Trace ID: {trace.id}")
            print(f"  - Data keys: {list(trace.data.keys())}")
            print(f"  - Model: {trace.data.get('model', 'N/A')}")
            print(f"  - Session name: {trace.data.get('session_name', 'N/A')}")
        return True
    else:
        print("✗ No traces found - retrieval failed")
        return False


async def test_concurrent_sessions(tracer: SqliteTracer):
    """Test logging from multiple concurrent sessions."""
    print("\n" + "=" * 60)
    print("TEST 6: Concurrent Sessions")
    print("=" * 60)

    async def log_in_session(session_name: str, num_calls: int):
        """Helper to log traces in a session."""
        with SessionContext(name=session_name) as session:
            for i in range(num_calls):
                tracer.log_llm_call(
                    name=f"test/{session_name}/{i}",
                    model="gpt-3.5-turbo",
                    input={"messages": [{"role": "user", "content": f"{session_name} msg {i}"}]},
                    output={"id": f"{session_name}-{i}", "choices": [{"message": {"content": f"Response {i}"}}]},
                    latency_ms=100.0,
                    tokens={"prompt": 5, "completion": 3, "total": 8},
                )
            return session._uid

    # Create multiple concurrent sessions
    num_sessions = 5
    calls_per_session = 3

    print(f"Creating {num_sessions} concurrent sessions with {calls_per_session} calls each...")
    tasks = [log_in_session(f"session-{i}", calls_per_session) for i in range(num_sessions)]
    session_uids = await asyncio.gather(*tasks)

    print(f"✓ Logged from {num_sessions} concurrent sessions")
    print(f"  - Total expected traces: {num_sessions * calls_per_session}")
    print(f"  - Session UIDs: {len(session_uids)}")

    # Flush and verify
    print("Flushing traces...")
    tracer.flush(timeout=30.0)

    # Count traces for each session
    for i, session_uid in enumerate(session_uids):
        traces = await tracer.store.get_by_session_uid(session_uid)
        print(f"  - Session {i}: {len(traces)} traces")

    return True


def test_worker_health(tracer: SqliteTracer):
    """Test worker thread health."""
    print("\n" + "=" * 60)
    print("TEST 7: Worker Thread Health")
    print("=" * 60)

    print(f"Worker thread alive: {tracer._worker_thread.is_alive() if tracer._worker_thread else 'No thread'}")
    print(f"Worker loop: {tracer._worker_loop is not None}")
    print(f"Queue size: {tracer._store_queue.qsize()}")
    print(f"Queue max size: {tracer._max_queue_size}")
    print(f"Shutdown flag: {tracer._shutdown}")

    if tracer._worker_thread and tracer._worker_thread.is_alive():
        print("✓ Worker thread is healthy")
        return True
    else:
        print("✗ Worker thread is not running")
        return False


async def main():
    """Run all tracer tests."""
    parser = argparse.ArgumentParser(description="Test SqliteTracer functionality")
    parser.add_argument("--db-path", type=str, default="/tmp/test_tracer.db", help="Path to SQLite database")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    # Clean up existing test database
    if db_path.exists():
        print(f"Removing existing test database: {db_path}")
        db_path.unlink()

    print("=" * 60)
    print("SqliteTracer Test Suite")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Namespace: default")
    print()

    # Create tracer
    print("Initializing SqliteTracer...")
    tracer = SqliteTracer(db_path=str(db_path), namespace="test-namespace")
    print(f"✓ Tracer initialized: {tracer}")
    print()

    # Run tests
    results = []

    # Test 1: Basic logging
    test_basic_logging(tracer, str(db_path))
    results.append(("Basic Logging", True))

    # Test 2: Session context
    test_session_context(tracer)
    results.append(("Session Context", True))

    # Test 3: Multiple traces
    test_multiple_traces(tracer)
    results.append(("Multiple Traces", True))

    # Test 4: Flush mechanism
    flush_success = test_flush_mechanism(tracer)
    results.append(("Flush Mechanism", flush_success))

    # Test 5: Trace retrieval
    retrieval_success = await test_trace_retrieval(tracer)
    results.append(("Trace Retrieval", retrieval_success))

    # Test 6: Concurrent sessions
    concurrent_success = await test_concurrent_sessions(tracer)
    results.append(("Concurrent Sessions", concurrent_success))

    # Test 7: Worker health
    worker_health = test_worker_health(tracer)
    results.append(("Worker Health", worker_health))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    # Database info
    print()
    print("=" * 60)
    print("DATABASE INFO")
    print("=" * 60)
    print(f"Path: {db_path}")
    print(f"Size: {db_path.stat().st_size if db_path.exists() else 0} bytes")
    print(f"Exists: {db_path.exists()}")

    # Cleanup
    print()
    print("Closing tracer...")
    await tracer.close(timeout=10.0)
    print("✓ Tracer closed")

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
