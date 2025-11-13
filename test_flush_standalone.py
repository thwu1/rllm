#!/usr/bin/env python3
"""
Standalone comprehensive stress test for SqliteTracer flush function.

This test creates a minimal version of the tracer to avoid import dependencies,
focusing solely on testing the flush() functionality.
"""

import asyncio
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import aiosqlite

# Configure logging to see tracer internals
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Minimal SqliteTraceStore implementation (for testing)
# ============================================================================

class SqliteTraceStore:
    """Minimal SQLite store for testing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._sqlite_busy_timeout_ms = 5000
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._init_database()
            self._initialized = True

    async def _configure_connection(self, conn: aiosqlite.Connection) -> None:
        pragmas = [
            "PRAGMA journal_mode=DELETE",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA busy_timeout={self._sqlite_busy_timeout_ms}",
        ]
        for pragma in pragmas:
            try:
                await conn.execute(pragma)
            except Exception:
                pass

    async def _connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path, timeout=self._sqlite_busy_timeout_ms / 1000.0)
        await self._configure_connection(conn)
        return conn

    async def _init_database(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    context_type TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            await conn.commit()
        finally:
            await conn.close()

    async def store(
        self,
        trace_id: str,
        data: dict[str, Any],
        namespace: str = "default",
        context_type: str = "llm_trace",
        metadata: dict[str, Any] | None = None,
        session_uids: list[str] | None = None,
    ) -> None:
        await self._ensure_initialized()
        now = time.time()

        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT OR REPLACE INTO traces
                (id, context_type, namespace, data, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    context_type,
                    namespace,
                    json.dumps(data),
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            await conn.commit()
        finally:
            await conn.close()


# ============================================================================
# Minimal SqliteTracer implementation (for testing)
# ============================================================================

class SqliteTracer:
    """Minimal SQLite tracer for testing flush functionality."""

    _STOP = object()

    def __init__(self, db_path: str, max_queue_size: int = 10000):
        self.store = SqliteTraceStore(db_path=db_path)
        self.namespace = "default"

        self._store_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._shutdown = False

        self._worker_thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_started = threading.Event()

        self._start_background_worker()

    def _start_background_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(target=self._run_worker_loop, daemon=True, name="SqliteTracer-Worker")
        self._worker_thread.start()
        self._worker_started.wait(timeout=5.0)

    def _run_worker_loop(self) -> None:
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)
        self._worker_started.set()
        self._worker_loop.create_task(self._worker_coroutine())

        try:
            self._worker_loop.run_forever()
        finally:
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
            self._worker_loop = None

    async def _worker_coroutine(self) -> None:
        logger.info("[SqliteTracer._worker_coroutine] Worker started")
        while True:
            item = self._store_queue.get()

            if item is self._STOP:
                logger.info("[SqliteTracer._worker_coroutine] Stop sentinel received, exiting worker")
                break

            trace_id = item.get("trace_id", "unknown")
            logger.info(f"[SqliteTracer._worker_coroutine] Processing trace_id={trace_id}")

            try:
                await self._store_trace_with_retry(item)
                logger.info(f"[SqliteTracer._worker_coroutine] Successfully stored trace_id={trace_id}")
            except Exception as e:
                logger.exception(f"Worker error processing trace {trace_id}: {e}")
            finally:
                self._store_queue.task_done()

        if self._worker_loop and not self._worker_loop.is_closed():
            self._worker_loop.call_soon(self._worker_loop.stop)

    async def _store_trace_with_retry(self, item: dict[str, Any]) -> None:
        max_retries = 3
        retry_delays = [1, 2, 4]
        trace_id = item.get("trace_id", "unknown")

        for attempt in range(max_retries):
            try:
                await self.store.store(
                    trace_id=item["trace_id"],
                    data=item["data"],
                    namespace=item["namespace"],
                    context_type=item["context_type"],
                    metadata=item.get("metadata"),
                    session_uids=item.get("session_uids"),
                )
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to store trace {trace_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    logger.exception(f"Dropping trace {trace_id} after {max_retries} failed attempts: {e}")

    def _stop_worker_loop(self) -> None:
        if self._worker_loop is None:
            return

        try:
            def _stop():
                if self._worker_loop and not self._worker_loop.is_closed():
                    self._worker_loop.stop()

            self._worker_loop.call_soon_threadsafe(_stop)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
        except Exception as e:
            logger.exception("Error stopping worker loop: %s", e)
        finally:
            self._worker_loop = None

    def _queue_trace(
        self,
        trace_id: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        session_uids: list[str] | None = None,
    ) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_background_worker()

        try:
            queue_item = {
                "trace_id": trace_id,
                "data": data,
                "namespace": self.namespace,
                "context_type": "llm_trace",
                "metadata": metadata,
                "session_uids": session_uids,
            }
            self._store_queue.put_nowait(queue_item)
            logger.info(f"[SqliteTracer._queue_trace] Queued trace_id={trace_id}, queue_size={self._store_queue.qsize()}")
        except queue.Full:
            logger.warning(f"Store queue full (max size: {self._max_queue_size}), dropping trace {trace_id}")

    def log_llm_call(
        self,
        name: str,
        input: str | list | dict,
        output: str | dict,
        model: str,
        latency_ms: float,
        tokens: dict[str, int],
        **kwargs,
    ) -> None:
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"

        trace_data = {
            "name": name,
            "input": input,
            "output": output,
            "model": model,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "timestamp": time.time(),
        }

        self._queue_trace(
            trace_id=trace_id,
            data=trace_data,
            metadata=kwargs.get("metadata"),
            session_uids=kwargs.get("session_uids"),
        )

    def flush(self, timeout: float = 30.0) -> None:
        """
        Block until all queued traces are persisted (synchronous version).

        Args:
            timeout: Maximum time to wait in seconds
        """
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async flush - wait for queue to be processed
            loop.run_until_complete(asyncio.wait_for(asyncio.to_thread(self._store_queue.join), timeout=timeout))
        except Exception:
            # Best-effort: do not raise to callers
            pass

    def close_sync(self, timeout: float = 30.0) -> None:
        self._shutdown = True
        try:
            self._store_queue.put_nowait(self._STOP)
        except Exception:
            pass
        self._stop_worker_loop()


# ============================================================================
# Test Suite
# ============================================================================

class TestStats:
    """Track test statistics."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        logger.info(f"✓ PASSED: {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        logger.error(f"✗ FAILED: {test_name} - {error}")

    def print_summary(self):
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Passed: {self.passed}")
        print(f"Total Failed: {self.failed}")
        if self.errors:
            print("\nFailed Tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        print("="*80)
        return self.failed == 0


stats = TestStats()


def create_test_tracer() -> SqliteTracer:
    """Create a tracer with a temporary database."""
    db_path = tempfile.mktemp(suffix=".db")
    logger.info(f"Creating tracer with db_path: {db_path}")
    return SqliteTracer(db_path=db_path)


def test_basic_flush():
    """Test basic flush functionality - ensure it waits for items to be processed."""
    test_name = "Basic Flush"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue some traces
        num_traces = 10
        for i in range(num_traces):
            tracer.log_llm_call(
                name=f"test_call_{i}",
                input={"prompt": f"test {i}"},
                output={"response": f"response {i}"},
                model="gpt-4",
                latency_ms=100,
                tokens={"prompt": 10, "completion": 20, "total": 30},
            )

        logger.info(f"Queued {num_traces} traces, calling flush...")
        start_time = time.time()
        tracer.flush(timeout=10.0)
        flush_duration = time.time() - start_time

        logger.info(f"Flush completed in {flush_duration:.2f}s")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty after flush: {queue_size} items remaining")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_flush_timeout():
    """Test flush with short timeout - should not hang."""
    test_name = "Flush Timeout"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue a trace
        tracer.log_llm_call(
            name="test_call",
            input={"prompt": "test"},
            output={"response": "response"},
            model="gpt-4",
            latency_ms=100,
            tokens={"prompt": 10, "completion": 20, "total": 30},
        )

        # Flush with very short timeout
        start_time = time.time()
        tracer.flush(timeout=0.1)
        flush_duration = time.time() - start_time

        # Should complete quickly (within 2x timeout to account for overhead)
        if flush_duration > 0.5:
            stats.add_fail(test_name, f"Flush took too long: {flush_duration:.2f}s")
        else:
            logger.info(f"Flush completed in {flush_duration:.2f}s")
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_stress_high_volume():
    """Stress test with high volume of traces."""
    test_name = "High Volume Stress Test"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue many traces
        num_traces = 1000
        logger.info(f"Queueing {num_traces} traces...")

        start_queue = time.time()
        for i in range(num_traces):
            tracer.log_llm_call(
                name=f"stress_test_{i}",
                input={"prompt": f"stress test {i}" * 10},
                output={"response": f"response {i}" * 10},
                model="gpt-4",
                latency_ms=100 + i % 50,
                tokens={"prompt": 100 + i, "completion": 200 + i, "total": 300 + i},
            )
        queue_duration = time.time() - start_queue
        logger.info(f"Queued {num_traces} traces in {queue_duration:.2f}s")

        # Flush with generous timeout
        logger.info("Calling flush...")
        start_flush = time.time()
        tracer.flush(timeout=60.0)
        flush_duration = time.time() - start_flush

        logger.info(f"Flush completed in {flush_duration:.2f}s")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty after flush: {queue_size} items remaining")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_concurrent_flushes():
    """Test multiple concurrent flush calls."""
    test_name = "Concurrent Flushes"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue some traces
        num_traces = 100
        for i in range(num_traces):
            tracer.log_llm_call(
                name=f"concurrent_test_{i}",
                input={"prompt": f"test {i}"},
                output={"response": f"response {i}"},
                model="gpt-4",
                latency_ms=100,
                tokens={"prompt": 10, "completion": 20, "total": 30},
            )

        # Call flush from multiple threads
        num_threads = 5
        logger.info(f"Calling flush from {num_threads} threads concurrently...")

        def flush_worker(thread_id):
            logger.info(f"Thread {thread_id} calling flush")
            start = time.time()
            tracer.flush(timeout=30.0)
            duration = time.time() - start
            logger.info(f"Thread {thread_id} flush completed in {duration:.2f}s")
            return thread_id, duration

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(flush_worker, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]

        logger.info(f"All {num_threads} flushes completed")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty after concurrent flushes: {queue_size} items")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_flush_empty_queue():
    """Test flush on empty queue - should not hang."""
    test_name = "Flush Empty Queue"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Don't queue anything, just flush
        start_time = time.time()
        tracer.flush(timeout=5.0)
        flush_duration = time.time() - start_time

        # Should complete very quickly
        if flush_duration > 1.0:
            stats.add_fail(test_name, f"Flush took too long on empty queue: {flush_duration:.2f}s")
        else:
            logger.info(f"Flush completed in {flush_duration:.2f}s")
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_multiple_rapid_flushes():
    """Test multiple rapid flush calls in sequence."""
    test_name = "Multiple Rapid Flushes"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue some traces
        for i in range(50):
            tracer.log_llm_call(
                name=f"rapid_flush_test_{i}",
                input={"prompt": f"test {i}"},
                output={"response": f"response {i}"},
                model="gpt-4",
                latency_ms=100,
                tokens={"prompt": 10, "completion": 20, "total": 30},
            )

        # Call flush multiple times rapidly
        num_flushes = 10
        logger.info(f"Calling flush {num_flushes} times rapidly...")

        for i in range(num_flushes):
            start = time.time()
            tracer.flush(timeout=10.0)
            duration = time.time() - start
            logger.info(f"Flush {i+1}/{num_flushes} completed in {duration:.2f}s")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty after rapid flushes: {queue_size} items")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_flush_during_processing():
    """Test flush while worker is actively processing items."""
    test_name = "Flush During Processing"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue traces and immediately flush (worker will be processing)
        num_traces = 200
        logger.info(f"Queueing {num_traces} traces...")

        for i in range(num_traces):
            tracer.log_llm_call(
                name=f"processing_test_{i}",
                input={"prompt": f"test {i}"},
                output={"response": f"response {i}"},
                model="gpt-4",
                latency_ms=100,
                tokens={"prompt": 10, "completion": 20, "total": 30},
            )

        # Immediately flush (worker should still be processing)
        logger.info("Calling flush while worker is processing...")
        start_time = time.time()
        tracer.flush(timeout=30.0)
        flush_duration = time.time() - start_time

        logger.info(f"Flush completed in {flush_duration:.2f}s")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty: {queue_size} items remaining")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_flush_after_close():
    """Test flush after tracer is closed - should not hang or crash."""
    test_name = "Flush After Close"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue a trace
        tracer.log_llm_call(
            name="test_call",
            input={"prompt": "test"},
            output={"response": "response"},
            model="gpt-4",
            latency_ms=100,
            tokens={"prompt": 10, "completion": 20, "total": 30},
        )

        # Close the tracer
        tracer.close_sync()
        logger.info("Tracer closed")

        # Try to flush after close
        start_time = time.time()
        tracer.flush(timeout=5.0)
        flush_duration = time.time() - start_time

        # Should complete without hanging
        if flush_duration > 10.0:
            stats.add_fail(test_name, f"Flush hung after close: {flush_duration:.2f}s")
        else:
            logger.info(f"Flush after close completed in {flush_duration:.2f}s")
            stats.add_pass(test_name)

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_interleaved_queue_and_flush():
    """Test interleaving queue operations and flush calls."""
    test_name = "Interleaved Queue and Flush"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Queue, flush, queue, flush pattern
        for batch in range(5):
            logger.info(f"Batch {batch+1}: Queueing 20 traces...")
            for i in range(20):
                tracer.log_llm_call(
                    name=f"interleaved_batch{batch}_trace{i}",
                    input={"prompt": f"test {i}"},
                    output={"response": f"response {i}"},
                    model="gpt-4",
                    latency_ms=100,
                    tokens={"prompt": 10, "completion": 20, "total": 30},
                )

            logger.info(f"Batch {batch+1}: Flushing...")
            tracer.flush(timeout=10.0)
            queue_size = tracer._store_queue.qsize()
            logger.info(f"Batch {batch+1}: Queue size after flush: {queue_size}")

        # Final verification
        final_queue_size = tracer._store_queue.qsize()
        if final_queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty after all batches: {final_queue_size} items")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def test_very_large_payload():
    """Test flush with very large trace payloads."""
    test_name = "Very Large Payload"
    logger.info(f"\n{'='*60}\nTEST: {test_name}\n{'='*60}")

    try:
        tracer = create_test_tracer()

        # Create large payloads
        large_data = "x" * 100000  # 100KB string
        num_traces = 50

        logger.info(f"Queueing {num_traces} traces with large payloads...")
        for i in range(num_traces):
            tracer.log_llm_call(
                name=f"large_payload_{i}",
                input={"prompt": large_data, "index": i},
                output={"response": large_data, "index": i},
                model="gpt-4",
                latency_ms=100,
                tokens={"prompt": 10000, "completion": 20000, "total": 30000},
            )

        logger.info("Calling flush...")
        start_time = time.time()
        tracer.flush(timeout=60.0)
        flush_duration = time.time() - start_time

        logger.info(f"Flush completed in {flush_duration:.2f}s")

        # Verify queue is empty
        queue_size = tracer._store_queue.qsize()
        if queue_size != 0:
            stats.add_fail(test_name, f"Queue not empty: {queue_size} items remaining")
        else:
            stats.add_pass(test_name)

        tracer.close_sync()

    except Exception as e:
        stats.add_fail(test_name, f"Exception: {str(e)}")
        logger.exception(e)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SQLITE TRACER FLUSH FUNCTION STRESS TEST")
    print("="*80)

    tests = [
        test_basic_flush,
        test_flush_timeout,
        test_flush_empty_queue,
        test_flush_during_processing,
        test_multiple_rapid_flushes,
        test_interleaved_queue_and_flush,
        test_concurrent_flushes,
        test_stress_high_volume,
        test_very_large_payload,
        test_flush_after_close,
    ]

    for test_func in tests:
        try:
            test_func()
            time.sleep(0.5)  # Brief pause between tests
        except Exception as e:
            logger.exception(f"Test {test_func.__name__} crashed: {e}")
            stats.add_fail(test_func.__name__, f"Test crashed: {str(e)}")

    # Print summary
    success = stats.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
