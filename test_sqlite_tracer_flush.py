#!/usr/bin/env python3
"""
Comprehensive stress test for SqliteTracer flush function.

This test suite verifies that the flush() method:
1. Doesn't hang under normal and stress conditions
2. Correctly waits for all queued items to be processed
3. Handles concurrent flush calls properly
4. Respects timeout parameters
5. Works correctly in edge cases
"""

import asyncio
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add rllm to path and import directly without going through __init__.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly to avoid loading heavy dependencies
import importlib.util

def load_module_from_path(module_name, file_path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load the modules we need directly
base_dir = os.path.dirname(os.path.abspath(__file__))
sqlite_module = load_module_from_path(
    "rllm.sdk.tracers.sqlite",
    os.path.join(base_dir, "rllm/sdk/tracers/sqlite.py")
)
SqliteTracer = sqlite_module.SqliteTracer

# Configure logging to see tracer internals
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


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
                input={"prompt": f"stress test {i}" * 10},  # Larger payload
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
