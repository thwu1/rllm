"""Simple test of storage classes without full SDK imports."""

import sys

sys.path.insert(0, "/home/user/rllm")

# Direct imports to avoid torch dependency
from rllm.sdk.session.storage import InMemoryStorage, SessionStorage
from rllm.sdk.protocol import Trace
import time


def test_inmemory_storage():
    """Test InMemoryStorage directly."""
    print("Testing InMemoryStorage...")

    storage = InMemoryStorage()

    # Create a test trace
    trace = Trace(
        trace_id="tr_001",
        session_id="sess_123",
        name="test_call",
        input="test input",
        output="test output",
        model="gpt-4",
        latency_ms=100.0,
        tokens={"prompt": 10, "completion": 5, "total": 15},
        metadata={},
        timestamp=time.time(),
    )

    # Add trace
    session_uid = "ctx_abc"
    storage.add_trace(session_uid, trace)

    # Retrieve traces
    traces = storage.get_traces(session_uid)

    assert len(traces) == 1, f"Expected 1 trace, got {len(traces)}"
    assert traces[0].trace_id == "tr_001"
    assert traces[0].model == "gpt-4"

    print("✓ InMemoryStorage works!")

    # Test clear
    storage.clear(session_uid)
    traces_after_clear = storage.get_traces(session_uid)
    assert len(traces_after_clear) == 0, f"Expected 0 traces after clear, got {len(traces_after_clear)}"

    print("✓ InMemoryStorage.clear() works!")


def test_protocol():
    """Test that InMemoryStorage implements SessionStorage protocol."""
    print("\nTesting SessionStorage protocol...")

    storage = InMemoryStorage()

    # Check protocol
    assert isinstance(storage, SessionStorage)
    print("✓ InMemoryStorage implements SessionStorage protocol!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Storage Classes")
    print("=" * 60 + "\n")

    try:
        test_inmemory_storage()
        test_protocol()

        print("\n" + "=" * 60)
        print("✅ ALL STORAGE TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
