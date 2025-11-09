"""Tests for @client.entrypoint() decorator."""

import asyncio
import multiprocessing as mp

from rllm.sdk import RLLMClient, get_current_metadata, get_current_session


def test_entrypoint_basic():
    """Test basic entrypoint decorator without metadata."""
    client = RLLMClient()

    @client.entrypoint
    def handle_request(payload):
        # Inside the function, context should be set
        session_id = get_current_session()
        metadata = get_current_metadata()
        return {"session_id": session_id, "metadata": metadata, "payload": payload}

    # Outside, no context
    assert get_current_session() is None

    # Call without _metadata - auto-generated session, no metadata
    result = handle_request({"input": "test"})
    assert result["session_id"] is not None  # Auto-generated
    assert result["metadata"] == {}  # No metadata
    assert result["payload"]["input"] == "test"

    # After call, context is cleaned up
    assert get_current_session() is None


def test_entrypoint_async():
    """Test entrypoint decorator with async function."""
    client = RLLMClient()

    @client.entrypoint
    async def handle_async_request(payload):
        # Inside the function, context should be set
        session_id = get_current_session()
        metadata = get_current_metadata()
        await asyncio.sleep(0.01)  # Simulate async work
        return {"session_id": session_id, "metadata": metadata, "payload": payload}

    # Run async function with _metadata
    result = asyncio.run(handle_async_request({"input": "async-test"}, _metadata={"service": "async-test", "environment": "staging"}))

    # Should have auto-generated session_id and metadata from _metadata
    assert result["session_id"] is not None
    assert result["metadata"]["service"] == "async-test"
    assert result["metadata"]["environment"] == "staging"


def test_entrypoint_preserves_signature():
    """Test that entrypoint preserves function signature and metadata."""
    client = RLLMClient()

    @client.entrypoint
    def my_function(x: int, y: str = "default") -> dict:
        """My docstring."""
        return {"x": x, "y": y}

    # Check that function metadata is preserved
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."

    # Check that function works correctly
    result = my_function(42, y="custom")
    assert result["x"] == 42
    assert result["y"] == "custom"


def test_entrypoint_nested_sessions():
    """Test that entrypoint works with nested sessions."""
    client = RLLMClient()

    @client.entrypoint
    def outer_handler(payload):
        metadata_outer = get_current_metadata()

        # Nested session should merge metadata
        with client.session(experiment="inner-exp", level="2"):
            metadata_inner = get_current_metadata()
            return {
                "outer": metadata_outer,
                "inner": metadata_inner,
            }

    result = outer_handler({"test": "data"}, _metadata={"service": "outer", "level": "1"})

    # Outer should have entrypoint metadata from _metadata
    assert result["outer"]["service"] == "outer"
    assert result["outer"]["level"] == "1"

    # Inner should merge both
    assert result["inner"]["service"] == "outer"  # Inherited
    assert result["inner"]["experiment"] == "inner-exp"  # Added
    assert result["inner"]["level"] == "2"  # Overridden


def test_entrypoint_dynamic_metadata():
    """Test that entrypoint accepts dynamic metadata from caller via _metadata kwarg."""
    client = RLLMClient()

    @client.entrypoint
    def my_handler(payload):
        session_id = get_current_session()
        metadata = get_current_metadata()
        return {"session_id": session_id, "metadata": metadata, "payload": payload}

    # Call without _metadata - auto-generated session, no metadata
    result1 = my_handler({"input": "test1"})
    assert result1["session_id"] is not None  # Auto-generated
    assert result1["metadata"] == {}  # No metadata

    # Call with _metadata - uses provided session_id and metadata
    result2 = my_handler({"input": "test2"}, _metadata={"session_id": "run-123", "experiment": "v1", "mode": "training"})
    assert result2["session_id"] == "run-123"  # From _metadata
    assert result2["metadata"]["mode"] == "training"  # From _metadata
    assert result2["metadata"]["experiment"] == "v1"  # From _metadata


def test_entrypoint_run_facade_pattern():
    """Test the Run Facade usage pattern: user decorates, facade passes _metadata."""
    client = RLLMClient()

    # User's code - simple decorator with no args
    @client.entrypoint()
    def my_agent(task):
        session_id = get_current_session()
        metadata = get_current_metadata()
        return {"task": task, "session_id": session_id, "metadata": metadata}

    # User calls normally - auto-generated session
    result1 = my_agent({"id": 1})
    assert result1["session_id"] is not None
    assert result1["metadata"] == {}

    # Run Facade calls with dynamic metadata
    result2 = my_agent(
        {"id": 2},
        _metadata={
            "session_id": "facade-run-456",
            "job": "training",
            "experiment": "v2",
            "batch": 1,
        },
    )
    assert result2["session_id"] == "facade-run-456"
    assert result2["metadata"]["job"] == "training"
    assert result2["metadata"]["experiment"] == "v2"
    assert result2["metadata"]["batch"] == 1


# ============================================================================
# Multiprocess Context Propagation Tests
# ============================================================================

# Create client at module level for pickling
_test_client = RLLMClient()


def worker_without_entrypoint(payload):
    """Worker function WITHOUT entrypoint - context will be lost."""
    from rllm.sdk.context import get_current_metadata, get_current_session

    return {
        "session_id": get_current_session(),
        "metadata": get_current_metadata(),
        "payload": payload,
    }


@_test_client.entrypoint
def worker_with_entrypoint(payload):
    """Worker function WITH entrypoint - context will be preserved."""
    from rllm.sdk.context import get_current_metadata, get_current_session

    return {
        "session_id": get_current_session(),
        "metadata": get_current_metadata(),
        "payload": payload,
    }


@_test_client.entrypoint
def my_agent_function(task):
    """User's agent that makes LLM calls."""
    from rllm.sdk.context import get_current_metadata, get_current_session

    # Simulate LLM call - context is automatically available
    session_id = get_current_session()
    metadata = get_current_metadata()

    return {
        "task": task,
        "session_id": session_id,
        "metadata": metadata,
        "result": "solved",
    }


# Module-level wrapper for multiprocess test (must be picklable)
def call_worker_with_metadata(payload):
    return worker_with_entrypoint(payload, _metadata={"service": "worker", "mode": "training"})


def call_agent_with_metadata(task):
    return my_agent_function(task, _metadata={"session_id": f"run-{task['id']}", "service": "my-agent", "experiment": "v1", "mode": "training"})


def test_multiprocess_without_entrypoint():
    """Test that context is LOST without entrypoint in multiprocess."""
    # This demonstrates the problem
    ctx = mp.get_context("spawn")

    # Set context in parent
    from rllm.sdk import RLLMClient

    client = RLLMClient()
    with client.session("parent-session", experiment="v1"):
        # Spawn child process
        with ctx.Pool(1) as pool:
            result = pool.apply(worker_without_entrypoint, ({"input": "test"},))

    # Context is LOST in child process
    assert result["session_id"] is None  # ❌ Lost!
    assert result["metadata"] == {}  # ❌ Lost!


def test_multiprocess_with_entrypoint():
    """Test that context is PRESERVED with entrypoint in multiprocess."""
    # This demonstrates the solution!
    ctx = mp.get_context("spawn")

    # Spawn child process - use module-level wrapper
    with ctx.Pool(1) as pool:
        result = pool.apply(call_worker_with_metadata, ({"input": "test"},))

    # Context is PRESERVED because entrypoint creates it from _metadata!
    assert result["session_id"] is not None  # ✅ Created by entrypoint!
    assert result["metadata"]["service"] == "worker"  # ✅ From _metadata!
    assert result["metadata"]["mode"] == "training"  # ✅ From _metadata!


def test_entrypoint_solves_run_facade_problem():
    """
    Demonstrate how @entrypoint solves the Run Facade multiprocess problem.

    The key insight:
    1. User wraps their agent function with @client.entrypoint
    2. Run Facade passes _metadata when calling the function
    3. When ProcessExecutor spawns a child process and calls the function
    4. The entrypoint decorator automatically creates a session context from _metadata
    5. All LLM calls inside get the correct metadata
    """
    # Simulate ProcessExecutor spawning child process
    ctx = mp.get_context("spawn")

    tasks = [{"id": 1, "input": "task1"}, {"id": 2, "input": "task2"}]

    # Use module-level wrapper that passes _metadata
    with ctx.Pool(2) as pool:
        results = pool.map(call_agent_with_metadata, tasks)

    # All results have correct context!
    for i, result in enumerate(results):
        assert result["session_id"] == f"run-{i + 1}"  # ✅ From _metadata
        assert result["metadata"]["service"] == "my-agent"  # ✅ From _metadata
        assert result["metadata"]["experiment"] == "v1"  # ✅ From _metadata
        assert result["metadata"]["mode"] == "training"  # ✅ From _metadata


if __name__ == "__main__":
    # Run the key test
    print("Testing multiprocess context propagation with @entrypoint...")
    test_multiprocess_with_entrypoint()
    print("✅ Success! Context preserved across process boundary.")

    print("\nTesting Run Facade scenario...")
    test_entrypoint_solves_run_facade_problem()
    print("✅ Success! Run Facade problem solved with @entrypoint.")
