"""Tests for context-based session and metadata tracking."""

import threading
import time

import pytest

from rllm.sdk import RLLMClient, get_current_metadata, get_current_session


def test_basic_session():
    """Test basic session context."""
    client = RLLMClient()

    # Outside context, no session
    assert get_current_session() is None
    assert get_current_metadata() == {}

    # Inside context, session available
    with client.session("test-session-1"):
        assert get_current_session() == "test-session-1"
        assert get_current_metadata() == {}

    # After context, back to None
    assert get_current_session() is None


def test_session_with_metadata():
    """Test session with custom metadata."""
    client = RLLMClient()

    with client.session("test-session-2", experiment="v1", user="alice"):
        assert get_current_session() == "test-session-2"
        metadata = get_current_metadata()
        assert metadata["experiment"] == "v1"
        assert metadata["user"] == "alice"


def test_auto_generated_session_id():
    """Test auto-generated session ID."""
    client = RLLMClient()

    with client.session(experiment="v1"):
        session_id = get_current_session()
        assert session_id is not None
        assert session_id.startswith("session-")
        assert get_current_metadata()["experiment"] == "v1"


def test_nested_sessions():
    """Test nested session contexts with metadata inheritance."""
    client = RLLMClient()

    with client.session("outer", experiment="v1", model="gpt-4"):
        assert get_current_session() == "outer"
        outer_meta = get_current_metadata()
        assert outer_meta["experiment"] == "v1"
        assert outer_meta["model"] == "gpt-4"

        with client.session("inner", task="math"):
            # Inner session changes session_id
            assert get_current_session() == "inner"
            # But inherits parent metadata
            inner_meta = get_current_metadata()
            assert inner_meta["experiment"] == "v1"  # Inherited
            assert inner_meta["model"] == "gpt-4"  # Inherited
            assert inner_meta["task"] == "math"  # Added

        # Back to outer session
        assert get_current_session() == "outer"
        assert get_current_metadata()["experiment"] == "v1"
        assert "task" not in get_current_metadata()


def test_metadata_override_in_nested():
    """Test that child can override parent metadata."""
    client = RLLMClient()

    with client.session("outer", experiment="v1", temperature=0.7):
        assert get_current_metadata()["experiment"] == "v1"
        assert get_current_metadata()["temperature"] == 0.7

        with client.session("inner", experiment="v2"):
            # Child overrides parent's experiment
            assert get_current_metadata()["experiment"] == "v2"
            # But still inherits temperature
            assert get_current_metadata()["temperature"] == 0.7

        # Parent's metadata unchanged
        assert get_current_metadata()["experiment"] == "v1"


def test_thread_safety():
    """Test that contexts are isolated across threads."""
    client = RLLMClient()
    results = {}

    def thread1():
        with client.session("thread-1", thread_name="one"):
            time.sleep(0.1)  # Let thread2 start
            results["thread1_session"] = get_current_session()
            results["thread1_meta"] = get_current_metadata()["thread_name"]

    def thread2():
        with client.session("thread-2", thread_name="two"):
            time.sleep(0.1)
            results["thread2_session"] = get_current_session()
            results["thread2_meta"] = get_current_metadata()["thread_name"]

    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Each thread should see its own context
    assert results["thread1_session"] == "thread-1"
    assert results["thread1_meta"] == "one"
    assert results["thread2_session"] == "thread-2"
    assert results["thread2_meta"] == "two"


def test_multiple_sequential_sessions():
    """Test multiple sessions in sequence."""
    client = RLLMClient()

    with client.session("session-1", run=1):
        assert get_current_session() == "session-1"
        assert get_current_metadata()["run"] == 1

    with client.session("session-2", run=2):
        assert get_current_session() == "session-2"
        assert get_current_metadata()["run"] == 2

    with client.session("session-3", run=3):
        assert get_current_session() == "session-3"
        assert get_current_metadata()["run"] == 3


def test_nested_function_calls():
    """Test that context propagates through nested function calls."""
    client = RLLMClient()

    def deeply_nested_function():
        return get_current_session(), get_current_metadata()

    def middle_function():
        return deeply_nested_function()

    def outer_function():
        return middle_function()

    with client.session("propagate-test", level="deep"):
        session_id, metadata = outer_function()
        assert session_id == "propagate-test"
        assert metadata["level"] == "deep"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
