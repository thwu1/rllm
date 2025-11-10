"""Tests for SDK shortcuts module."""

import os

import pytest

from rllm.sdk import get_chat_client, get_chat_client_async, session
from rllm.sdk.context import get_current_metadata, get_current_session

# ============================================================================
# Session Shortcut Tests
# ============================================================================


def test_session_shortcut_basic():
    """Test basic session shortcut usage."""
    # Outside context, no session
    assert get_current_session() is None
    assert get_current_metadata() == {}

    # Inside context, session is set
    with session("test-session"):
        assert get_current_session() == "test-session"
        assert get_current_metadata() == {}

    # After context, session is cleared
    assert get_current_session() is None
    assert get_current_metadata() == {}


def test_session_shortcut_with_metadata():
    """Test session shortcut with metadata."""
    with session("test-session", experiment="v1", user="alice"):
        assert get_current_session() == "test-session"
        assert get_current_metadata() == {"experiment": "v1", "user": "alice"}


def test_session_shortcut_auto_generated_id():
    """Test session shortcut with auto-generated session ID."""
    with session(experiment="v1"):
        session_id = get_current_session()
        assert session_id is not None
        assert session_id.startswith("session-")
        assert get_current_metadata() == {"experiment": "v1"}


def test_session_shortcut_nested():
    """Test nested session contexts with metadata inheritance."""
    with session("outer", experiment="v1"):
        assert get_current_session() == "outer"
        assert get_current_metadata() == {"experiment": "v1"}

        with session("inner", task="math"):
            assert get_current_session() == "inner"
            # Inner inherits experiment="v1" and adds task="math"
            assert get_current_metadata() == {"experiment": "v1", "task": "math"}

        # Back to outer context
        assert get_current_session() == "outer"
        assert get_current_metadata() == {"experiment": "v1"}


def test_session_shortcut_metadata_override():
    """Test that nested session can override parent metadata."""
    with session("outer", experiment="v1", model="gpt-3"):
        assert get_current_metadata() == {"experiment": "v1", "model": "gpt-3"}

        with session("inner", model="gpt-4"):
            # Inner overrides model but keeps experiment
            assert get_current_metadata() == {"experiment": "v1", "model": "gpt-4"}

        # Back to outer context
        assert get_current_metadata() == {"experiment": "v1", "model": "gpt-3"}


def test_session_shortcut_multiple_sequential():
    """Test multiple sequential sessions don't interfere."""
    with session("session-1", tag="first"):
        assert get_current_session() == "session-1"
        assert get_current_metadata() == {"tag": "first"}

    # Between sessions, context is cleared
    assert get_current_session() is None
    assert get_current_metadata() == {}

    with session("session-2", tag="second"):
        assert get_current_session() == "session-2"
        assert get_current_metadata() == {"tag": "second"}


def test_session_shortcut_empty():
    """Test session shortcut with no arguments."""
    with session():
        session_id = get_current_session()
        assert session_id is not None
        assert session_id.startswith("session-")
        assert get_current_metadata() == {}


# ============================================================================
# Chat Client Shortcut Tests
# ============================================================================


def test_get_chat_client_basic():
    """Test basic get_chat_client usage."""
    llm = get_chat_client(api_key="test-key", model="gpt-4")
    assert llm is not None
    assert llm.default_model == "gpt-4"
    assert hasattr(llm, "chat")
    assert hasattr(llm.chat, "completions")


def test_get_chat_client_with_base_url():
    """Test get_chat_client with base_url (proxy mode)."""
    llm = get_chat_client(api_key="test-key", base_url="http://localhost:8000/v1", model="gpt-4")
    assert llm is not None
    assert llm.default_model == "gpt-4"
    # Should use ProxyTrackedChatClient when base_url is provided
    assert hasattr(llm, "_proxy_base_url")
    assert llm._proxy_base_url == "http://localhost:8000/v1"


def test_get_chat_client_from_env():
    """Test get_chat_client using OPENAI_API_KEY from environment."""
    # Save original env var
    original_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Set test API key in environment
        os.environ["OPENAI_API_KEY"] = "test-env-key"

        llm = get_chat_client(model="gpt-4")
        assert llm is not None
        assert llm.default_model == "gpt-4"
    finally:
        # Restore original env var
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)


def test_get_chat_client_no_api_key():
    """Test get_chat_client raises error when no API key is provided."""
    # Save original env var
    original_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Remove API key from environment
        os.environ.pop("OPENAI_API_KEY", None)

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            get_chat_client(model="gpt-4")
    finally:
        # Restore original env var
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key


def test_get_chat_client_unsupported_provider():
    """Test get_chat_client raises error for unsupported provider."""
    with pytest.raises(ValueError, match="Unsupported chat provider"):
        get_chat_client(provider="anthropic", api_key="test-key")


def test_get_chat_client_with_options():
    """Test get_chat_client with additional options."""
    llm = get_chat_client(api_key="test-key", model="gpt-4", organization="org-123", timeout=60.0, max_retries=3)
    assert llm is not None
    assert llm.default_model == "gpt-4"


def test_get_chat_client_async_basic():
    """Test basic get_chat_client_async usage."""
    llm = get_chat_client_async(api_key="test-key", model="gpt-4")
    assert llm is not None
    assert llm.default_model == "gpt-4"
    assert hasattr(llm, "chat")
    assert hasattr(llm.chat, "completions")


def test_get_chat_client_async_with_base_url():
    """Test get_chat_client_async with base_url (proxy mode)."""
    llm = get_chat_client_async(api_key="test-key", base_url="http://localhost:8000/v1", model="gpt-4")
    assert llm is not None
    assert llm.default_model == "gpt-4"
    # Should use ProxyTrackedAsyncChatClient when base_url is provided
    assert hasattr(llm, "_proxy_base_url")
    assert llm._proxy_base_url == "http://localhost:8000/v1"


def test_get_chat_client_async_no_api_key():
    """Test get_chat_client_async raises error when no API key is provided."""
    # Save original env var
    original_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Remove API key from environment
        os.environ.pop("OPENAI_API_KEY", None)

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            get_chat_client_async(model="gpt-4")
    finally:
        # Restore original env var
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
