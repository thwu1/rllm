"""Tests for reward assignment functions."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rllm.sdk.reward import set_reward, set_reward_async


class MockResponse:
    """Mock LLM response object with id attribute."""

    def __init__(self, response_id: str):
        self.id = response_id


def test_set_reward_with_string_id():
    """Test set_reward with a string context_id."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published", "context_id": "reward_test-id", "reward": 1.0}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = set_reward("test-id", reward=1.0)

        assert result["status"] == "published"
        assert result["context_id"] == "reward_test-id"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "admin/publish-reward" in call_args[0][0]
        assert call_args[1]["json"]["context_id"] == "reward_test-id"
        assert call_args[1]["json"]["reward"] == 1.0
        assert call_args[1]["json"]["metadata"]["response_context_id"] == "test-id"


def test_set_reward_with_response_object():
    """Test set_reward with a response object."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published", "context_id": "reward_chatcmpl-123", "reward": 0.5}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        response_obj = MockResponse("chatcmpl-123")
        result = set_reward(response_obj, reward=0.5)

        assert result["status"] == "published"
        assert result["context_id"] == "reward_chatcmpl-123"
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["context_id"] == "reward_chatcmpl-123"
        assert call_args[1]["json"]["reward"] == 0.5
        assert call_args[1]["json"]["metadata"]["response_context_id"] == "chatcmpl-123"


def test_set_reward_with_metadata():
    """Test set_reward with metadata."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        metadata = {"quality": "good", "is_correct": True}
        set_reward("test-id", reward=1.0, metadata=metadata)

        call_args = mock_client.post.call_args
        # Should include both response_context_id and custom metadata
        assert call_args[1]["json"]["metadata"]["response_context_id"] == "test-id"
        assert call_args[1]["json"]["metadata"]["quality"] == "good"
        assert call_args[1]["json"]["metadata"]["is_correct"] is True


def test_set_reward_with_custom_proxy_url():
    """Test set_reward with custom proxy URL."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        set_reward("test-id", reward=1.0, proxy_url="http://custom:5000")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://custom:5000/admin/publish-reward"


def test_set_reward_with_admin_token():
    """Test set_reward with admin token."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        set_reward("test-id", reward=1.0, admin_token="secret-token")

        call_args = mock_client.post.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer secret-token"


def test_set_reward_env_vars():
    """Test set_reward uses environment variables."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with patch.dict(os.environ, {"LITELLM_PROXY_URL": "http://env:6000", "LITELLM_PROXY_ADMIN_TOKEN": "env-token"}):
            set_reward("test-id", reward=1.0)

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://env:6000/admin/publish-reward"
            assert call_args[1]["headers"]["Authorization"] == "Bearer env-token"


def test_set_reward_invalid_response_object():
    """Test set_reward raises error for invalid response object."""
    with pytest.raises(ValueError, match="Cannot extract id from response object"):
        set_reward({"no_id": "here"}, reward=1.0)


@pytest.mark.anyio
async def test_set_reward_async_with_string_id():
    """Test set_reward_async with a string context_id."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published", "context_id": "reward_test-id", "reward": 1.0}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await set_reward_async("test-id", reward=1.0)

        assert result["status"] == "published"
        assert result["context_id"] == "reward_test-id"


@pytest.mark.anyio
async def test_set_reward_async_with_response_object():
    """Test set_reward_async with a response object."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "published", "context_id": "reward_chatcmpl-456", "reward": 0.8}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        response_obj = MockResponse("chatcmpl-456")
        result = await set_reward_async(response_obj, reward=0.8)

        assert result["status"] == "published"
        assert result["context_id"] == "reward_chatcmpl-456"
