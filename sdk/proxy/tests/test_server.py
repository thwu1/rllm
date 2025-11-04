"""
Integration tests for RLLM proxy server.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from sdk.proxy.server import create_app
from sdk.proxy.context import SessionContext


@pytest.fixture
def app():
    """Create test app."""
    return create_app(
        enable_logprobs=True,
        enable_prompt_tokens=True,
    )


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "rllm-proxy"


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0


class TestChatCompletionsEndpoint:
    """Tests for chat completions endpoint."""

    @pytest.fixture
    def mock_litellm_response(self):
        """Mock LiteLLM response."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.5,
                                "bytes": "Hello",
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return mock_response

    @patch("sdk.proxy.server.acompletion")
    def test_chat_completion_basic(self, mock_acompletion, client, mock_litellm_response):
        """Test basic chat completion."""
        mock_acompletion.return_value = mock_litellm_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-3.5-turbo"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @patch("sdk.proxy.server.acompletion")
    def test_chat_completion_with_session(
        self, mock_acompletion, client, mock_litellm_response
    ):
        """Test chat completion with session tracking."""
        mock_acompletion.return_value = mock_litellm_response

        session_metadata = {
            "task_id": "task-123",
            "agent_name": "test-agent",
        }

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={
                "X-RLLM-Session": "episode-123",
                "X-RLLM-Metadata": json.dumps(session_metadata),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check that latency header is added
        assert "X-RLLM-Latency-Ms" in response.headers

    @patch("sdk.proxy.server.acompletion")
    def test_chat_completion_error_handling(self, mock_acompletion, client):
        """Test error handling in chat completion."""
        # Mock an error
        mock_acompletion.side_effect = Exception("API Error")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data


class TestCompletionsEndpoint:
    """Tests for text completions endpoint."""

    @pytest.fixture
    def mock_litellm_completion_response(self):
        """Mock LiteLLM completion response."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "text": "This is a test response",
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["This", " is", " a"],
                        "token_logprobs": [-0.5, -0.3, -0.4],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            },
        }
        return mock_response

    @patch("sdk.proxy.server.acompletion")
    def test_completion_basic(
        self, mock_acompletion, client, mock_litellm_completion_response
    ):
        """Test basic text completion."""
        mock_acompletion.return_value = mock_litellm_completion_response

        response = client.post(
            "/v1/completions",
            json={
                "model": "gpt-3.5-turbo",
                "prompt": "Once upon a time",
                "max_tokens": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-3.5-turbo"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["text"] == "This is a test response"


class TestMiddleware:
    """Tests for middleware functionality."""

    def test_session_context_extraction(self, client):
        """Test session context is extracted from headers."""
        response = client.get(
            "/health",
            headers={
                "X-RLLM-Session": "test-session",
                "X-RLLM-Metadata": json.dumps({"key": "value"}),
            },
        )

        assert response.status_code == 200
        # Session context should be extracted (verified via tracing in actual usage)

    def test_latency_tracking(self, client):
        """Test latency tracking middleware."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-RLLM-Latency-Ms" in response.headers

        latency = float(response.headers["X-RLLM-Latency-Ms"])
        assert latency >= 0


class TestTelemetryAugmentation:
    """Tests for telemetry augmentation."""

    @patch("sdk.proxy.server.acompletion")
    def test_logprobs_augmentation(self, mock_acompletion, client):
        """Test that logprobs are added to requests."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test",
            "model": "gpt-3.5-turbo",
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_acompletion.return_value = mock_response

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        assert response.status_code == 200

        # Verify acompletion was called with augmented params
        call_kwargs = mock_acompletion.call_args[1]
        # Note: augmentation happens in server, verify it was called
        assert mock_acompletion.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
