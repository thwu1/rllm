"""Tests for high-level RLLMClient chat access."""

from types import SimpleNamespace

import pytest

from sdk.client import RLLMClient


class StubTracer:
    def __init__(self):
        self.calls = []

    def log_llm_call(self, **kwargs):
        self.calls.append(kwargs)


class CapturingCompletions:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(model_dump=lambda: self.response_payload)


class CapturingTextCompletions:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(model_dump=lambda: self.response_payload)


class DummyOpenAIClient:
    def __init__(self, response_payload):
        self.chat_completions_stub = CapturingCompletions(response_payload)
        self.chat = SimpleNamespace(completions=self.chat_completions_stub)
        self.completions_stub = self.chat_completions_stub
        self.text_completions_stub = CapturingTextCompletions(response_payload)
        self.completions = self.text_completions_stub


class AsyncCapturingCompletions:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    async def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(model_dump=lambda: self.response_payload)


class AsyncCapturingTextCompletions:
    def __init__(self, response_payload):
        self.response_payload = response_payload
        self.calls = []

    async def create(self, **payload):
        self.calls.append(payload)
        return SimpleNamespace(model_dump=lambda: self.response_payload)


class DummyAsyncOpenAIClient:
    def __init__(self, response_payload):
        self.chat_completions_stub = AsyncCapturingCompletions(response_payload)
        self.chat = SimpleNamespace(completions=self.chat_completions_stub)
        self.completions_stub = self.chat_completions_stub
        self.text_completions_stub = AsyncCapturingTextCompletions(response_payload)
        self.completions = self.text_completions_stub


def test_get_chat_client_uses_simple_adapter_with_tracer():
    response_payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 123,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello-world"},
                "finish_reason": "stop",
                "output_token_ids": [101, 102],
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }

    tracer = StubTracer()
    dummy_openai = DummyOpenAIClient(response_payload)

    client = RLLMClient(api_key="dummy-key", tracer=tracer)
    chat = client.get_chat_client(model="mock-model", client=dummy_openai)

    response = chat.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

    assert response.model_dump() == response_payload
    assert dummy_openai.chat_completions_stub.calls
    assert tracer.calls
    assert tracer.calls[0]["model"] == "mock-model"


def test_get_chat_client_supports_text_completions():
    response_payload = {
        "id": "cmpl-mock",
        "object": "text_completion",
        "created": 234,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "text": "completion",
                "finish_reason": "stop",
                "output_token_ids": [301, 302],
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    tracer = StubTracer()
    dummy_openai = DummyOpenAIClient(response_payload)

    client = RLLMClient(api_key="dummy-key", tracer=tracer)
    chat = client.get_chat_client(model="mock-model", client=dummy_openai)

    response = chat.completions.create(prompt="hi there")

    assert response.model_dump() == response_payload
    assert dummy_openai.text_completions_stub.calls
    assert tracer.calls
    trace = tracer.calls[-1]
    assert trace["model"] == "mock-model"
    assert trace["input"]["messages"][0]["content"] == "hi there"
    assert hasattr(chat, "_sdk_owner")


@pytest.mark.asyncio
async def test_get_chat_client_async_returns_async_adapter():
    response_payload = {
        "id": "chatcmpl-async",
        "object": "chat.completion",
        "created": 456,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "async-hello"},
                "finish_reason": "stop",
                "output_token_ids": [201, 202],
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    tracer = StubTracer()
    dummy_openai = DummyAsyncOpenAIClient(response_payload)

    client = RLLMClient(api_key="dummy-key", tracer=tracer)
    chat = client.get_chat_client_async(model="mock-model", client=dummy_openai)

    response = await chat.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

    assert response.model_dump() == response_payload
    assert dummy_openai.chat_completions_stub.calls
    assert tracer.calls
    assert tracer.calls[0]["model"] == "mock-model"
    assert hasattr(chat, "_sdk_owner")


@pytest.mark.asyncio
async def test_get_chat_client_async_supports_text_completions():
    response_payload = {
        "id": "cmpl-async",
        "object": "text_completion",
        "created": 789,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "text": "async-completion",
                "finish_reason": "stop",
                "output_token_ids": [401, 402],
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }

    tracer = StubTracer()
    dummy_openai = DummyAsyncOpenAIClient(response_payload)

    client = RLLMClient(api_key="dummy-key", tracer=tracer)
    chat = client.get_chat_client_async(model="mock-model", client=dummy_openai)

    response = await chat.completions.create(prompt="hi there")

    assert response.model_dump() == response_payload
    assert dummy_openai.text_completions_stub.calls
    assert tracer.calls
    trace = tracer.calls[-1]
    assert trace["model"] == "mock-model"
    assert trace["input"]["messages"][0]["content"] == "hi there"
