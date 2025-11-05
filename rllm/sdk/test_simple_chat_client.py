"""Basic example-style tests for SimpleTrackedChatClient."""

from types import SimpleNamespace

from sdk.chat.simple_chat_client import SimpleTrackedChatClient


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


class DummyOpenAIClient:
    def __init__(self, response_payload):
        self.completions_stub = CapturingCompletions(response_payload)
        self.chat = SimpleNamespace(completions=self.completions_stub)


def test_simple_tracked_client_records_chat_completion():
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
                "output_token_ids": [101, 102, 103],
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }

    tracer = StubTracer()
    dummy_client = DummyOpenAIClient(response_payload)

    client = SimpleTrackedChatClient(
        tracer=tracer,
        default_model="mock-model",
        client=dummy_client,
    )

    messages = [{"role": "user", "content": "hi"}]
    response = client.chat.completions.create(messages=messages)

    assert response.model_dump() == response_payload
    assert dummy_client.completions_stub.calls
    assert len(tracer.calls) == 1

    trace = tracer.calls[0]
    assert trace["model"] == "mock-model"
    assert trace["output"]["choices"][0]["message"]["content"] == "hello-world"
    assert trace["metadata"]["token_ids"]["completion"] == [101, 102, 103]
