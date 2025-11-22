"""Chat provider clients exposed by the RLLM SDK."""

# from rllm.sdk.chat.openai_client import OpenAIChatClient  # TODO: Module doesn't exist yet
from rllm.sdk.chat.proxy_chat_client import ProxyTrackedAsyncChatClient, ProxyTrackedChatClient
from rllm.sdk.chat.simple_chat_client import SimpleTrackedAsyncChatClient, SimpleTrackedChatClient
from rllm.sdk.session import SESSION_BACKEND

# Conditionally import OTEL clients only if backend is "opentelemetry"
if SESSION_BACKEND == "opentelemetry":
    from rllm.sdk.chat.otel_tracked_client import (
        AsyncOpenAIOTelClient,
        OpenAIOTelClient,
        OpenTelemetryTrackedAsyncChatClient,
        OpenTelemetryTrackedChatClient,
    )
else:
    # Stub definitions when OTEL not in use
    OpenTelemetryTrackedChatClient = None  # type: ignore
    OpenTelemetryTrackedAsyncChatClient = None  # type: ignore
    OpenAIOTelClient = None  # type: ignore
    AsyncOpenAIOTelClient = None  # type: ignore

__all__ = [
    # "OpenAIChatClient",  # TODO: Module doesn't exist yet
    "SimpleTrackedChatClient",
    "SimpleTrackedAsyncChatClient",
    "ProxyTrackedChatClient",
    "ProxyTrackedAsyncChatClient",
    "OpenTelemetryTrackedChatClient",
    "OpenTelemetryTrackedAsyncChatClient",
    "OpenAIOTelClient",
    "AsyncOpenAIOTelClient",
]
