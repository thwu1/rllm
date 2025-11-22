"""Chat provider clients exposed by the RLLM SDK."""

from rllm.sdk.chat.util import (
    TimedCall,
    extract_completion_tokens,
    extract_usage_tokens,
    merge_args,
)
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
    # Utility functions
    "TimedCall",
    "merge_args",
    "extract_completion_tokens",
    "extract_usage_tokens",
    # Client implementations
    "SimpleTrackedChatClient",
    "SimpleTrackedAsyncChatClient",
    "ProxyTrackedChatClient",
    "ProxyTrackedAsyncChatClient",
    "OpenTelemetryTrackedChatClient",
    "OpenTelemetryTrackedAsyncChatClient",
    "OpenAIOTelClient",
    "AsyncOpenAIOTelClient",
]
