"""Chat provider clients exposed by the RLLM SDK.

This module provides unified chat client implementations that support both
proxy-based tracing and optional local in-memory tracing.

Client Hierarchy:
- SimpleTrackedChatClient: Direct tracing without proxy involvement
- ProxyTrackedChatClient: Proxy + optional local tracing (default: enabled)
- OpenTelemetryTrackedChatClient: Proxy without local tracing (OTel mode)

The OpenTelemetry clients are now aliases for ProxyTracked clients with
`enable_local_tracing=False`, simplifying the codebase while maintaining
backward compatibility.
"""

from rllm.sdk.chat.util import (
    extract_completion_tokens,
    extract_usage_tokens,
    merge_args,
)
from rllm.sdk.chat.openai import (
    AsyncOpenAIOTelClient,
    OpenAIOTelClient,
    OpenTelemetryTrackedAsyncChatClient,
    OpenTelemetryTrackedChatClient,
    ProxyTrackedAsyncChatClient,
    ProxyTrackedChatClient,
)
from rllm.sdk.chat.simple_chat_client import SimpleTrackedAsyncChatClient, SimpleTrackedChatClient

__all__ = [
    # Utility functions
    "merge_args",
    "extract_completion_tokens",
    "extract_usage_tokens",
    # Client implementations
    "SimpleTrackedChatClient",
    "SimpleTrackedAsyncChatClient",
    "ProxyTrackedChatClient",
    "ProxyTrackedAsyncChatClient",
    # OTel-mode clients (backward-compatible aliases)
    "OpenTelemetryTrackedChatClient",
    "OpenTelemetryTrackedAsyncChatClient",
    "OpenAIOTelClient",
    "AsyncOpenAIOTelClient",
]
