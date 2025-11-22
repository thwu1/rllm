"""Chat provider clients exposed by the RLLM SDK.

This module provides unified chat client implementations that support both
proxy-based tracing and optional local in-memory tracing.

All clients are now consolidated in the openai module:
- ProxyTrackedChatClient: Full-featured client with proxy and tracing support
- SimpleTrackedChatClient: Alias for ProxyTrackedChatClient with use_proxy=False
- OpenTelemetryTrackedChatClient: Alias with enable_local_tracing=False (OTel mode)

Configuration options:
- use_proxy: Enable/disable proxy URL metadata injection (default: True)
- enable_local_tracing: Enable/disable local trace logging (default: True)
- tracer: Custom tracer for logging (default: shared in-memory tracer)
"""

from rllm.sdk.chat.util import (
    extract_completion_tokens,
    extract_usage_tokens,
    merge_args,
)
from rllm.sdk.chat.openai import (
    # Core clients
    ProxyTrackedAsyncChatClient,
    ProxyTrackedChatClient,
    # Simple client aliases (use_proxy=False)
    SimpleTrackedAsyncChatClient,
    SimpleTrackedChatClient,
    # OTel client aliases (enable_local_tracing=False)
    AsyncOpenAIOTelClient,
    OpenAIOTelClient,
    OpenTelemetryTrackedAsyncChatClient,
    OpenTelemetryTrackedChatClient,
)

__all__ = [
    # Utility functions
    "merge_args",
    "extract_completion_tokens",
    "extract_usage_tokens",
    # Core client implementations
    "ProxyTrackedChatClient",
    "ProxyTrackedAsyncChatClient",
    # Simple client aliases (use_proxy=False)
    "SimpleTrackedChatClient",
    "SimpleTrackedAsyncChatClient",
    # OTel-mode client aliases (enable_local_tracing=False)
    "OpenTelemetryTrackedChatClient",
    "OpenTelemetryTrackedAsyncChatClient",
    "OpenAIOTelClient",
    "AsyncOpenAIOTelClient",
]
