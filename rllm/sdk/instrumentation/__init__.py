"""Auto-instrumentation for LLM SDKs.

This module provides automatic instrumentation for LLM clients (OpenAI, etc.)
that captures traces within session contexts without requiring custom wrapper
clients.

Features:
- Automatic trace capture for all LLM calls within session() contexts
- Proxy URL modification for metadata routing (optional)
- In-memory tracing via InMemorySessionTracer
- Support for both ContextVar and OpenTelemetry session backends

Example:
    >>> from openai import AsyncOpenAI
    >>> from rllm.sdk import session
    >>> from rllm.sdk.instrumentation import instrument
    >>>
    >>> # Enable instrumentation (call once at startup)
    >>> instrument(proxy_urls=["http://proxy:4000"])
    >>>
    >>> # Use standard OpenAI client - traces are automatic!
    >>> client = AsyncOpenAI(base_url="http://proxy:4000/v1", api_key="...")
    >>>
    >>> with session(agent="solver") as sess:
    ...     response = await client.chat.completions.create(
    ...         model="gpt-4",
    ...         messages=[{"role": "user", "content": "Hello"}]
    ...     )
    ...     print(sess.llm_calls)  # Captured automatically!

Note:
    For most use cases, the factory pattern (`get_chat_client()`) is recommended
    as it's more explicit and safer. Use auto-instrumentation when:
    - Integrating with existing code that uses native clients
    - Tracing calls from third-party libraries
    - Client injection is not possible
"""

from __future__ import annotations

from typing import Sequence

from rllm.sdk.instrumentation.httpx_transport import (
    clear_proxy_urls,
    patch_httpx,
    register_proxy_url,
    unpatch_httpx,
)
from rllm.sdk.instrumentation.openai_provider import (
    instrument_openai,
    is_instrumented as is_openai_instrumented,
    uninstrument_openai,
)

# Track overall instrumentation state
_instrumented = False


def instrument(
    providers: Sequence[str] | None = None,
    proxy_urls: Sequence[str] | None = None,
) -> None:
    """Enable auto-instrumentation for LLM providers.

    After calling this, any LLM call within a session() context
    will be automatically traced and stored.

    This provides the same functionality as ProxyTrackedChatClient and
    OpenTelemetryTrackedChatClient:
    - Trace capture with all metadata (model, messages, response, latency, tokens)
    - In-memory tracing via InMemorySessionTracer for session.llm_calls
    - Proxy URL modification for metadata routing (if proxy_urls specified)
    - Session context reading from both ContextVar and OpenTelemetry backends

    Args:
        providers: List of providers to instrument. If None, instruments all available.
                  Currently supported: ["openai"]
        proxy_urls: List of base URLs that should have session metadata injected.
                   If None, no URL modification occurs (trace capture only).
                   Example: ["http://proxy:4000"]

    Example:
        >>> from rllm.sdk import instrument, session
        >>> from openai import AsyncOpenAI
        >>>
        >>> # Enable instrumentation with proxy URL modification
        >>> instrument(proxy_urls=["http://proxy:4000"])
        >>>
        >>> # Use standard OpenAI client
        >>> client = AsyncOpenAI(base_url="http://proxy:4000/v1", api_key="...")
        >>>
        >>> with session(agent="solver") as sess:
        ...     response = await client.chat.completions.create(
        ...         model="gpt-4",
        ...         messages=[{"role": "user", "content": "Hello"}]
        ...     )
        ...     print(sess.llm_calls)  # Traces captured!

    Note:
        - Call this once at application startup
        - For most use cases, get_chat_client() is recommended instead
        - Use this when you need to trace native clients or third-party libraries
    """
    global _instrumented

    if _instrumented:
        return

    # Register proxy URLs for URL modification
    if proxy_urls:
        for url in proxy_urls:
            register_proxy_url(url)
        # Patch httpx for URL modification
        patch_httpx()

    # Determine which providers to instrument
    provider_names = list(providers) if providers else ["openai"]

    # Instrument each provider
    for name in provider_names:
        if name.lower() == "openai":
            instrument_openai()
        else:
            raise ValueError(f"Unknown provider: {name}. Available: ['openai']")

    _instrumented = True


def uninstrument(providers: Sequence[str] | None = None) -> None:
    """Disable auto-instrumentation for LLM providers.

    Args:
        providers: List of providers to uninstrument. If None, uninstruments all.
    """
    global _instrumented

    if not _instrumented:
        return

    provider_names = list(providers) if providers else ["openai"]

    for name in provider_names:
        if name.lower() == "openai":
            uninstrument_openai()

    # Clear proxy URLs and unpatch httpx
    clear_proxy_urls()
    unpatch_httpx()

    _instrumented = False


def is_instrumented(provider: str | None = None) -> bool:
    """Check if instrumentation is active.

    Args:
        provider: Specific provider to check. If None, returns True if any active.
    """
    if provider:
        if provider.lower() == "openai":
            return is_openai_instrumented()
        return False
    return _instrumented


__all__ = [
    "instrument",
    "uninstrument",
    "is_instrumented",
    "register_proxy_url",
    "clear_proxy_urls",
]
