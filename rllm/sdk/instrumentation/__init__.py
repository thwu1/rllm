"""Auto-instrumentation for LLM SDKs.

Provides automatic trace capture for LLM clients without requiring custom wrapper clients.

Example:
    >>> from openai import AsyncOpenAI
    >>> from rllm.sdk import instrument, session
    >>>
    >>> instrument(proxy_urls=["http://proxy:4000"])
    >>> client = AsyncOpenAI(base_url="http://proxy:4000/v1", api_key="...")
    >>>
    >>> with session(agent="solver") as sess:
    ...     await client.chat.completions.create(model="gpt-4", messages=[...])
    ...     print(sess.llm_calls)  # Captured automatically!
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

_instrumented = False

SUPPORTED_PROVIDERS = ["openai"]


def instrument(
    providers: Sequence[str] | None = None,
    proxy_urls: Sequence[str] | None = None,
) -> None:
    """Enable auto-instrumentation for LLM providers.

    Args:
        providers: Providers to instrument (default: all supported).
        proxy_urls: Base URLs for proxy metadata injection (optional).
    """
    global _instrumented

    if _instrumented:
        return

    try:
        # Setup proxy URL modification if requested
        if proxy_urls:
            for url in proxy_urls:
                register_proxy_url(url)
            patch_httpx()

        # Instrument providers - only mark as instrumented if at least one succeeds
        any_success = False
        for name in (providers or SUPPORTED_PROVIDERS):
            if name.lower() == "openai":
                if instrument_openai():
                    any_success = True
            else:
                raise ValueError(f"Unknown provider: {name}. Supported: {SUPPORTED_PROVIDERS}")

        _instrumented = any_success
    except Exception:
        # Clean up partial instrumentation on failure
        _cleanup()
        raise


def _cleanup() -> None:
    """Clean up all instrumentation state (internal helper)."""
    uninstrument_openai()
    clear_proxy_urls()
    unpatch_httpx()


def uninstrument(providers: Sequence[str] | None = None) -> None:
    """Disable auto-instrumentation.

    Always cleans up httpx patches and proxy URLs, even if _instrumented is False,
    to handle partial/failed instrumentation cleanup.
    """
    global _instrumented

    for name in (providers or SUPPORTED_PROVIDERS):
        if name.lower() == "openai":
            uninstrument_openai()

    # Always clean up httpx/proxy state to handle partial instrumentation
    clear_proxy_urls()
    unpatch_httpx()
    _instrumented = False


def is_instrumented(provider: str | None = None) -> bool:
    """Check if instrumentation is active."""
    if provider:
        return provider.lower() == "openai" and is_openai_instrumented()
    return _instrumented


__all__ = ["instrument", "uninstrument", "is_instrumented", "register_proxy_url", "clear_proxy_urls"]
