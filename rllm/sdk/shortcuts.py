"""Convenient shortcuts for common SDK operations.

This module provides standalone functions for common SDK operations,
making the SDK more ergonomic for simple use cases.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI

from rllm.sdk.chat import (
    OpenTelemetryTrackedAsyncChatClient,
    OpenTelemetryTrackedChatClient,
    ProxyTrackedAsyncChatClient,
    ProxyTrackedChatClient,
)
from rllm.sdk.session import SESSION_BACKEND, SessionContext, otel_session


def _session_with_name(name: str | None = None, **metadata: Any):
    """Create a session context manager with explicit name (INTERNAL USE ONLY).

    This is an internal function that allows setting an explicit session name.
    For public use, use the `session()` function which auto-generates the name.

    Args:
        name: Explicit session name (auto-generated if None)
        **metadata: Arbitrary metadata to attach to all traces in this session

    Returns:
        SessionContext: A context manager that sets session name and metadata
    """
    if SESSION_BACKEND == "opentelemetry":
        if otel_session is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        return otel_session(name=name, **metadata)
    return SessionContext(name=name, **metadata)


def session(**metadata: Any):
    """Create session context for automatic trace tracking with auto-generated name.

    Session name is auto-generated. Nested sessions inherit parent metadata.
    For internal use with explicit names, use _session_with_name() instead.

    Args:
        **metadata: Metadata attached to all traces in this session.

    Returns:
        SessionContext: Context manager for session and metadata.

    Example:
        >>> with session(experiment="v1"):
        ...     llm.chat.completions.create(...)  # Traces get metadata
    """
    assert "name" not in metadata, "name is auto-generated and cannot be specified"
    if SESSION_BACKEND == "opentelemetry":
        if otel_session is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        return otel_session(**metadata)
    return SessionContext(**metadata)


def get_chat_client(
    provider: str = "openai",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    use_proxy: bool = True,
    **kwargs: Any,
):
    """Get OpenAI chat client with automatic session tracking.

    Returns ProxyTrackedChatClient that tracks calls within session() contexts.
    API key from parameter or OPENAI_API_KEY env var.

    Args:
        provider: Provider name (only "openai" supported).
        api_key: OpenAI API key (optional if OPENAI_API_KEY set).
        base_url: Base URL for proxy or custom endpoints.
        model: Default model for completions.
        organization: OpenAI organization ID.
        timeout: Request timeout in seconds.
        max_retries: Max retries for failed requests.
        use_proxy: Enable proxy features (default: True).
        **kwargs: Additional OpenAI client arguments.

    Returns:
        ProxyTrackedChatClient: OpenAI client with session tracking.

    Example:
        >>> llm = get_chat_client(api_key="sk-...", model="gpt-4")
        >>> with session(experiment="v1"):
        ...     llm.chat.completions.create(messages=[...])
    """
    if provider.lower() != "openai":
        raise ValueError(f"Unsupported chat provider '{provider}'. Only 'openai' is supported.")

    # Get API key from parameter or environment
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key is required. Provide api_key=... or set OPENAI_API_KEY environment variable.")

    # Build OpenAI client kwargs
    openai_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if base_url is not None:
        openai_kwargs["base_url"] = base_url
    if organization is not None:
        openai_kwargs["organization"] = organization
    if timeout is not None:
        openai_kwargs["timeout"] = timeout
    if max_retries is not None:
        openai_kwargs["max_retries"] = max_retries
    openai_kwargs.update(kwargs)

    # Create OpenAI client
    client = OpenAI(**openai_kwargs)

    # Wrap with backend-routed client
    # When use_proxy=True, injects metadata slugs for proxy routing
    # OTel backend uses OpenTelemetry-aware client; ContextVar uses proxy client
    if SESSION_BACKEND == "opentelemetry":
        if OpenTelemetryTrackedChatClient is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        wrapper = OpenTelemetryTrackedChatClient(
            api_key=resolved_api_key,
            base_url=base_url,
            default_model=model,
            client=client,
            use_proxy=use_proxy,
        )
    else:
        wrapper = ProxyTrackedChatClient(
            tracer=None,
            default_model=model,
            base_url=base_url,
            client=client,
            use_proxy=use_proxy,
        )

    return wrapper


def get_chat_client_async(
    provider: str = "openai",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    organization: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    use_proxy: bool = True,
    **kwargs: Any,
):
    """Get async OpenAI chat client with automatic session tracking.

    Async version of get_chat_client(). See get_chat_client() for details.

    Returns:
        ProxyTrackedAsyncChatClient: Async OpenAI client with session tracking.

    Example:
        >>> llm = get_chat_client_async(api_key="sk-...", model="gpt-4")
        >>> with session(experiment="v1"):
        ...     await llm.chat.completions.create(messages=[...])
    """
    if provider.lower() != "openai":
        raise ValueError(f"Unsupported chat provider '{provider}'. Only 'openai' is supported.")

    # Get API key from parameter or environment
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key is required. Provide api_key=... or set OPENAI_API_KEY environment variable.")

    # Build AsyncOpenAI client kwargs
    openai_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if base_url is not None:
        openai_kwargs["base_url"] = base_url
    if organization is not None:
        openai_kwargs["organization"] = organization
    if timeout is not None:
        openai_kwargs["timeout"] = timeout
    if max_retries is not None:
        openai_kwargs["max_retries"] = max_retries
    openai_kwargs.update(kwargs)

    # Create AsyncOpenAI client
    client = AsyncOpenAI(**openai_kwargs)

    # Wrap with backend-routed client
    if SESSION_BACKEND == "opentelemetry":
        if OpenTelemetryTrackedAsyncChatClient is None:
            raise RuntimeError("OpenTelemetry backend requested but opentelemetry package not installed")
        wrapper = OpenTelemetryTrackedAsyncChatClient(
            api_key=resolved_api_key,
            base_url=base_url,
            default_model=model,
            client=client,
            use_proxy=use_proxy,
        )
    else:
        wrapper = ProxyTrackedAsyncChatClient(
            tracer=None,
            default_model=model,
            base_url=base_url,
            use_proxy=use_proxy,
            client=client,
        )

    return wrapper
