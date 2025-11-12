"""Convenient shortcuts for common SDK operations.

This module provides standalone functions for common SDK operations,
making the SDK more ergonomic for simple use cases.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI, OpenAI

from rllm.sdk.chat import ProxyTrackedAsyncChatClient, ProxyTrackedChatClient
from rllm.sdk.session import SessionContext


def _session_with_id(session_id: str | None = None, **metadata: Any) -> SessionContext:
    """Create a session context manager with explicit session_id (INTERNAL USE ONLY).

    This is an internal function that allows setting an explicit session_id.
    For public use, use the `session()` function which auto-generates session_id.

    Args:
        session_id: Explicit session ID (auto-generated if None)
        **metadata: Arbitrary metadata to attach to all traces in this session

    Returns:
        SessionContext: A context manager that sets session_id and metadata
    """
    return SessionContext(session_id=session_id, **metadata)


def session(**metadata: Any) -> SessionContext:
    """Create a session context manager for automatic trace tracking.

    This is a standalone convenience function that creates a SessionContext.
    The session_id is automatically generated and cannot be overridden to
    ensure proper session management.

    When nested inside a `_session_with_id()` context, the SessionContext
    will automatically inherit the parent session_id instead of generating
    a new one, allowing internal code to control the session_id while keeping
    it hidden from users.

    Usage:
        from rllm.sdk import session
        with session(...):

    Examples:
        Session with metadata:
        >>> from rllm.sdk import session
        >>> with session(experiment="v1", user="alice"):
        ...     # All traces get auto-generated session_id + custom metadata

        Multiple sessions:
        >>> with session(experiment="v1"):
        ...     # session_id auto-generated, metadata preserved
        ...     pass

        Nested sessions (metadata inheritance):
        >>> with session(experiment="v1"):
        ...     with session(task="math"):
        ...         # Inherits experiment="v1", adds task="math"

        Nested with _session_with_id (internal use):
        >>> from rllm.sdk.shortcuts import _session_with_id
        >>> with _session_with_id(session_id="internal-id"):
        ...     with session(experiment="v1"):
        ...         # Uses "internal-id" as session_id, adds experiment="v1"

    Args:
        **metadata: Arbitrary metadata to attach to all traces in this session.
                   Note: session_id is auto-generated and cannot be specified.

    Returns:
        SessionContext: A context manager that sets session_id and metadata

    Note:
        For internal use with explicit session_id, use `_session_with_id()` instead.
    """
    assert "session_id" not in metadata, "session_id is auto-generated and cannot be specified"
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
    **kwargs: Any,
):
    """Get a chat client.

    This is a standalone convenience function that creates a chat client
    directly. Usage:
        from rllm.sdk import get_chat_client
        llm = get_chat_client(api_key=api_key, model=model)

    The returned client (ProxyTrackedChatClient) automatically tracks session
    context when used within a session() context manager.

    Examples:
        Basic usage:
        >>> from rllm.sdk import get_chat_client
        >>> llm = get_chat_client(api_key="sk-...", model="gpt-4")
        >>> response = llm.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )

        With session context:
        >>> from rllm.sdk import get_chat_client, session
        >>> llm = get_chat_client(api_key="sk-...", model="gpt-4")
        >>> with session("my-session", experiment="v1"):
        ...     response = llm.chat.completions.create(
        ...         messages=[{"role": "user", "content": "Hello"}]
        ...     )

        With proxy (base_url):
        >>> llm = get_chat_client(
        ...     api_key="sk-...",
        ...     base_url="http://localhost:8000/v1",
        ...     model="gpt-4"
        ... )

        Using environment variables:
        >>> # OPENAI_API_KEY environment variable is used automatically
        >>> llm = get_chat_client(model="gpt-4")

    Args:
        provider: Provider name (currently only "openai" is supported)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: Base URL for API requests (for proxy or custom endpoints)
        model: Default model to use for completions
        organization: OpenAI organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        **kwargs: Additional arguments passed to the OpenAI client

    Returns:
        ProxyTrackedChatClient with the same interface as OpenAI's client,
        plus automatic session tracking.

    Raises:
        ValueError: If provider is not "openai" or if api_key is not provided
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

    # Wrap with proxy tracked client (default)
    wrapper = ProxyTrackedChatClient(
        tracer=None,  # disable SDK-side logging; proxy handles tracing
        default_model=model,
        base_url=base_url,
        client=client,
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
    **kwargs: Any,
):
    """Get an async chat client.

    This is the async version of get_chat_client(). See get_chat_client()
    for detailed documentation.

    Examples:
        Basic usage:
        >>> from rllm.sdk import get_chat_client_async
        >>> llm = get_chat_client_async(api_key="sk-...", model="gpt-4")
        >>> response = await llm.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )

        With session context:
        >>> from rllm.sdk import get_chat_client_async, session
        >>> llm = get_chat_client_async(api_key="sk-...", model="gpt-4")
        >>> with session("my-session", experiment="v1"):
        ...     response = await llm.chat.completions.create(
        ...         messages=[{"role": "user", "content": "Hello"}]
        ...     )

    Args:
        provider: Provider name (currently only "openai" is supported)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: Base URL for API requests (for proxy or custom endpoints)
        model: Default model to use for completions
        organization: OpenAI organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries for failed requests
        **kwargs: Additional arguments passed to the AsyncOpenAI client

    Returns:
        ProxyTrackedAsyncChatClient with the same interface as OpenAI's async client,
        plus automatic session tracking.

    Raises:
        ValueError: If provider is not "openai" or if api_key is not provided
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

    # Wrap with proxy tracked client (default)
    wrapper = ProxyTrackedAsyncChatClient(
        tracer=None,  # disable SDK-side logging; proxy handles tracing
        default_model=model,
        base_url=base_url,
        client=client,
    )

    return wrapper
