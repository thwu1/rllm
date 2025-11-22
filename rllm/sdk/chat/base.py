"""Base classes and utilities for chat client implementations.

This module provides shared functionality for all chat client variants:
- SimpleTrackedChatClient (direct tracing)
- ProxyTrackedChatClient (proxy-based tracing)
- OpenTelemetryTrackedChatClient (OTel-based tracing)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

if TYPE_CHECKING:
    pass

# Type variables for generic client types
SyncClientT = TypeVar("SyncClientT", bound=OpenAI)
AsyncClientT = TypeVar("AsyncClientT", bound=AsyncOpenAI)


# =============================================================================
# Shared Utility Functions
# =============================================================================


def merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword arguments into a single dict.

    Supports passing a single dict as the first positional argument,
    which will be merged with keyword arguments.

    Args:
        args: Positional arguments (at most one dict allowed)
        kwargs: Keyword arguments

    Returns:
        Merged dictionary of all arguments

    Raises:
        TypeError: If positional arguments are not supported format
    """
    if args:
        if len(args) == 1 and isinstance(args[0], Mapping):
            merged = dict(args[0])
            merged.update(kwargs)
            return merged
        raise TypeError("Positional arguments are not supported; pass keyword arguments.")
    return dict(kwargs)


def extract_completion_tokens(response_payload: Mapping[str, Any]) -> list[int] | None:
    """Extract completion token IDs from response payload.

    Looks for token IDs in various locations within the response:
    1. choice.output_token_ids (some providers)
    2. choice.logprobs.token_ids (vLLM and others)

    Args:
        response_payload: The response dict from the LLM API

    Returns:
        List of token IDs if found, None otherwise
    """
    choices = response_payload.get("choices") or []
    if not choices:
        return None
    choice0 = choices[0]
    output_ids = choice0.get("output_token_ids")
    if isinstance(output_ids, list):
        return [int(tok) for tok in output_ids]
    logprobs = choice0.get("logprobs")
    if isinstance(logprobs, Mapping):
        token_ids = logprobs.get("token_ids")
        if isinstance(token_ids, list):
            return [int(tok) for tok in token_ids]
    return None


def extract_usage_tokens(response_payload: Mapping[str, Any]) -> dict[str, int]:
    """Extract token usage statistics from response payload.

    Args:
        response_payload: The response dict from the LLM API

    Returns:
        Dict with prompt, completion, and total token counts
    """
    usage = response_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens,
    }


# =============================================================================
# Base Chat Client Classes
# =============================================================================


class BaseChatClient(ABC, Generic[SyncClientT]):
    """Abstract base class for synchronous chat clients.

    Provides common initialization and structure for all sync chat client variants.
    Subclasses must implement _create_completions_namespace() and _create_chat_namespace().
    """

    _client: SyncClientT
    default_model: str | None
    base_url: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: SyncClientT | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the chat client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            default_model: Default model to use for completions
            client: Pre-configured OpenAI client instance
            **client_kwargs: Additional kwargs passed to OpenAI client
        """
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = OpenAI(**init_kwargs)  # type: ignore

        self.default_model = default_model
        self.base_url = base_url

        # Set up namespaces - subclasses provide the implementation
        self.chat = self._create_chat_namespace()
        self.completions = self._create_completions_namespace()

    @abstractmethod
    def _create_chat_namespace(self) -> Any:
        """Create the chat namespace object."""
        ...

    @abstractmethod
    def _create_completions_namespace(self) -> Any:
        """Create the completions namespace object."""
        ...


class BaseAsyncChatClient(ABC, Generic[AsyncClientT]):
    """Abstract base class for asynchronous chat clients.

    Provides common initialization and structure for all async chat client variants.
    Subclasses must implement _create_completions_namespace() and _create_chat_namespace().
    """

    _client: AsyncClientT
    default_model: str | None
    base_url: str | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        client: AsyncClientT | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initialize the async chat client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            default_model: Default model to use for completions
            client: Pre-configured AsyncOpenAI client instance
            **client_kwargs: Additional kwargs passed to AsyncOpenAI client
        """
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**init_kwargs)  # type: ignore

        self.default_model = default_model
        self.base_url = base_url

        # Set up namespaces - subclasses provide the implementation
        self.chat = self._create_chat_namespace()
        self.completions = self._create_completions_namespace()

    @abstractmethod
    def _create_chat_namespace(self) -> Any:
        """Create the chat namespace object."""
        ...

    @abstractmethod
    def _create_completions_namespace(self) -> Any:
        """Create the completions namespace object."""
        ...


# =============================================================================
# Namespace Dataclasses - Generic implementations
# =============================================================================


@dataclass
class ChatCompletionsBase:
    """Base class for chat.completions namespace."""

    parent: Any

    def _validate_and_prepare(self, args: tuple, kwargs: dict) -> tuple[dict, str, dict]:
        """Validate inputs and prepare call kwargs.

        Returns:
            Tuple of (call_kwargs, model, metadata)
        """
        call_kwargs = merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        return call_kwargs, model, metadata


@dataclass
class CompletionsBase:
    """Base class for completions namespace."""

    parent: Any

    def _validate_and_prepare(self, args: tuple, kwargs: dict) -> tuple[dict, str, dict]:
        """Validate inputs and prepare call kwargs.

        Returns:
            Tuple of (call_kwargs, model, metadata)
        """
        call_kwargs = merge_args(args, kwargs)
        prompt = call_kwargs.get("prompt")
        if prompt is None:
            raise ValueError("prompt must be provided for completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}
        return call_kwargs, model, metadata


@dataclass
class ChatNamespaceBase:
    """Base class for chat namespace that provides completions property."""

    parent: Any
    _completions_class: type

    @property
    def completions(self):
        """Return the completions namespace."""
        return self._completions_class(self.parent)


# =============================================================================
# Timing Helper
# =============================================================================


class TimedCall:
    """Context manager for timing API calls."""

    def __init__(self):
        self.start_time: float = 0
        self.latency_ms: float = 0

    def __enter__(self) -> "TimedCall":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.latency_ms = (time.perf_counter() - self.start_time) * 1000
