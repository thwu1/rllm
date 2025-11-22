# Design Doc: Auto-Instrumentation for rLLM SDK

**Status:** Proposal
**Author:** Claude
**Date:** 2025-11-22

## Overview

This document proposes a design for **auto-instrumentation** in the rLLM SDK, allowing users to use standard LLM clients (OpenAI, Anthropic, etc.) without needing custom wrapper clients, while still automatically capturing LLM traces within session contexts.

## Problem Statement

### Current State

Users must use special wrapper clients to get automatic trace capture:

```python
from rllm.sdk import get_chat_client_async, session

# Must use rLLM's wrapped client
client = get_chat_client_async(base_url="...", api_key="...")

with session(agent="solver") as sess:
    response = await client.chat.completions.create(messages=[...])
    print(sess.llm_calls)  # Works!
```

### Issues

1. **Friction**: Users must replace their existing clients with rLLM wrappers
2. **Multiple Clients**: We have 4+ client variants:
   - `ProxyTrackedChatClient` / `ProxyTrackedAsyncChatClient`
   - `OpenTelemetryTrackedChatClient` / `OpenTelemetryTrackedAsyncChatClient`
   - `SimpleTrackedChatClient` / `SimpleTrackedAsyncChatClient`
3. **Maintenance**: Each new LLM provider requires a new wrapper implementation
4. **Complexity**: Backend-specific routing logic (`SESSION_BACKEND`) scattered across code

### Desired State

Users can use **standard clients** with zero changes:

```python
from openai import AsyncOpenAI
from rllm.sdk import session, instrument

# One-time setup (or via engine)
instrument()

# Use standard OpenAI client - unchanged!
client = AsyncOpenAI()

with session(agent="solver") as sess:
    response = await client.chat.completions.create(messages=[...])
    print(sess.llm_calls)  # Still works!
```

## Design Goals

1. **Zero Code Changes**: Standard LLM clients work with no modifications
2. **Unified API**: Single `instrument()` call handles all providers
3. **Simplicity**: Remove redundant wrapper classes
4. **Extensibility**: Easy to add new providers
5. **Backward Compatibility**: Existing wrapper APIs continue to work

## Proposed Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Code                                  │
│  ───────────────────────────────────────────────────────────── │
│  from openai import AsyncOpenAI                                │
│  from anthropic import Anthropic                               │
│  from rllm.sdk import session, instrument                      │
│                                                                 │
│  instrument()  # One-time setup                                │
│                                                                 │
│  with session(name="agent"):                                   │
│      openai_client.chat.completions.create(...)                │
│      anthropic_client.messages.create(...)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Instrumentation Layer                          │
│  ───────────────────────────────────────────────────────────── │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ OpenAI Provider │ │Anthropic Provider│ │ Other Providers │   │
│  │  - wrap create  │ │  - wrap create  │ │  - wrap create  │   │
│  │  - read baggage │ │  - read baggage │ │  - read baggage │   │
│  │  - store trace  │ │  - store trace  │ │  - store trace  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Session Context                              │
│  ───────────────────────────────────────────────────────────── │
│  - W3C Baggage (session_uids, metadata)                        │
│  - SqliteTraceStore (persistent storage)                       │
│  - session.llm_calls → query traces by session_uid             │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. `InstrumentationProvider` Protocol

```python
# rllm/sdk/instrumentation/base.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class InstrumentationProvider(Protocol):
    """Protocol for LLM provider instrumentation."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    def instrument(self) -> None:
        """Apply monkey-patches to the provider library."""
        ...

    def uninstrument(self) -> None:
        """Remove monkey-patches from the provider library."""
        ...

    def is_instrumented(self) -> bool:
        """Check if provider is currently instrumented."""
        ...
```

#### 2. `OpenAIInstrumentationProvider`

```python
# rllm/sdk/instrumentation/openai_provider.py

import functools
import time
from typing import Any, Callable

from rllm.sdk.instrumentation.base import InstrumentationProvider
from rllm.sdk.instrumentation.trace_capture import capture_trace


class OpenAIInstrumentationProvider(InstrumentationProvider):
    """Instrumentation for OpenAI Python SDK."""

    def __init__(self):
        self._original_sync_create: Callable | None = None
        self._original_async_create: Callable | None = None
        self._instrumented = False

    @property
    def name(self) -> str:
        return "openai"

    def instrument(self) -> None:
        if self._instrumented:
            return

        try:
            from openai.resources.chat.completions import (
                AsyncCompletions,
                Completions,
            )
        except ImportError:
            return  # OpenAI not installed, skip

        # Store originals
        self._original_sync_create = Completions.create
        self._original_async_create = AsyncCompletions.create

        # Apply patches
        Completions.create = self._wrap_sync(self._original_sync_create)
        AsyncCompletions.create = self._wrap_async(self._original_async_create)

        self._instrumented = True

    def uninstrument(self) -> None:
        if not self._instrumented:
            return

        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )

        if self._original_sync_create:
            Completions.create = self._original_sync_create
        if self._original_async_create:
            AsyncCompletions.create = self._original_async_create

        self._instrumented = False

    def is_instrumented(self) -> bool:
        return self._instrumented

    def _wrap_sync(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapped(self_client, *args, **kwargs):
            start = time.perf_counter()
            response = original(self_client, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            capture_trace(
                provider="openai",
                method="chat.completions.create",
                input_data={"messages": kwargs.get("messages", [])},
                output_data=_extract_openai_response(response),
                model=kwargs.get("model", "unknown"),
                latency_ms=latency_ms,
                tokens=_extract_openai_tokens(response),
            )

            return response
        return wrapped

    def _wrap_async(self, original: Callable) -> Callable:
        @functools.wraps(original)
        async def wrapped(self_client, *args, **kwargs):
            start = time.perf_counter()
            response = await original(self_client, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            capture_trace(
                provider="openai",
                method="chat.completions.create",
                input_data={"messages": kwargs.get("messages", [])},
                output_data=_extract_openai_response(response),
                model=kwargs.get("model", "unknown"),
                latency_ms=latency_ms,
                tokens=_extract_openai_tokens(response),
            )

            return response
        return wrapped


def _extract_openai_response(response) -> dict:
    """Extract response content from OpenAI response object."""
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            return {
                "role": choice.message.role,
                "content": choice.message.content,
            }
    return {}


def _extract_openai_tokens(response) -> dict:
    """Extract token usage from OpenAI response object."""
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }
    return {}
```

#### 3. `capture_trace` - Unified Trace Capture

```python
# rllm/sdk/instrumentation/trace_capture.py

import time
import uuid
from typing import Any

from rllm.sdk.protocol import Trace
from rllm.sdk.session.opentelemetry import (
    get_active_otel_session_uids,
    get_current_otel_metadata,
    get_current_otel_session_name,
)


# Global store reference (set by instrument())
_trace_store = None


def set_trace_store(store) -> None:
    """Set the global trace store for instrumentation."""
    global _trace_store
    _trace_store = store


def get_trace_store():
    """Get the global trace store."""
    global _trace_store
    if _trace_store is None:
        from rllm.sdk.store import SqliteTraceStore
        _trace_store = SqliteTraceStore()
    return _trace_store


def capture_trace(
    provider: str,
    method: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    model: str,
    latency_ms: float,
    tokens: dict[str, int] | None = None,
) -> None:
    """
    Capture an LLM trace if we're inside a session context.

    This function reads session context from W3C baggage (single source of truth)
    and stores the trace to SQLite with session UIDs.

    Args:
        provider: LLM provider name (e.g., 'openai', 'anthropic')
        method: API method called (e.g., 'chat.completions.create')
        input_data: Input to the LLM call
        output_data: Output from the LLM call
        model: Model name used
        latency_ms: Call latency in milliseconds
        tokens: Token usage dict (optional)
    """
    # Read session context from baggage
    session_uids = get_active_otel_session_uids()

    # Only capture if we're inside a session context
    if not session_uids:
        return

    session_name = get_current_otel_session_name()
    session_metadata = get_current_otel_metadata()

    # Build trace
    trace = Trace(
        trace_id=str(uuid.uuid4()),
        session_name=session_name or "unknown",
        name=f"{provider}/{method}",
        input=input_data,
        output=output_data,
        model=model,
        latency_ms=latency_ms,
        tokens=tokens or {},
        metadata={
            **session_metadata,
            "provider": provider,
            "session_uids": session_uids,
        },
        timestamp=time.time(),
    )

    # Store trace asynchronously
    _store_trace_async(trace, session_uids)


def _store_trace_async(trace: Trace, session_uids: list[str]) -> None:
    """Store trace to SQLite (fire-and-forget)."""
    import asyncio

    store = get_trace_store()

    async def _store():
        await store.store(
            trace_id=trace.trace_id,
            data=trace.model_dump(),
            session_uids=session_uids,
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_store())
    except RuntimeError:
        # No running loop - run synchronously
        asyncio.run(_store())
```

#### 4. `instrument()` - Unified Entry Point

```python
# rllm/sdk/instrumentation/__init__.py

from typing import Sequence

from rllm.sdk.instrumentation.base import InstrumentationProvider
from rllm.sdk.instrumentation.openai_provider import OpenAIInstrumentationProvider
from rllm.sdk.instrumentation.trace_capture import set_trace_store

# Registry of available providers
_PROVIDERS: dict[str, type[InstrumentationProvider]] = {
    "openai": OpenAIInstrumentationProvider,
    # "anthropic": AnthropicInstrumentationProvider,  # Future
    # "together": TogetherInstrumentationProvider,    # Future
}

# Active provider instances
_active_providers: dict[str, InstrumentationProvider] = {}


def instrument(
    providers: Sequence[str] | None = None,
    store = None,
) -> None:
    """
    Enable auto-instrumentation for LLM providers.

    After calling this, any LLM call within a session() context
    will be automatically traced and stored.

    Args:
        providers: List of providers to instrument. If None, instruments all available.
                  Options: "openai", "anthropic", "together"
        store: SqliteTraceStore instance. If None, uses default.

    Example:
        >>> from rllm.sdk import instrument, session
        >>> from openai import AsyncOpenAI
        >>>
        >>> # Enable instrumentation (call once at startup)
        >>> instrument()
        >>>
        >>> # Use standard clients - traces are automatic!
        >>> client = AsyncOpenAI()
        >>> with session(name="my-agent") as sess:
        ...     await client.chat.completions.create(...)
        ...     print(sess.llm_calls)  # Captured automatically!
    """
    # Set trace store
    if store is not None:
        set_trace_store(store)

    # Determine which providers to instrument
    provider_names = providers or list(_PROVIDERS.keys())

    for name in provider_names:
        if name in _active_providers:
            continue  # Already instrumented

        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}")

        provider = provider_cls()
        provider.instrument()
        _active_providers[name] = provider


def uninstrument(providers: Sequence[str] | None = None) -> None:
    """
    Disable auto-instrumentation for LLM providers.

    Args:
        providers: List of providers to uninstrument. If None, uninstruments all.
    """
    provider_names = providers or list(_active_providers.keys())

    for name in list(provider_names):
        provider = _active_providers.get(name)
        if provider:
            provider.uninstrument()
            del _active_providers[name]


def is_instrumented(provider: str | None = None) -> bool:
    """
    Check if instrumentation is active.

    Args:
        provider: Specific provider to check. If None, returns True if any active.
    """
    if provider:
        return provider in _active_providers
    return len(_active_providers) > 0
```

### File Structure

```
rllm/sdk/
├── __init__.py                    # Expose instrument(), session, etc.
├── instrumentation/
│   ├── __init__.py                # instrument(), uninstrument()
│   ├── base.py                    # InstrumentationProvider protocol
│   ├── trace_capture.py           # capture_trace() - unified trace storage
│   ├── openai_provider.py         # OpenAI instrumentation
│   ├── anthropic_provider.py      # Anthropic instrumentation (future)
│   └── together_provider.py       # Together instrumentation (future)
├── session/
│   ├── __init__.py
│   └── opentelemetry.py           # otel_session, baggage helpers
├── store/
│   ├── __init__.py
│   └── sqlite_store.py            # SqliteTraceStore
├── protocol.py                    # Trace, TrajectoryView, etc.
└── shortcuts.py                   # session(), get_chat_client() (backward compat)
```

## Migration Path

### Phase 1: Add Instrumentation (Non-Breaking)

1. Add `rllm/sdk/instrumentation/` module
2. Add `instrument()` to `rllm.sdk` exports
3. Existing wrapper clients continue to work

### Phase 2: Deprecate Wrapper Clients

1. Mark `get_chat_client()` as deprecated
2. Update examples to use `instrument()` + standard clients
3. Wrapper clients remain functional but emit deprecation warnings

### Phase 3: Simplify (Major Version)

1. Remove wrapper client classes
2. Remove `get_chat_client()` and `get_chat_client_async()`
3. `instrument()` is the only way to enable tracing

## Usage Examples

### Basic Usage

```python
from openai import AsyncOpenAI
from rllm.sdk import instrument, session

# One-time setup
instrument()

# Standard OpenAI client
client = AsyncOpenAI()

async def solve(problem: str):
    with session(name="solver") as sess:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": problem}]
        )

        # Traces automatically captured!
        print(f"Captured {len(sess.llm_calls)} traces")
        return response.choices[0].message.content
```

### With AgentSDKEngine

```python
from rllm.sdk import instrument
from rllm.engine import AgentSDKEngine

class MyEngine(AgentSDKEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-instrument when engine starts
        instrument()

    async def run_rollout(self, task):
        with session(name="rollout") as sess:
            result = await self.agent.solve(task)
            return result, sess.llm_calls
```

### Multiple Providers

```python
from openai import OpenAI
from anthropic import Anthropic
from rllm.sdk import instrument, session

# Instrument specific providers
instrument(providers=["openai", "anthropic"])

openai_client = OpenAI()
anthropic_client = Anthropic()

with session(name="multi-provider") as sess:
    # Both calls are traced
    openai_client.chat.completions.create(...)
    anthropic_client.messages.create(...)

    print(sess.llm_calls)  # Contains both traces
```

### Backward Compatible

```python
# Old code still works!
from rllm.sdk import get_chat_client_async, session

client = get_chat_client_async(...)  # Still works, but deprecated

with session(name="agent") as sess:
    await client.chat.completions.create(...)
```

## Comparison: Before vs After

### Before (Current)

```python
# Multiple wrapper classes to maintain
class ProxyTrackedChatClient: ...
class ProxyTrackedAsyncChatClient: ...
class OpenTelemetryTrackedChatClient: ...
class OpenTelemetryTrackedAsyncChatClient: ...
class SimpleTrackedChatClient: ...
class SimpleTrackedAsyncChatClient: ...

# Backend-specific routing scattered across code
if SESSION_BACKEND == "opentelemetry":
    wrapper = OpenTelemetryTrackedChatClient(...)
else:
    wrapper = ProxyTrackedChatClient(...)

# User must use wrapper
client = get_chat_client_async(...)
```

### After (Proposed)

```python
# Single instrumentation entry point
instrument()

# User uses standard client
from openai import AsyncOpenAI
client = AsyncOpenAI()

# Everything else works automatically
with session(name="agent") as sess:
    await client.chat.completions.create(...)
    print(sess.llm_calls)  # Just works!
```

## Trade-offs

### Pros

1. **Simpler User Experience**: No wrapper classes to learn
2. **Less Code**: Remove 6+ wrapper class implementations
3. **Extensible**: Add new providers by implementing one class
4. **Standard Clients**: Users use official SDKs with full features
5. **Consistent**: Same pattern regardless of provider

### Cons

1. **Monkey-Patching**: Can be fragile if libraries change internal structure
2. **Global State**: `instrument()` affects all clients in the process
3. **Debugging**: Stack traces include wrapper code
4. **Version Coupling**: Must track SDK version changes

### Mitigations

1. **Version Pinning**: Test against specific SDK versions
2. **Graceful Fallback**: If patching fails, log warning and continue
3. **Uninstrument**: Provide `uninstrument()` for testing/debugging
4. **Provider Isolation**: Each provider patches independently

## Critical Limitation: Proxy Mode

### The Problem

The current `ProxyTrackedChatClient` provides a **proxy mode** that modifies the request URL
to include session metadata:

```python
# Original URL
base_url = "http://proxy:4000/v1"

# Modified URL (with metadata slug)
proxied_url = "http://proxy:4000/meta/rllm1:eyJzZXNzaW9uX25hbWUiOiJzb2x2ZXIifQ/v1"
```

The proxy server extracts this metadata from the URL path to:
1. Associate traces with sessions
2. Route requests appropriately
3. Return token IDs for RL training

**Auto-instrumentation CANNOT reproduce this** because:
- Monkey-patching intercepts calls AFTER the URL is already set
- We cannot modify the client's `base_url` from within the wrapper
- The request goes to the original URL, not the proxied URL

### Features Comparison

| Feature | Wrapper Client | Auto-Instrumentation |
|---------|---------------|---------------------|
| Trace capture | ✅ | ✅ |
| Session context from baggage | ✅ | ✅ |
| Token extraction | ✅ | ✅ |
| In-memory tracing | ✅ | ✅ |
| **Proxy URL modification** | ✅ | ❌ |
| **Per-call metadata override** | ✅ | ❌ |
| **Default model fallback** | ✅ | ❌ |

### Recommended Approach: Hybrid

Given this limitation, we recommend a **hybrid approach**:

```python
# Mode 1: Direct mode (vLLM, SGLang, etc.) - Use auto-instrumentation
from rllm.sdk import instrument, session
from openai import AsyncOpenAI

instrument()  # Enable auto-instrumentation

client = AsyncOpenAI(base_url="http://vllm:8000/v1")

with session(name="agent") as sess:
    await client.chat.completions.create(...)  # Automatically traced!


# Mode 2: Proxy mode - Use wrapper client (required for URL modification)
from rllm.sdk import get_chat_client_async, session

client = get_chat_client_async(
    base_url="http://proxy:4000/v1",
    use_proxy=True,  # Enables URL modification
)

with session(name="agent") as sess:
    await client.chat.completions.create(...)  # URL modified for proxy
```

### Solution: Patch httpx Transport Layer

Since OpenAI SDK (and many other LLM SDKs) use `httpx` internally, we can patch at the
httpx transport level to modify URLs **before** requests are sent.

#### How It Works

```
User Code → OpenAI SDK → httpx Client → Patched Transport → HTTP Request
                                              ↓
                                    Modify URL if in session context
```

#### Implementation

```python
# rllm/sdk/instrumentation/httpx_transport.py

import httpx
from rllm.sdk.session.opentelemetry import get_active_otel_session_uids, get_current_otel_metadata
from rllm.sdk.proxy.metadata_slug import encode_metadata_slug

# Store original transport classes
_original_async_transport_init = None
_original_sync_transport_init = None
_instrumented = False

# Configurable: which base URLs should have metadata injected
_proxy_base_urls: set[str] = set()


def register_proxy_url(base_url: str) -> None:
    """Register a base URL that should have session metadata injected."""
    _proxy_base_urls.add(base_url.rstrip("/"))


def _should_inject_metadata(url: httpx.URL) -> bool:
    """Check if this URL should have metadata injected."""
    base = f"{url.scheme}://{url.host}:{url.port}" if url.port else f"{url.scheme}://{url.host}"
    return base in _proxy_base_urls or any(base.startswith(p) for p in _proxy_base_urls)


def _inject_metadata_into_url(url: httpx.URL) -> httpx.URL:
    """Inject session metadata into URL path."""
    session_uids = get_active_otel_session_uids()
    if not session_uids:
        return url  # No session context, return unchanged

    metadata = get_current_otel_metadata()
    metadata["session_uids"] = session_uids

    slug = encode_metadata_slug(metadata)

    # Transform: /v1/chat/completions → /meta/{slug}/v1/chat/completions
    path = url.path
    if "/v1" in path:
        # Insert before /v1
        idx = path.index("/v1")
        new_path = f"{path[:idx]}/meta/{slug}{path[idx:]}"
    else:
        # Insert at beginning
        new_path = f"/meta/{slug}{path}"

    return url.copy_with(path=new_path)


class SessionAwareTransport(httpx.AsyncHTTPTransport):
    """Async transport that injects session metadata into URLs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if _should_inject_metadata(request.url):
            new_url = _inject_metadata_into_url(request.url)
            request = httpx.Request(
                method=request.method,
                url=new_url,
                headers=request.headers,
                content=request.content,
            )
        return await super().handle_async_request(request)


class SessionAwareSyncTransport(httpx.HTTPTransport):
    """Sync transport that injects session metadata into URLs."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if _should_inject_metadata(request.url):
            new_url = _inject_metadata_into_url(request.url)
            request = httpx.Request(
                method=request.method,
                url=new_url,
                headers=request.headers,
                content=request.content,
            )
        return super().handle_request(request)


def instrument(proxy_urls: list[str] | None = None) -> None:
    """
    Enable auto-instrumentation with proxy URL modification.

    Args:
        proxy_urls: List of base URLs that should have session metadata injected.
                   If None, no URL modification occurs (trace capture only).

    Example:
        >>> from rllm.sdk import instrument, session
        >>> from openai import AsyncOpenAI
        >>>
        >>> # Enable instrumentation with proxy URL
        >>> instrument(proxy_urls=["http://proxy:4000"])
        >>>
        >>> # Standard OpenAI client - URLs automatically modified!
        >>> client = AsyncOpenAI(base_url="http://proxy:4000/v1")
        >>>
        >>> with session(name="agent") as sess:
        ...     # Request URL becomes: http://proxy:4000/meta/{slug}/v1/chat/completions
        ...     await client.chat.completions.create(...)
    """
    global _instrumented

    if _instrumented:
        return

    # Register proxy URLs
    if proxy_urls:
        for url in proxy_urls:
            register_proxy_url(url)

    # Patch httpx to use our transports
    _patch_httpx()

    _instrumented = True


def _patch_httpx() -> None:
    """Monkey-patch httpx to use session-aware transports."""
    global _original_async_transport_init, _original_sync_transport_init

    # Store originals
    _original_async_client_init = httpx.AsyncClient.__init__
    _original_sync_client_init = httpx.Client.__init__

    def patched_async_init(self, *args, **kwargs):
        # If no transport specified, use our session-aware transport
        if "transport" not in kwargs:
            kwargs["transport"] = SessionAwareTransport()
        _original_async_client_init(self, *args, **kwargs)

    def patched_sync_init(self, *args, **kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = SessionAwareSyncTransport()
        _original_sync_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = patched_async_init
    httpx.Client.__init__ = patched_sync_init
```

#### User Experience

```python
from openai import AsyncOpenAI
from rllm.sdk import instrument, session

# One-time setup: register proxy URL
instrument(proxy_urls=["http://proxy:4000"])

# Standard OpenAI client - no wrapper needed!
client = AsyncOpenAI(base_url="http://proxy:4000/v1", api_key="...")

async def solve(problem: str):
    with session(name="solver") as sess:
        # URL automatically transformed:
        # http://proxy:4000/v1/chat/completions
        # → http://proxy:4000/meta/rllm1:eyJ.../v1/chat/completions
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": problem}]
        )

        # Traces captured automatically
        print(f"Captured {len(sess.llm_calls)} traces")
        return response.choices[0].message.content
```

#### Why This Works

1. **OpenAI SDK uses httpx internally** - so patching httpx affects all OpenAI clients
2. **Transport is the lowest level** - we intercept before the actual HTTP request
3. **URL modification happens at the right time** - before the request is sent
4. **Works with any httpx-based client** - Anthropic, Together, etc.

#### Features Now Supported

| Feature | Auto-Instrumentation |
|---------|---------------------|
| Trace capture | ✅ |
| Session context from baggage | ✅ |
| Token extraction | ✅ |
| In-memory tracing | ✅ |
| **Proxy URL modification** | ✅ |
| Per-call metadata override | ❌ (not needed) |
| Default model fallback | ❌ (use client default) |

## Alternatives Considered

### 1. Keep Wrapper Clients Only

**Rejected**: Too much maintenance, poor user experience

### 2. Use OpenTelemetry Instrumentation Libraries

**Considered**: `opentelemetry-instrumentation-openai` exists, but:
- Requires full OpenTelemetry setup
- Doesn't integrate with our session/baggage system
- Less control over trace format

**Decision**: Build our own lightweight instrumentation that integrates with our session system

### 3. Proxy-Only Approach

**Rejected**: Requires running a proxy server, adds latency, complex setup

## Implementation Plan

### Week 1
- [ ] Implement `InstrumentationProvider` protocol
- [ ] Implement `OpenAIInstrumentationProvider`
- [ ] Implement `capture_trace()` function
- [ ] Add `instrument()` / `uninstrument()` functions

### Week 2
- [ ] Add tests for instrumentation
- [ ] Update examples to use `instrument()`
- [ ] Add deprecation warnings to `get_chat_client()`
- [ ] Write migration guide

### Week 3
- [ ] Add Anthropic provider (if needed)
- [ ] Performance testing
- [ ] Documentation updates

## Open Questions

1. **Streaming Support**: How to handle streaming responses?
   - Proposal: Capture on stream completion, aggregate tokens

2. **Error Handling**: Should failed LLM calls be traced?
   - Proposal: Yes, with error metadata

3. **Concurrent Calls**: Multiple LLM calls in parallel within same session?
   - Already handled: Each call gets unique trace_id, all share session_uids

4. **Memory Management**: Long-running processes with many traces?
   - Already handled: SQLite storage, session-scoped queries

## Conclusion

Auto-instrumentation provides a cleaner, more maintainable approach to LLM tracing in the rLLM SDK. By monkey-patching standard clients at runtime, we eliminate the need for wrapper classes while maintaining full trace capture capabilities within session contexts.

The key insight is that **session context via W3C baggage** is already the source of truth - instrumentation just needs to read from baggage and store traces. This makes the implementation straightforward and the user experience seamless.
