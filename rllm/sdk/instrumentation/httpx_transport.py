"""Session-aware httpx transport for proxy URL modification.

This module provides custom httpx transports that automatically inject
session metadata into request URLs when inside a session context.
This enables auto-instrumentation to work with proxy URL modification.
"""

from __future__ import annotations

import base64
import functools
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

# Metadata slug prefix (same as in metadata_slug.py)
_SLUG_PREFIX = "rllm1:"


def _encode_metadata_slug(metadata: dict[str, Any]) -> str:
    """Encode metadata into a versioned slug suitable for URL embedding.

    This is a local copy of encode_metadata_slug to avoid circular imports
    with the proxy package which has litellm dependencies.
    """
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
    return f"{_SLUG_PREFIX}{encoded.decode('ascii')}"


# Registry of proxy URLs that should have metadata injected
_proxy_urls: set[str] = set()

# Flag to track if httpx is patched
_httpx_patched = False

# Original httpx client init methods
_original_async_client_init = None
_original_sync_client_init = None


def register_proxy_url(base_url: str) -> None:
    """Register a base URL that should have session metadata injected.

    Args:
        base_url: Base URL (e.g., "http://proxy:4000" or "http://proxy:4000/v1")
    """
    # Normalize: strip path and trailing slash
    parsed = urlparse(base_url)
    normalized = f"{parsed.scheme}://{parsed.netloc}"
    _proxy_urls.add(normalized)


def clear_proxy_urls() -> None:
    """Clear all registered proxy URLs."""
    _proxy_urls.clear()


def _should_inject_metadata(url: Any) -> bool:
    """Check if this URL should have metadata injected."""
    if not _proxy_urls:
        return False

    # Handle httpx.URL or string
    url_str = str(url)
    parsed = urlparse(url_str)
    base = f"{parsed.scheme}://{parsed.netloc}"

    return base in _proxy_urls


def _inject_metadata_into_url(url: Any) -> Any:
    """Inject session metadata into URL path.

    Transforms: /v1/chat/completions -> /meta/{slug}/v1/chat/completions
    """
    import httpx

    from rllm.sdk.session import SESSION_BACKEND, get_active_session_uids, get_current_metadata

    # Get session context
    session_uids = get_active_session_uids()
    if not session_uids:
        return url  # No session context, return unchanged

    # Build metadata
    metadata = dict(get_current_metadata())
    metadata["session_uids"] = session_uids

    # Get session name
    if SESSION_BACKEND == "opentelemetry":
        from rllm.sdk.session.opentelemetry import get_current_otel_session_name
        session_name = get_current_otel_session_name()
    else:
        from rllm.sdk.session.contextvar import get_current_cv_session_name
        session_name = get_current_cv_session_name()

    if session_name:
        metadata["session_name"] = session_name

    # Encode metadata to slug (use local function to avoid litellm dependency)
    slug = _encode_metadata_slug(metadata)

    # Handle httpx.URL
    if isinstance(url, httpx.URL):
        path = str(url.path)
        # Insert metadata slug before /v1 if present, otherwise at beginning
        if "/v1" in path:
            idx = path.index("/v1")
            new_path = f"{path[:idx]}/meta/{slug}{path[idx:]}"
        else:
            new_path = f"/meta/{slug}{path}"

        return url.copy_with(raw_path=new_path.encode())

    # Handle string URL
    url_str = str(url)
    parsed = urlparse(url_str)
    path = parsed.path

    if "/v1" in path:
        idx = path.index("/v1")
        new_path = f"{path[:idx]}/meta/{slug}{path[idx:]}"
    else:
        new_path = f"/meta/{slug}{path}"

    return parsed._replace(path=new_path).geturl()


class SessionAwareTransport:
    """Mixin that wraps transport handle methods to inject session metadata."""

    def _wrap_request(self, request: "httpx.Request") -> "httpx.Request":
        """Wrap request to inject metadata if needed."""
        import httpx

        if _should_inject_metadata(request.url):
            new_url = _inject_metadata_into_url(request.url)
            # Create new request with modified URL
            request = httpx.Request(
                method=request.method,
                url=new_url,
                headers=request.headers,
                content=request.content,
                extensions=request.extensions,
            )
        return request


class SessionAwareAsyncTransport(SessionAwareTransport):
    """Async httpx transport that injects session metadata into URLs."""

    def __init__(self, wrapped_transport: "httpx.AsyncBaseTransport"):
        self._wrapped = wrapped_transport

    async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
        request = self._wrap_request(request)
        return await self._wrapped.handle_async_request(request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class SessionAwareSyncTransport(SessionAwareTransport):
    """Sync httpx transport that injects session metadata into URLs."""

    def __init__(self, wrapped_transport: "httpx.BaseTransport"):
        self._wrapped = wrapped_transport

    def handle_request(self, request: "httpx.Request") -> "httpx.Response":
        request = self._wrap_request(request)
        return self._wrapped.handle_request(request)

    def close(self) -> None:
        self._wrapped.close()


def patch_httpx() -> None:
    """Patch httpx client classes to use session-aware transports.

    This wraps httpx.AsyncClient and httpx.Client to automatically use
    our session-aware transports when proxy URLs are registered.
    """
    global _httpx_patched, _original_async_client_init, _original_sync_client_init

    if _httpx_patched:
        return

    try:
        import httpx
    except ImportError:
        return  # httpx not installed

    _original_async_client_init = httpx.AsyncClient.__init__
    _original_sync_client_init = httpx.Client.__init__

    @functools.wraps(_original_async_client_init)
    def patched_async_init(self, *args, **kwargs):
        _original_async_client_init(self, *args, **kwargs)
        # Wrap the transport if proxy URLs are registered
        if _proxy_urls and self._transport is not None:
            self._transport = SessionAwareAsyncTransport(self._transport)

    @functools.wraps(_original_sync_client_init)
    def patched_sync_init(self, *args, **kwargs):
        _original_sync_client_init(self, *args, **kwargs)
        # Wrap the transport if proxy URLs are registered
        if _proxy_urls and self._transport is not None:
            self._transport = SessionAwareSyncTransport(self._transport)

    httpx.AsyncClient.__init__ = patched_async_init
    httpx.Client.__init__ = patched_sync_init

    _httpx_patched = True


def unpatch_httpx() -> None:
    """Remove httpx patches."""
    global _httpx_patched, _original_async_client_init, _original_sync_client_init

    if not _httpx_patched:
        return

    try:
        import httpx
    except ImportError:
        return

    if _original_async_client_init is not None:
        httpx.AsyncClient.__init__ = _original_async_client_init
    if _original_sync_client_init is not None:
        httpx.Client.__init__ = _original_sync_client_init

    _httpx_patched = False
    _original_async_client_init = None
    _original_sync_client_init = None
