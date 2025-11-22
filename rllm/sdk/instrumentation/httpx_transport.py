"""Session-aware httpx transport for proxy URL modification.

Automatically injects session metadata into request URLs when inside a session context.
"""

from __future__ import annotations

import base64
import functools
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

# State
_proxy_urls: set[str] = set()
_httpx_patched = False
_original_inits: dict[str, Any] = {}


def _encode_slug(metadata: dict[str, Any]) -> str:
    """Encode metadata into a URL-safe slug."""
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(body.encode()).rstrip(b"=")
    return f"rllm1:{encoded.decode()}"


def _insert_slug_in_path(path: str, slug: str) -> str:
    """Insert metadata slug into URL path before /v1 or at start."""
    if "/v1" in path:
        idx = path.index("/v1")
        return f"{path[:idx]}/meta/{slug}{path[idx:]}"
    return f"/meta/{slug}{path}"


def register_proxy_url(base_url: str) -> None:
    """Register a base URL for metadata injection."""
    parsed = urlparse(base_url)
    _proxy_urls.add(f"{parsed.scheme}://{parsed.netloc}")


def clear_proxy_urls() -> None:
    """Clear all registered proxy URLs."""
    _proxy_urls.clear()


def _should_inject(url: Any) -> bool:
    """Check if URL matches a registered proxy URL."""
    if not _proxy_urls:
        return False
    parsed = urlparse(str(url))
    return f"{parsed.scheme}://{parsed.netloc}" in _proxy_urls


def _inject_metadata(url: Any) -> Any:
    """Inject session metadata into URL if inside a session context.

    Preserves query parameters when rewriting URLs.
    """
    import httpx
    from rllm.sdk.session import get_active_session_uids, get_current_metadata, get_current_session_name

    session_uids = get_active_session_uids()
    if not session_uids:
        return url

    metadata = {**get_current_metadata(), "session_uids": session_uids}
    session_name = get_current_session_name()
    if session_name:
        metadata["session_name"] = session_name

    slug = _encode_slug(metadata)

    if isinstance(url, httpx.URL):
        new_path = _insert_slug_in_path(str(url.path), slug)
        # Preserve query string by appending it to the raw path
        query = url.query
        raw_path = new_path.encode() + (b"?" + query if query else b"")
        return url.copy_with(raw_path=raw_path)

    parsed = urlparse(str(url))
    new_path = _insert_slug_in_path(parsed.path, slug)
    return parsed._replace(path=new_path).geturl()


class _TransportWrapper:
    """Base wrapper that injects metadata into requests."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def _modify_request(self, request: "httpx.Request") -> "httpx.Request":
        import httpx
        if _should_inject(request.url):
            return httpx.Request(
                method=request.method,
                url=_inject_metadata(request.url),
                headers=request.headers,
                content=request.content,
                extensions=request.extensions,
            )
        return request


class AsyncTransportWrapper(_TransportWrapper):
    """Async transport wrapper."""

    async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
        return await self._wrapped.handle_async_request(self._modify_request(request))

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class SyncTransportWrapper(_TransportWrapper):
    """Sync transport wrapper."""

    def handle_request(self, request: "httpx.Request") -> "httpx.Response":
        return self._wrapped.handle_request(self._modify_request(request))

    def close(self) -> None:
        self._wrapped.close()


def patch_httpx() -> None:
    """Patch httpx clients to use session-aware transports."""
    global _httpx_patched

    if _httpx_patched:
        return

    try:
        import httpx
    except ImportError:
        return

    for client_cls, wrapper_cls, key in [
        (httpx.AsyncClient, AsyncTransportWrapper, "async"),
        (httpx.Client, SyncTransportWrapper, "sync"),
    ]:
        original = client_cls.__init__
        _original_inits[key] = original

        @functools.wraps(original)
        def patched_init(self, *args, _orig=original, _wrapper=wrapper_cls, **kwargs):
            _orig(self, *args, **kwargs)
            if _proxy_urls and self._transport:
                self._transport = _wrapper(self._transport)

        client_cls.__init__ = patched_init

    _httpx_patched = True


def unpatch_httpx() -> None:
    """Remove httpx patches."""
    global _httpx_patched

    if not _httpx_patched:
        return

    try:
        import httpx
    except ImportError:
        return

    if "async" in _original_inits:
        httpx.AsyncClient.__init__ = _original_inits["async"]
    if "sync" in _original_inits:
        httpx.Client.__init__ = _original_inits["sync"]

    _original_inits.clear()
    _httpx_patched = False
