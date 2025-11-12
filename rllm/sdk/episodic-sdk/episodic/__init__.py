"""
Episodic Python SDK - Context Store for AI Applications

A clean implementation of the Episodic context store with clear client-server separation:
- SqliteContextStore: Local SQLite-based storage backend
- Episodic: HTTP/WebSocket client for remote servers
- Server: FastAPI server implementation
"""

from .base import BaseContextStore
from .client import ContextStore, ContextStoreClient, Episodic
from .core import Context, ContextFilter, ContextNotFoundException, ContextStoreException, ContextUpdate, SubscriptionException
from .store import SqliteContextStore
from .subscriptions import ActionHandler, ContextSubscriber, WebSocketSubscription

__version__ = "0.1.0"
__all__ = [
    "Context",
    "ContextFilter",
    "ContextUpdate",
    "BaseContextStore",
    "SqliteContextStore",
    "Episodic",
    "ContextStoreClient",
    "ContextStore",  # Backward compatibility alias
    "ContextSubscriber",
    "ActionHandler",
    "WebSocketSubscription",
    "ContextStoreException",
    "ContextNotFoundException",
    "SubscriptionException",
]
