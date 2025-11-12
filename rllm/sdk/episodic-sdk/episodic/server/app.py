"""
FastAPI server implementation for Episodic Context Store.
Exposes the SqliteContextStore backend via HTTP REST API with WebSocket support.
Compatible with episodic-cloud API interface.
"""
# ruff: noqa: B008, B904
# B008: FastAPI uses Depends() in default arguments as part of its dependency injection pattern
# B904: HTTPException raising pattern is standard for FastAPI

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, validator

from ..core import Context, ContextNotFoundException, ContextUpdate
from ..core import ContextFilter as CoreContextFilter
from ..store import SqliteContextStore

logger = logging.getLogger(__name__)


# Enums for search functionality
class TextSearchMode(str, Enum):
    """Text search modes."""

    EXACT = "exact"
    PHRASE = "phrase"
    FUZZY = "fuzzy"


class SearchType(str, Enum):
    """Search result types."""

    TEXT = "text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchStrategy(str, Enum):
    """Search strategies for hybrid search."""

    SEMANTIC_FIRST = "semantic_first"
    TEXT_FIRST = "text_first"
    BALANCED = "balanced"


# Pydantic models for API - Updated to match episodic-cloud interface
class ContextData(BaseModel):
    """Request model for storing context - matches episodic-cloud ContextData."""

    context_id: str = Field(..., alias="id")
    data: dict[str, Any]
    text: str | None = None
    namespace: str = "default"
    context_type: str = Field("generic", alias="type")
    metadata: dict[str, Any] = {}
    tags: list[str] = []
    ttl: int | None = None
    created_at: float | None = None
    updated_at: float | None = None
    expires_at: float | None = None
    auto_render_text: bool = False

    class Config:
        populate_by_name = True


class StoreContextDirectRequest(BaseModel):
    """Request model for storing context object directly."""

    context: dict[str, Any]  # Context object as dict


class ContextBatchRequest(BaseModel):
    """Request model for batch storing contexts."""

    contexts: list[ContextData] = Field(..., min_items=1, max_items=100)
    preserve_order: bool = True


class ContextFilter(BaseModel):
    """Request model for querying contexts - matches episodic-cloud."""

    namespaces: list[str] | None = None
    tags: list[str] | None = None
    context_types: list[str] | None = None
    since: str | None = None
    limit: int = 100
    include_expired: bool = False


class SearchRequest(BaseModel):
    """Basic search request model."""

    query: str
    namespaces: list[str] | None = None
    limit: int = 10


class TextSearchRequest(BaseModel):
    """Request model for text search - matches episodic-cloud."""

    query: str
    namespaces: list[str] | None = None
    search_mode: TextSearchMode = TextSearchMode.PHRASE
    include_ranking: bool = True
    rank_threshold: float = 0.0
    limit: int = 10

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("rank_threshold")
    def validate_rank_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Rank threshold must be between 0.0 and 1.0")
        return v

    @validator("limit")
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError("Limit must be between 1 and 100")
        return v


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search - matches episodic-cloud."""

    query: str
    namespaces: list[str] | None = None
    similarity_threshold: float = 0.7
    include_similarity_score: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    limit: int = 10

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("similarity_threshold")
    def validate_similarity_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v

    @validator("limit")
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError("Limit must be between 1 and 100")
        return v


class HybridSemanticSearchRequest(BaseModel):
    """Request model for hybrid semantic search - matches episodic-cloud."""

    query: str
    filters: ContextFilter | None = None
    search_strategy: SearchStrategy = SearchStrategy.BALANCED
    semantic_weight: float = 0.6
    text_weight: float = 0.4
    similarity_threshold: float = 0.5
    text_rank_threshold: float = 0.1
    limit: int = 15

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator("semantic_weight", "text_weight")
    def validate_weights(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v

    @validator("text_weight")
    def validate_weights_sum(cls, v, values):
        if "semantic_weight" in values:
            weight_sum = v + values["semantic_weight"]
            if not (0.99 <= weight_sum <= 1.01):  # 0.01 tolerance
                raise ValueError("Semantic weight + text weight must sum to 1.0")
        return v

    @validator("similarity_threshold", "text_rank_threshold")
    def validate_thresholds(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Thresholds must be between 0.0 and 1.0")
        return v

    @validator("limit")
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError("Limit must be between 1 and 100")
        return v


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""

    text_query: str
    filters: ContextFilter
    limit: int = 15


class CompositionRequest(BaseModel):
    """Request model for context composition - matches episodic-cloud."""

    composition_id: str
    components: list[dict[str, str]]
    merge_strategy: str = "priority_weighted"


class WebhookConfig(BaseModel):
    """Webhook configuration model."""

    url: HttpUrl
    secret: str | None = None
    headers: dict[str, str] | None = {}


class Subscription(BaseModel):
    """Subscription model - matches episodic-cloud."""

    subscription_id: str | None = None
    client_id: str
    delivery_method: str = "websocket"  # websocket or webhook
    webhook_config: WebhookConfig | None = None
    filters: ContextFilter


class Project(BaseModel):
    """Project model for organizing traces by project."""

    id: str
    name: str
    description: str | None = None
    namespace: str
    created_at: float
    updated_at: float | None = None


class ProjectCreate(BaseModel):
    """Request model for creating a project."""

    name: str
    description: str | None = None
    namespace: str

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()

    @validator("namespace")
    def validate_namespace(cls, v):
        if not v or not v.strip():
            raise ValueError("Namespace cannot be empty")
        return v.strip()


class ProjectUpdate(BaseModel):
    """Request model for updating a project."""

    name: str | None = None
    description: str | None = None

    @validator("name")
    def validate_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip() if v else None


class Dataset(BaseModel):
    """Dataset model for organizing traces for evaluation and annotation."""

    id: str
    name: str
    description: str | None = None
    trace_ids: list[str] = Field(default_factory=list)
    project_id: str | None = None
    created_at: float
    updated_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    """Request model for creating a dataset."""

    name: str
    description: str | None = None
    trace_ids: list[str] = Field(default_factory=list)
    project_id: str | None = None

    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()


class DatasetUpdate(BaseModel):
    """Request model for updating a dataset."""

    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None

    @validator("name")
    def validate_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip() if v else None


class DatasetTracesRequest(BaseModel):
    """Request model for adding/removing traces from a dataset."""

    trace_ids: list[str]

    @validator("trace_ids")
    def validate_trace_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one trace_id must be provided")
        return v


class SearchResult(BaseModel):
    """Individual search result."""

    context: dict[str, Any]
    semantic_similarity: float | None = None
    text_rank: float | None = None
    search_type: SearchType
    matched_fields: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response model for search operations."""

    results: list[SearchResult]
    total_found: int
    search_time_ms: float
    search_metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    texts: list[str]
    model: str = "all-MiniLM-L6-v2"

    @validator("texts")
    def validate_texts(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one text must be provided")
        return v


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    generation_time_ms: float


def _context_summary(context: Context) -> dict[str, Any]:
    """Create lightweight payload for routing/filtering."""
    full = context.to_dict()
    return {
        "id": full["id"],
        "data": {},  # omit heavy payload
        "text": None,
        "namespace": full.get("namespace", "default"),
        "type": full.get("type", "generic"),
        "metadata": full.get("metadata", {}),
        "tags": full.get("tags", []),
        "ttl": full.get("ttl"),
        "created_at": full.get("created_at"),
        "updated_at": full.get("updated_at"),
        "expires_at": full.get("expires_at"),
        "auto_render_text": full.get("auto_render_text", False),
    }


class WebSocketManager:
    """Manages WebSocket connections and subscriptions."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.client_subscriptions: dict[str, set[str]] = {}  # client_id -> set of namespaces
        self.subscriptions: dict[str, dict] = {}  # subscription_id -> subscription data

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = set()
        logger.info(f"WebSocket client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_update(self, update: ContextUpdate):
        """Broadcast context update to subscribed clients."""
        context_payload = update.context.to_dict() if update.includes_full_context else _context_summary(update.context)
        message = {"type": "context_update", "data": {"context_id": update.context.id, "context": context_payload, "operation": update.operation, "namespace": update.namespace, "includes_full_context": update.includes_full_context, "timestamp": time.time()}}
        message_str = json.dumps(message)

        # Send to clients subscribed to this namespace
        # Iterate over a snapshot so disconnects during send don't mutate the dict mid-loop
        for client_id, namespaces in list(self.client_subscriptions.items()):
            if not namespaces or update.namespace in namespaces or "*" in namespaces:
                await self.send_personal_message(message_str, client_id)

    def subscribe_to_namespace(self, client_id: str, namespace: str):
        """Subscribe a client to a namespace."""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].add(namespace)

    def unsubscribe_from_namespace(self, client_id: str, namespace: str):
        """Unsubscribe a client from a namespace."""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].discard(namespace)


class EpisodicServer:
    """Episodic Context Store Server."""

    def __init__(self, db_path: str | None = None, namespace: str = "default", enable_notifications: bool = True, compact_notifications: bool = False):
        """
        Initialize the Episodic server.

        Args:
            db_path: Path to SQLite database file
            namespace: Default namespace
            enable_notifications: Whether to broadcast context updates via WebSocket
        """
        self.context_store = SqliteContextStore(endpoint="sqlite://", namespace=namespace, db_path=db_path)
        self.notifications_enabled = enable_notifications
        self.compact_notifications = compact_notifications
        self.websocket_manager = WebSocketManager()

        if self.notifications_enabled:
            # Subscribe to context store updates to broadcast via WebSocket
            self.context_store.add_subscriber(self)
        else:
            logger.info("Context update notifications are disabled; broadcasts will be skipped.")

    async def handle_context_update(self, update: ContextUpdate):
        """Handle context updates from the store and broadcast to WebSocket clients."""
        if not self.notifications_enabled:
            return
        update.includes_full_context = not self.compact_notifications
        await self.websocket_manager.broadcast_update(update)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Episodic Context Store Server")
    yield
    # Shutdown
    logger.info("Shutting down Episodic Context Store Server")
    if hasattr(app.state, "server"):
        await app.state.server.context_store.close()


def create_app(db_path: str | None = None, namespace: str = "default", enable_notifications: bool | None = None, compact_notifications: bool | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        db_path: Path to SQLite database file
        namespace: Default namespace

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="Episodic Context Store Server", description="HTTP API for Episodic Context Store with real-time WebSocket subscriptions", version="0.1.0", lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if enable_notifications is None:
        disable_env = os.getenv("EPISODIC_DISABLE_NOTIFICATIONS", "")
        enable_notifications = disable_env.lower() not in ("1", "true", "yes")
        if not enable_notifications:
            logger.warning("EPISODIC_DISABLE_NOTIFICATIONS is set; skipping context update notifications.")
    if compact_notifications is None:
        compact_env = os.getenv("EPISODIC_COMPACT_NOTIFICATIONS", "")
        compact_notifications = compact_env.lower() in ("1", "true", "yes")
    # Initialize server
    server = EpisodicServer(db_path=db_path, namespace=namespace, enable_notifications=enable_notifications, compact_notifications=compact_notifications)
    app.state.server = server

    def get_server() -> EpisodicServer:
        """Dependency to get the server instance."""
        return app.state.server

    def _require_notifications_enabled(server: EpisodicServer):
        """Ensure notification-dependent features are enabled."""
        if not server.notifications_enabled:
            raise HTTPException(status_code=503, detail="Context update notifications are disabled on this server")

    # API Key validation - matches episodic-cloud approach
    async def verify_api_key(x_api_key: str | None = Header(None)):
        """Verify API key if configured."""
        # For local SQLite server, we don't enforce API keys by default
        # This can be configured if needed
        return True

    def _generate_text_from_data(data: dict[str, Any], context_type: str, context_id: str) -> str:
        """Generate text representation from structured data."""
        if not data:
            return f"Context {context_id}"

        text_parts = []
        for key, value in data.items():
            if isinstance(value, int | float | str | bool):
                text_parts.append(f"{key}: {value}")

        return f"{context_type} - {', '.join(text_parts)}"

    def _context_to_dict(context: Context) -> dict[str, Any]:
        """Convert Context object to dictionary matching episodic-cloud format."""
        context_dict = context.to_dict()
        # Ensure all expected fields are present
        context_dict.setdefault("auto_render_text", False)
        context_dict.setdefault("embedding", None)
        context_dict.setdefault("embedding_model", None)
        context_dict.setdefault("embedding_generated_at", None)
        return context_dict

    # Health check endpoint
    @app.get("/health")
    async def health_check(server: EpisodicServer = Depends(get_server)):  # noqa: B008
        """Health check endpoint."""
        try:
            health_result = await server.context_store.health_check()
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "services": {"sqlite": "healthy", "context_store": "healthy"}, **health_result}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    @app.get("/diagnostics")
    async def get_diagnostics(server: EpisodicServer = Depends(get_server)):  # noqa: B008
        """Get diagnostic information."""
        try:
            diagnostics = await server.context_store.get_diagnostics()
            # Add WebSocket connection info
            diagnostics.update({"active_websocket_connections": len(server.websocket_manager.active_connections), "websocket_subscriptions": len(server.websocket_manager.subscriptions), "timestamp": datetime.utcnow().isoformat()})
            return diagnostics
        except Exception as e:
            return {"error": f"Diagnostics collection failed: {str(e)}"}

    @app.get("/metrics")
    async def metrics(server: EpisodicServer = Depends(get_server)):  # noqa: B008
        """Basic metrics endpoint - matches episodic-cloud interface."""
        try:
            diagnostics = await server.context_store.get_diagnostics()
            return {
                "active_contexts": diagnostics.get("total_contexts", 0),
                "active_websocket_connections": len(server.websocket_manager.active_connections),
                "websocket_subscriptions": len(server.websocket_manager.subscriptions),
                "webhook_subscriptions": 0,  # Not supported in local mode
                "webhook_deliveries_24h": {"total": 0, "successful": 0, "failed": 0},
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {"error": "Metrics collection failed"}

    @app.get("/auth/validate")
    async def auth_validate(_: bool = Depends(verify_api_key)):
        """Validate API key and return OK when valid."""
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    # Context management endpoints - Updated to match episodic-cloud interface
    @app.post("/contexts")
    async def store_context(context: ContextData, background_tasks: BackgroundTasks, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Store context data - main endpoint used by SDK."""
        try:
            current_time = time.time()
            context_id = context.context_id

            # Set timestamps
            if not context.created_at:
                context.created_at = current_time
            context.updated_at = current_time

            # Calculate expiration
            if context.ttl:
                context.expires_at = current_time + context.ttl

            # Generate text if auto_render_text is enabled and text is not provided
            text = context.text
            if context.auto_render_text and not text:
                text = _generate_text_from_data(context.data, context.context_type, context_id)

            stored_context = await server.context_store.store(context_id=context_id, data=context.data, text=text, ttl=context.ttl, tags=context.tags, namespace=context.namespace, context_type=context.context_type, metadata=context.metadata)

            return _context_to_dict(stored_context)
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            raise HTTPException(status_code=500, detail="Storage failed") from e

    @app.post("/contexts/object")
    async def store_context_object(context: ContextData, background_tasks: BackgroundTasks, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Store a Context object directly - alternative endpoint."""
        return await store_context(context, background_tasks, server)

    @app.post("/contexts/direct")
    async def store_context_direct(request: StoreContextDirectRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Store a context object directly."""
        try:
            context = Context.from_dict(request.context)
            stored_context = await server.context_store.store_context(context)
            return _context_to_dict(stored_context)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/contexts/batch")
    async def store_contexts_batch(request: ContextBatchRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Store multiple contexts in a single transaction."""
        try:
            contexts_to_store: list[Context] = []
            for context_data in request.contexts:
                current_time = time.time()
                context_id = context_data.context_id
                created_at = context_data.created_at or current_time
                updated_at = current_time
                expires_at = None
                if context_data.ttl:
                    expires_at = current_time + context_data.ttl

                text = context_data.text
                if context_data.auto_render_text and not text:
                    text = _generate_text_from_data(context_data.data, context_data.context_type, context_id)

                context_obj = Context(
                    id=context_id,
                    data=context_data.data,
                    text=text,
                    namespace=context_data.namespace,
                    type=context_data.context_type,
                    metadata=context_data.metadata,
                    tags=context_data.tags,
                    ttl=context_data.ttl,
                    created_at=created_at,
                    updated_at=updated_at,
                    expires_at=expires_at,
                    auto_render_text=context_data.auto_render_text,
                )
                contexts_to_store.append(context_obj)

            stored_contexts = await server.context_store.store_batch(contexts_to_store)
            return {"count": len(stored_contexts), "preserve_order": request.preserve_order, "contexts": [_context_to_dict(context) for context in stored_contexts]}
        except Exception as e:
            logger.error(f"Batch storage error: {e}")
            raise HTTPException(status_code=500, detail="Batch storage failed") from e

    @app.get("/contexts/{context_id}")
    async def get_context(context_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Get a specific context by ID."""
        try:
            context = await server.context_store.get(context_id)
            return _context_to_dict(context)
        except ContextNotFoundException as e:
            raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/contexts/{context_id}")
    async def delete_context(context_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Delete a context by ID."""
        try:
            success = await server.context_store.delete(context_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found")
            return {"status": "deleted", "context_id": context_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/contexts/query")
    async def query_contexts(filter: ContextFilter, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Query contexts based on filters."""
        try:
            filter_obj = CoreContextFilter(namespaces=filter.namespaces, tags=filter.tags, context_types=filter.context_types, since=filter.since, limit=filter.limit)
            contexts = await server.context_store.query(filter_obj)
            return [_context_to_dict(context) for context in contexts]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/contexts/count")
    async def count_contexts(filter: ContextFilter, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Count contexts based on filters (without limit)."""
        try:
            # Use the core ContextFilter for counting (limit is ignored by count method)
            filter_obj = CoreContextFilter(
                namespaces=filter.namespaces or [],
                tags=filter.tags or [],
                context_types=filter.context_types or [],
                since=filter.since,
                limit=999999,  # High limit, but count() method ignores it anyway
                include_expired=filter.include_expired,
            )
            count = await server.context_store.count(filter_obj)
            return {"count": count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/contexts/search/text", response_model=SearchResponse)
    async def search_text(request: TextSearchRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Search contexts by text content."""
        try:
            start_time = time.time()
            contexts = await server.context_store.search_text(query=request.query, namespaces=request.namespaces, limit=request.limit)
            search_time_ms = (time.time() - start_time) * 1000

            results = []
            for context in contexts:
                results.append(SearchResult(context=_context_to_dict(context), search_type=SearchType.TEXT, matched_fields=["text"]))

            return SearchResponse(results=results, total_found=len(results), search_time_ms=search_time_ms, search_metadata={"search_mode": request.search_mode})
        except Exception as e:
            logger.error(f"Text search error: {e}")
            raise HTTPException(status_code=500, detail="Text search failed") from e

    @app.post("/contexts/search/semantic", response_model=SearchResponse)
    async def search_semantic(request: SemanticSearchRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Search contexts using semantic similarity."""
        try:
            start_time = time.time()
            contexts = await server.context_store.search_semantic(query=request.query, namespaces=request.namespaces, similarity_threshold=request.similarity_threshold, limit=request.limit)
            search_time_ms = (time.time() - start_time) * 1000

            results = []
            for context in contexts:
                results.append(
                    SearchResult(
                        context=_context_to_dict(context),
                        semantic_similarity=request.similarity_threshold,  # Placeholder
                        search_type=SearchType.SEMANTIC,
                        matched_fields=["text", "data"],
                    )
                )

            return SearchResponse(results=results, total_found=len(results), search_time_ms=search_time_ms, search_metadata={"similarity_threshold": request.similarity_threshold})
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            raise HTTPException(status_code=500, detail="Semantic search failed") from e

    @app.post("/contexts/search/hybrid", response_model=list[dict[str, Any]])
    async def search_hybrid(request: HybridSearchRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):  # noqa: B008
        """Hybrid search combining text search and metadata filters."""
        try:
            contexts = await server.context_store.search_hybrid(text_query=request.text_query, filters=request.filters, limit=request.limit)
            return [_context_to_dict(context) for context in contexts]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/contexts/search/hybrid_semantic", response_model=SearchResponse)
    async def search_hybrid_semantic(request: HybridSemanticSearchRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Hybrid search combining semantic similarity and text ranking."""
        try:
            start_time = time.time()
            # For local SQLite implementation, fall back to text search
            contexts = await server.context_store.search_text(query=request.query, namespaces=request.filters.namespaces if request.filters else None, limit=request.limit)
            search_time_ms = (time.time() - start_time) * 1000

            results = []
            for context in contexts:
                results.append(
                    SearchResult(
                        context=_context_to_dict(context),
                        semantic_similarity=request.similarity_threshold,
                        text_rank=0.5,  # Placeholder
                        search_type=SearchType.HYBRID,
                        matched_fields=["text", "data"],
                    )
                )

            return SearchResponse(results=results, total_found=len(results), search_time_ms=search_time_ms, search_metadata={"search_strategy": request.search_strategy, "semantic_weight": request.semantic_weight, "text_weight": request.text_weight})
        except Exception as e:
            logger.error(f"Hybrid semantic search error: {e}")
            raise HTTPException(status_code=500, detail="Hybrid semantic search failed")

    @app.post("/contexts/compose")
    async def compose_contexts(request: CompositionRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Compose multiple contexts into a single context."""
        try:
            context = await server.context_store.compose(composition_id=request.composition_id, components=request.components, merge_strategy=request.merge_strategy)
            return _context_to_dict(context)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Embedding generation endpoint - matches episodic-cloud
    @app.post("/embeddings/generate", response_model=EmbeddingResponse)
    async def generate_embeddings(request: EmbeddingRequest, _: bool = Depends(verify_api_key)):
        """Generate embeddings for given texts."""
        try:
            start_time = time.time()
            # For local implementation, return dummy embeddings
            # In a real implementation, this would use an embedding model
            embeddings = []
            for text in request.texts:
                # Generate dummy embedding of appropriate size
                embedding = [0.1] * 384  # all-MiniLM-L6-v2 has 384 dimensions
                embeddings.append(embedding)

            generation_time = time.time() - start_time

            return EmbeddingResponse(embeddings=embeddings, model=request.model, dimensions=384, generation_time_ms=generation_time * 1000)
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")

    @app.get("/health/search")
    async def health_check_search():
        """Health check for search functionality."""
        try:
            return {"status": "healthy", "search_engine": "local_sqlite", "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            logger.error(f"Search health check error: {e}")
            raise HTTPException(status_code=503, detail="Search health check failed")

    # Subscription management endpoints - matches episodic-cloud interface
    @app.post("/subscriptions")
    async def create_subscription(subscription: Subscription, server: EpisodicServer = Depends(get_server)):
        """Create a new subscription"""
        try:
            _require_notifications_enabled(server)
            subscription_id = subscription.subscription_id or str(uuid.uuid4())

            # Store subscription in WebSocket manager
            server.websocket_manager.subscriptions[subscription_id] = {
                "subscription_id": subscription_id,
                "client_id": subscription.client_id,
                "delivery_method": subscription.delivery_method,
                "webhook_config": subscription.webhook_config.dict() if subscription.webhook_config else None,
                "filters": subscription.filters.dict(),
            }

            return {"status": "created", "subscription_id": subscription_id}
        except Exception as e:
            logger.error(f"Subscription creation error: {e}")
            raise HTTPException(status_code=500, detail="Subscription creation failed")

    @app.delete("/subscriptions/{subscription_id}")
    async def delete_subscription(subscription_id: str, server: EpisodicServer = Depends(get_server)):
        """Delete a subscription"""
        try:
            _require_notifications_enabled(server)
            server.websocket_manager.subscriptions.pop(subscription_id, None)
            return {"status": "deleted", "subscription_id": subscription_id}
        except Exception as e:
            logger.error(f"Subscription deletion error: {e}")
            raise HTTPException(status_code=500, detail="Subscription deletion failed")

    @app.get("/subscriptions")
    async def list_subscriptions(client_id: str | None = None, server: EpisodicServer = Depends(get_server)):
        """List subscriptions"""
        try:
            _require_notifications_enabled(server)
            subscriptions = []
            for sub_id, sub_data in server.websocket_manager.subscriptions.items():
                if client_id is None or sub_data["client_id"] == client_id:
                    subscriptions.append({
                        "subscription_id": sub_id,
                        "client_id": sub_data["client_id"],
                        "delivery_method": sub_data["delivery_method"],
                        "webhook_url": sub_data.get("webhook_config", {}).get("url") if sub_data.get("webhook_config") else None,
                        "filters": sub_data["filters"],
                        "created_at": datetime.utcnow().isoformat(),
                        "last_delivery_at": None,
                        "delivery_failures": 0,
                    })
            return subscriptions
        except Exception as e:
            logger.error(f"Subscription listing error: {e}")
            raise HTTPException(status_code=500, detail="Subscription listing failed")

    # WebSocket endpoint for real-time subscriptions
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str, server: EpisodicServer = Depends(get_server)):
        """WebSocket endpoint for real-time context updates."""
        if not server.notifications_enabled:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "message": "Context update notifications are disabled on this server"}))
            await websocket.close()
            return
        await server.websocket_manager.connect(websocket, client_id)
        try:
            while True:
                # Receive subscription messages from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "subscribe":
                        namespace = message.get("namespace", "*")
                        server.websocket_manager.subscribe_to_namespace(client_id, namespace)
                        await websocket.send_text(json.dumps({"type": "subscription_confirmed", "namespace": namespace}))
                    elif message.get("type") == "unsubscribe":
                        namespace = message.get("namespace", "*")
                        server.websocket_manager.unsubscribe_from_namespace(client_id, namespace)
                        await websocket.send_text(json.dumps({"type": "unsubscription_confirmed", "namespace": namespace}))
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON message"}))
        except WebSocketDisconnect:
            server.websocket_manager.disconnect(client_id)

    # ===== Project Management Endpoints =====
    # Projects are stored as contexts with type="project" in namespace="system"

    @app.get("/projects", response_model=list[Project])
    async def list_projects(limit: int | None = Query(None), server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """List all projects."""
        try:
            # Query projects from context store
            filter_obj = ContextFilter(namespaces=["system"], context_types=["project"], limit=limit or 1000)
            contexts = await server.context_store.query(filter_obj)

            # Convert contexts to Project model
            projects = []
            for ctx in contexts:
                projects.append(Project(id=ctx.id, name=ctx.data.get("name", ""), description=ctx.data.get("description"), namespace=ctx.data.get("namespace", ""), created_at=ctx.created_at, updated_at=ctx.updated_at))

            return projects
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            raise HTTPException(status_code=500, detail="Failed to list projects")

    @app.post("/projects", response_model=Project, status_code=201)
    async def create_project(project: ProjectCreate, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Create a new project."""
        try:
            # Check if a project with this namespace already exists
            filter_obj = ContextFilter(namespaces=["system"], context_types=["project"], limit=1000)
            existing_projects = await server.context_store.query(filter_obj)

            for existing in existing_projects:
                if existing.data.get("namespace") == project.namespace:
                    raise HTTPException(status_code=409, detail=f"A project with namespace '{project.namespace}' already exists")

            # Generate project ID
            project_id = f"project_{uuid.uuid4().hex[:12]}"

            # Store project as a context
            context = await server.context_store.store(context_id=project_id, data={"name": project.name, "description": project.description, "namespace": project.namespace}, text=f"Project: {project.name}", namespace="system", context_type="project", metadata={})

            return Project(id=context.id, name=project.name, description=project.description, namespace=project.namespace, created_at=context.created_at, updated_at=context.updated_at)
        except HTTPException:
            # Re-raise HTTPExceptions without wrapping them
            raise
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

    @app.get("/projects/{project_id}", response_model=Project)
    async def get_project(project_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Get a specific project by ID."""
        try:
            context = await server.context_store.get(project_id)

            if context.type != "project":
                raise HTTPException(status_code=404, detail="Project not found")

            return Project(id=context.id, name=context.data.get("name", ""), description=context.data.get("description"), namespace=context.data.get("namespace", ""), created_at=context.created_at, updated_at=context.updated_at)
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Project not found")
        except Exception as e:
            logger.error(f"Error getting project: {e}")
            raise HTTPException(status_code=500, detail="Failed to get project")

    @app.put("/projects/{project_id}", response_model=Project)
    async def update_project(project_id: str, updates: ProjectUpdate, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Update a project."""
        try:
            # Get existing project
            context = await server.context_store.get(project_id)

            if context.type != "project":
                raise HTTPException(status_code=404, detail="Project not found")

            # Update data
            updated_data = context.data.copy()
            if updates.name is not None:
                updated_data["name"] = updates.name
            if updates.description is not None:
                updated_data["description"] = updates.description

            # Store updated context
            updated_context = await server.context_store.store(context_id=project_id, data=updated_data, text=f"Project: {updated_data.get('name', '')}", namespace="system", context_type="project", metadata=context.metadata)

            return Project(id=updated_context.id, name=updated_data.get("name", ""), description=updated_data.get("description"), namespace=updated_data.get("namespace", ""), created_at=updated_context.created_at, updated_at=updated_context.updated_at)
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Project not found")
        except Exception as e:
            logger.error(f"Error updating project: {e}")
            raise HTTPException(status_code=500, detail="Failed to update project")

    @app.delete("/projects/{project_id}", status_code=204)
    async def delete_project(project_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Delete a project."""
        try:
            # Verify it's a project before deleting
            context = await server.context_store.get(project_id)

            if context.type != "project":
                raise HTTPException(status_code=404, detail="Project not found")

            # Delete the project context
            success = await server.context_store.delete(project_id)

            if not success:
                raise HTTPException(status_code=404, detail="Project not found")

            return None
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Project not found")
        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete project")

    # ===== Dataset Management Endpoints =====
    # Datasets are stored as contexts with type="dataset" in namespace="system"

    @app.get("/datasets", response_model=list[Dataset])
    async def list_datasets(project_id: str | None = Query(None), limit: int | None = Query(None), server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """List all datasets."""
        try:
            # Query datasets from context store
            filter_obj = ContextFilter(namespaces=["system"], context_types=["dataset"], limit=limit or 1000)
            contexts = await server.context_store.query(filter_obj)

            # Convert contexts to Dataset model
            datasets = []
            for ctx in contexts:
                # Filter by project_id if provided
                if project_id and ctx.data.get("project_id") != project_id:
                    continue

                datasets.append(Dataset(id=ctx.id, name=ctx.data.get("name", ""), description=ctx.data.get("description"), trace_ids=ctx.data.get("trace_ids", []), project_id=ctx.data.get("project_id"), created_at=ctx.created_at, updated_at=ctx.updated_at, metadata=ctx.data.get("metadata", {})))

            return datasets
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise HTTPException(status_code=500, detail="Failed to list datasets")

    @app.post("/datasets", response_model=Dataset, status_code=201)
    async def create_dataset(dataset: DatasetCreate, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Create a new dataset."""
        try:
            # Generate dataset ID
            dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"

            # Store dataset as a context
            context = await server.context_store.store(context_id=dataset_id, data={"name": dataset.name, "description": dataset.description, "trace_ids": dataset.trace_ids, "project_id": dataset.project_id, "metadata": {}}, text=f"Dataset: {dataset.name}", namespace="system", context_type="dataset", metadata={})

            return Dataset(id=context.id, name=dataset.name, description=dataset.description, trace_ids=dataset.trace_ids, project_id=dataset.project_id, created_at=context.created_at, updated_at=context.updated_at, metadata={})
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

    @app.get("/datasets/{dataset_id}", response_model=Dataset)
    async def get_dataset(dataset_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Get a specific dataset by ID."""
        try:
            context = await server.context_store.get(dataset_id)

            if context.type != "dataset":
                raise HTTPException(status_code=404, detail="Dataset not found")

            return Dataset(id=context.id, name=context.data.get("name", ""), description=context.data.get("description"), trace_ids=context.data.get("trace_ids", []), project_id=context.data.get("project_id"), created_at=context.created_at, updated_at=context.updated_at, metadata=context.data.get("metadata", {}))
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as e:
            logger.error(f"Error getting dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to get dataset")

    @app.put("/datasets/{dataset_id}", response_model=Dataset)
    async def update_dataset(dataset_id: str, updates: DatasetUpdate, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Update a dataset."""
        try:
            # Get existing dataset
            context = await server.context_store.get(dataset_id)

            if context.type != "dataset":
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Update data
            updated_data = context.data.copy()
            if updates.name is not None:
                updated_data["name"] = updates.name
            if updates.description is not None:
                updated_data["description"] = updates.description
            if updates.metadata is not None:
                updated_data["metadata"] = updates.metadata

            # Store updated context
            updated_context = await server.context_store.store(
                context_id=dataset_id,
                data=updated_data,
                text=f"Dataset: {updated_data.get('name', '')}",
                namespace="system",
                context_type="dataset",
                metadata=context.metadata,
            )

            return Dataset(
                id=updated_context.id,
                name=updated_data.get("name", ""),
                description=updated_data.get("description"),
                trace_ids=updated_data.get("trace_ids", []),
                project_id=updated_data.get("project_id"),
                created_at=updated_context.created_at,
                updated_at=updated_context.updated_at,
                metadata=updated_data.get("metadata", {}),
            )
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as e:
            logger.error(f"Error updating dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to update dataset")

    @app.delete("/datasets/{dataset_id}", status_code=204)
    async def delete_dataset(dataset_id: str, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Delete a dataset."""
        try:
            # Verify it's a dataset before deleting
            context = await server.context_store.get(dataset_id)

            if context.type != "dataset":
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Delete the dataset context
            success = await server.context_store.delete(dataset_id)

            if not success:
                raise HTTPException(status_code=404, detail="Dataset not found")

            return None
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete dataset")

    @app.post("/datasets/{dataset_id}/traces", response_model=Dataset)
    async def add_traces_to_dataset(dataset_id: str, request: DatasetTracesRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Add traces to a dataset."""
        try:
            # Get existing dataset
            context = await server.context_store.get(dataset_id)

            if context.type != "dataset":
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Add trace IDs (avoid duplicates)
            current_trace_ids = set(context.data.get("trace_ids", []))
            current_trace_ids.update(request.trace_ids)

            # Update dataset
            updated_data = context.data.copy()
            updated_data["trace_ids"] = list(current_trace_ids)

            # Store updated context
            updated_context = await server.context_store.store(
                context_id=dataset_id,
                data=updated_data,
                text=f"Dataset: {updated_data.get('name', '')}",
                namespace="system",
                context_type="dataset",
                metadata=context.metadata,
            )

            return Dataset(
                id=updated_context.id,
                name=updated_data.get("name", ""),
                description=updated_data.get("description"),
                trace_ids=updated_data.get("trace_ids", []),
                project_id=updated_data.get("project_id"),
                created_at=updated_context.created_at,
                updated_at=updated_context.updated_at,
                metadata=updated_data.get("metadata", {}),
            )
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as e:
            logger.error(f"Error adding traces to dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to add traces to dataset")

    @app.delete("/datasets/{dataset_id}/traces", response_model=Dataset)
    async def remove_traces_from_dataset(dataset_id: str, request: DatasetTracesRequest, server: EpisodicServer = Depends(get_server), _: bool = Depends(verify_api_key)):
        """Remove traces from a dataset."""
        try:
            # Get existing dataset
            context = await server.context_store.get(dataset_id)

            if context.type != "dataset":
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Remove trace IDs
            current_trace_ids = set(context.data.get("trace_ids", []))
            for trace_id in request.trace_ids:
                current_trace_ids.discard(trace_id)

            # Update dataset
            updated_data = context.data.copy()
            updated_data["trace_ids"] = list(current_trace_ids)

            # Store updated context
            updated_context = await server.context_store.store(
                context_id=dataset_id,
                data=updated_data,
                text=f"Dataset: {updated_data.get('name', '')}",
                namespace="system",
                context_type="dataset",
                metadata=context.metadata,
            )

            return Dataset(
                id=updated_context.id,
                name=updated_data.get("name", ""),
                description=updated_data.get("description"),
                trace_ids=updated_data.get("trace_ids", []),
                project_id=updated_data.get("project_id"),
                created_at=updated_context.created_at,
                updated_at=updated_context.updated_at,
                metadata=updated_data.get("metadata", {}),
            )
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as e:
            logger.error(f"Error removing traces from dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to remove traces from dataset")

    return app


# Create default app instance for uvicorn
app = create_app()


# For running the server directly
if __name__ == "__main__":
    import asyncio

    import uvicorn

    async def run_server():
        app = create_app()
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    try:
        asyncio.run(run_server())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If we're in a running event loop (e.g., Jupyter), create a new task
            loop = asyncio.get_event_loop()
            loop.create_task(run_server())
        else:
            raise
