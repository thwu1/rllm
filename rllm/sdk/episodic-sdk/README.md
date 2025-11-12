# Context Store SDK - Client-Server Architecture

This Context Store SDK provides a flexible client-server architecture for building context-oriented AI agents. The SDK supports local SQLite storage, in-memory storage, and HTTP-based client-server communication.

## Architecture Overview

The SDK uses a clean factory pattern with separate implementations for local and remote storage:

- **BaseContextStore**: Abstract base class defining the interface
- **LocalContextStore**: SQLite-based implementation with persistent storage, semantic search, and full feature parity
- **RemoteContextStore**: HTTP client implementation for communicating with remote servers
- **ContextStore()**: Factory function that automatically chooses LocalContextStore or RemoteContextStore based on endpoint validation
- **FastAPI Server**: REST API server that hosts the context store

### New SQLite Local Store Features

The LocalContextStore now uses SQLite for persistent storage and includes:
- **Persistent Storage**: Survives application restarts
- **Semantic Search**: Optional vector embeddings using sentence-transformers
- **Full Feature Parity**: Same API as remote store (store, query, search, subscriptions)
- **Zero Setup**: Automatically creates database file in `~/.episodic/contexts.db`
- **High Performance**: Optimized indexes for fast queries

See [SQLITE_LOCAL_STORE.md](SQLITE_LOCAL_STORE.md) for detailed documentation.

### Factory Pattern Benefits

- **Automatic Selection**: No manual configuration - just provide an endpoint
- **Clean Separation**: Local and remote logic are completely separate
- **Easy Testing**: Mock individual components without complex fallback logic
- **Type Safety**: Better IDE support and type hints
- **Backward Compatible**: Existing code continues to work unchanged

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option A: Local SQLite Store (Recommended for Development)

The simplest way to get started - no server required!

```python
import asyncio
from episodic import LocalContextStore, ContextFilter

async def main():
    # Initialize local store with SQLite (creates ~/.episodic/contexts.db automatically)
    cs = LocalContextStore()
    
    # Store a context
    await cs.store(
        context_id="user.session.12345",
        data={"user_id": "alice", "preferences": {"theme": "dark"}},
        text="User Alice prefers dark theme",
        ttl=3600,  # 1 hour
        tags=["user", "session"]
    )
    
    # Retrieve and use the context
    context = await cs.get("user.session.12345")
    print(f"User: {context.data['user_id']}")
    
    # Search contexts
    results = await cs.search_text("Alice")
    print(f"Found {len(results)} contexts mentioning Alice")

asyncio.run(main())
```

For semantic search capabilities:
```bash
pip install episodic[semantic]  # Adds sentence-transformers for vector search
```

### Option B: Client-Server Architecture

For distributed systems or when you need centralized storage:

#### 1. Start the Server

Option 1: Using the startup script
```bash
python run_server.py
```

Option 2: Using uvicorn directly
```bash
uvicorn episodic.server:app --host 0.0.0.0 --port 8000 --reload
```

Option 3: Programmatically
```python
import uvicorn
from episodic.server import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

The server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

#### 2. Use the Context Store Client

The ContextStore factory function automatically chooses between LocalContextStore and RemoteContextStore based on the endpoint:

```python
import asyncio
from episodic import ContextStore, Context, ContextFilter

async def main():
    # Initialize the HTTP client
    client = ContextStore(
        endpoint="http://localhost:8000",
        api_key="your-api-key",
        namespace="my-app"
    )
    
    try:
        # Store a context
        context = await client.store(
            context_id="weather.sf.current",
            data={
                "temperature": 72,
                "conditions": "sunny",
                "humidity": 45
            },
            text="Current weather in San Francisco: 72°F, sunny",
            ttl=1800,  # 30 minutes
            tags=["weather", "real-time"]
        )
        
        # Retrieve the context
        retrieved = await client.get("weather.sf.current")
        print(f"Temperature: {retrieved.data['temperature']}°F")
        
        # Query contexts
        contexts = await client.query(
            ContextFilter(
                tags=["weather"],
                limit=10
            )
        )
        
        # Search by text
        results = await client.search_text("sunny weather")
        
    finally:
        await client.close()

asyncio.run(main())
```

### 3. Automatic Implementation Selection

The ContextStore factory automatically chooses the right implementation:

```python
import asyncio
from episodic import ContextStore, LocalContextStore, RemoteContextStore

async def main():
    # Invalid endpoints -> LocalContextStore
    local_store = ContextStore(endpoint="invalid-url")
    print(f"Type: {type(local_store).__name__}")  # LocalContextStore
    
    # Valid HTTP endpoints -> RemoteContextStore
    remote_store = ContextStore(endpoint="http://localhost:8000")
    print(f"Type: {type(remote_store).__name__}")  # RemoteContextStore
    
    # All operations work the same regardless of implementation
    context = await local_store.store("test", {"data": "works"})
    retrieved = await local_store.get("test")
    
    # Health check shows the mode
    health = await local_store.health_check()
    print(f"Mode: {health['mode']}")  # "local"
    
    # Direct instantiation for explicit control
    explicit_local = LocalContextStore(namespace="my-app")
    explicit_remote = RemoteContextStore("https://api.example.com")

asyncio.run(main())
```

### 4. Direct Instantiation

For explicit control over the implementation:

```python
import asyncio
from episodic import LocalContextStore, RemoteContextStore, ContextFilter

async def main():
    # Explicitly use local storage
    local_store = LocalContextStore(namespace="my-app")
    
    # Store a context
    context = await local_store.store(
        context_id="test.context",
        data={"key": "value"},
        text="Test context"
    )
    
    # Retrieve the context
    retrieved = await local_store.get("test.context")
    print(f"Retrieved: {retrieved.text}")
    
    # Explicitly use remote storage
    remote_store = RemoteContextStore("http://localhost:8000", namespace="my-app")
    # ... remote operations when server is available

asyncio.run(main())
```

## Benefits of the New Architecture

The factory pattern and separated implementations provide several advantages:

- **Clean Architecture**: Clear separation between local and remote implementations
- **Easy Testing**: Mock individual components without complex fallback logic
- **Type Safety**: Better IDE support and type hints for each implementation
- **Development Flexibility**: Start development without setting up a server
- **Explicit Control**: Choose implementation explicitly when needed
- **Backward Compatibility**: Existing code continues to work unchanged

## API Endpoints

The FastAPI server provides the following endpoints:

### Context Operations
- `POST /contexts` - Store a new context
- `POST /contexts/object` - Store a Context object directly
- `GET /contexts/{context_id}` - Get a specific context
- `DELETE /contexts/{context_id}` - Delete a context
- `POST /contexts/query` - Query contexts with filters
- `POST /contexts/search/text` - Search contexts by text
- `POST /contexts/search/hybrid` - Hybrid search (text + metadata)
- `POST /contexts/compose` - Compose multiple contexts

### System Operations
- `GET /health` - Health check
- `GET /diagnostics` - System diagnostics

## Core Classes

### BaseContextStore (Abstract)
Defines the interface that all context store implementations must follow:

```python
from episodic import BaseContextStore

class MyCustomStore(BaseContextStore):
    async def store(self, context_id: str, data: dict, **kwargs) -> Context:
        # Implement storage logic
        pass
    
    async def get(self, context_id: str) -> Context:
        # Implement retrieval logic
        pass
    
    # ... implement other abstract methods
```

### InMemoryContextStore
In-memory implementation with automatic cleanup of expired contexts:

```python
from episodic import InMemoryContextStore

store = InMemoryContextStore(
    endpoint="memory://",
    api_key="",
    namespace="default"
)
```

### ContextStore (HTTP Client)
HTTP client that communicates with the FastAPI server:

```python
from episodic import ContextStore

client = ContextStore(
    endpoint="http://localhost:8000",
    api_key="your-api-key",
    namespace="default",
    timeout=30
)
```

## Context Operations

### Storing Contexts

```python
# Simple storage
context = await client.store(
    context_id="unique.id",
    data={"key": "value"},
    text="Human readable text",
    ttl=3600,  # 1 hour
    tags=["tag1", "tag2"],
    namespace="my.namespace",
    context_type="custom_type",
    metadata={"source": "api"}
)

# Store Context object directly
context_obj = Context(
    id="another.id",
    data={"temperature": 25},
    text="Temperature is 25°C",
    namespace="sensors",
    type="temperature",
    tags=["sensor", "temperature"]
)
stored = await client.store_context(context_obj)
```

### Querying Contexts

```python
from episodic import ContextFilter

# Query with filters
contexts = await client.query(
    ContextFilter(
        namespaces=["sensors.*", "weather.*"],
        tags=["temperature"],
        context_types=["sensor_data"],
        since="1h",  # Last hour
        limit=50,
        include_expired=False
    )
)

# Text search
results = await client.search_text(
    query="temperature sensor",
    namespaces=["sensors.*"],
    limit=10
)

# Hybrid search (text + metadata)
hybrid_results = await client.search_hybrid(
    text_query="high temperature",
    filters=ContextFilter(
        namespaces=["sensors.*"],
        tags=["alert"]
    ),
    limit=15
)
```

### Context Composition

```python
# Compose multiple contexts
composed = await client.compose(
    composition_id="weather.summary",
    components=[
        {"namespace": "weather.current"},
        {"namespace": "weather.forecast"},
        {"namespace": "alerts.weather"}
    ],
    merge_strategy="priority_weighted"
)
```

## Error Handling

The SDK provides specific exceptions for different error conditions:

```python
from episodic import ContextNotFoundException, ContextStoreException

try:
    context = await client.get("non.existent.id")
except ContextNotFoundException as e:
    print(f"Context not found: {e}")
except ContextStoreException as e:
    print(f"Store error: {e}")
```

## Running the Demo

A comprehensive demo is available that shows all features:

```bash
# Start the server first
python run_server.py

# In another terminal, run the demo
python examples/client_server_demo.py
```

The demo will:
1. Check server health
2. Store various types of contexts
3. Retrieve and query contexts
4. Perform text and hybrid searches
5. Compose contexts
6. Show diagnostics
7. Clean up resources

## Development

### Project Structure
```
context-store-sdk/
├── episodic/
│   ├── __init__.py          # Package exports
│   ├── base.py              # BaseContextStore abstract class
│   ├── core.py              # Core data structures
│   ├── store.py             # InMemoryContextStore implementation
│   ├── client.py            # HTTP client implementation
│   ├── server.py            # FastAPI server
│   └── subscriptions.py     # Subscription system
├── examples/
│   └── client_server_demo.py # Demo script
├── requirements.txt         # Dependencies
├── run_server.py           # Server startup script
└── README.md               # This file
```

### Adding New Store Implementations

To add a new context store implementation:

1. Inherit from `BaseContextStore`
2. Implement all abstract methods
3. Add any implementation-specific features
4. Update the `__init__.py` exports

```python
from episodic.base import BaseContextStore

class DatabaseContextStore(BaseContextStore):
    def __init__(self, database_url: str, **kwargs):
        super().__init__(**kwargs)
        self.database_url = database_url
    
    async def store(self, context_id: str, data: dict, **kwargs) -> Context:
        # Implement database storage
        pass
    
    # ... implement other methods
```

## WebSocket Subscriptions

The Context Store SDK now supports real-time WebSocket subscriptions for receiving live updates when contexts are created, updated, or deleted.

### Using ContextSubscriber with Remote Context Store

The recommended approach is to use `ContextSubscriber` with a remote context store:

```python
import asyncio
from episodic import ContextStore, ContextSubscriber

async def main():
    # Create remote context store
    context_store = ContextStore(
        endpoint="http://localhost:8000",
        api_key="your-api-key",
        namespace="demo"
    )
    
    # Create subscriber
    subscriber = ContextSubscriber(context_store)
    
    # Define event handlers using decorators
    @subscriber.on_context_update(
        namespaces=["demo", "test"],
        tags=["important"],
        retry_policy={"max_retries": 3}
    )
    async def handle_important_updates(update):
        context = update.context
        print(f"Important update: {context.id} - {context.text}")
    
    @subscriber.on_context_batch(
        namespaces=["demo"],
        batch_size=5,
        max_wait_ms=3000
    )
    async def handle_batch_updates(contexts):
        print(f"Batch update: {len(contexts)} contexts")
    
    # Start the subscriber
    await subscriber.start()
    
    # Keep running
    await asyncio.sleep(30)
    
    # Clean up
    await subscriber.stop()
    await context_store.close()

asyncio.run(main())
```

### Using Standalone WebSocket Subscriptions

For direct WebSocket connections without using a context store:

```python
import asyncio
from episodic import WebSocketSubscription, ContextFilter

async def main():
    # Create direct WebSocket subscription
    websocket_url = "ws://localhost:8000/ws/client_id"
    
    async with WebSocketSubscription(websocket_url, api_key="your-api-key") as ws_sub:
        # Create filters
        important_filter = ContextFilter(
            namespaces=["demo"],
            tags=["important"]
        )
        
        # Define handlers
        async def handle_important(update):
            print(f"Important: {update.context.id}")
        
        # Subscribe
        await ws_sub.subscribe(important_filter, handle_important)
        
        # Wait for updates
        await asyncio.sleep(30)

asyncio.run(main())
```

### WebSocket Features

- **Automatic Reconnection**: Handles connection drops with exponential backoff
- **Filter-based Subscriptions**: Subscribe only to contexts matching specific criteria
- **Batch Processing**: Group multiple updates for efficient processing
- **Error Handling**: Retry logic with configurable policies
- **Async Support**: Fully asynchronous for high-performance applications

### Server Requirements

For WebSocket subscriptions to work, your Context Store server must:

1. Support WebSocket connections at `/ws/{client_id}`
2. Accept subscription messages with filters
3. Send context update notifications in the expected format
4. Handle authentication via headers or query parameters

### Notification Configuration

You can control whether the server emits WebSocket notifications—and how large those payloads are—via environment variables when starting `episodic.server.app`:

- `EPISODIC_DISABLE_NOTIFICATIONS`: set to `1`, `true`, or `yes` to keep the WebSocket endpoint offline. Subscriptions will be rejected, and the context store will not broadcast updates. Leave unset (default) to enable notifications.
- `EPISODIC_COMPACT_NOTIFICATIONS`: set to `1`, `true`, or `yes` to send compact notifications that include only routing metadata (context ID, namespace, tags, etc.). SDK clients automatically fetch the full context via `GET /contexts/{id}` when they receive one of these events, so your handler still sees a complete `Context`. Leave unset to include the full context payload directly in each WebSocket message.

Use the first switch when you want to run the API without any real-time features. Use the second switch when your contexts can be large and you prefer smaller, more reliable WebSocket frames while keeping the API surface unchanged.

## License

This project is part of the Context Store SDK for building context-oriented AI agents. 
