# Episodic Context Store - New Architecture

This document describes the restructured Episodic Context Store architecture with clear separation between local storage, remote clients, and server components.

## Class Hierarchy

- **BaseContextStore** - Abstract base class defining the context store interface
- **SqliteContextStore** - Local SQLite-based storage implementation (in `local.py`)
- **ContextStoreClient** - Base HTTP/WebSocket client for remote servers (in `client.py`)
- **Episodic** - Main client class (extends ContextStoreClient, in `client.py`)

## Architecture Overview

The new architecture consists of three main components:

1. **SqliteContextStore** - Local SQLite-based storage backend
2. **Episodic Client** - HTTP/WebSocket client for remote servers (extends ContextStoreClient)
3. **FastAPI Server** - HTTP API server with WebSocket support

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Episodic Client   │───▶│   FastAPI Server    │───▶│ SqliteContextStore  │
│ (ContextStoreClient)│    │   (HTTP + WS API)   │    │   (Local SQLite)    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                          │                          │
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   WebSocket         │    │   REST API          │    │   Local Database    │
│   Subscriptions     │    │   Endpoints         │    │   File Storage      │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Components

### 1. SqliteContextStore (Backend)

**Previously**: `LocalContextStore`  
**Now**: `SqliteContextStore`

The SQLite-based context store provides:
- Local persistent storage using SQLite
- Full context store functionality without remote dependencies
- Semantic search with local embeddings (sentence-transformers)
- Real-time subscriptions and notifications
- Automatic cleanup of expired contexts

```python
from episodic import SqliteContextStore

# Create local store
store = SqliteContextStore(db_path="./contexts.db")

# Store context
await store.store("user_123", {"name": "Alice"}, text="User Alice")

# Query contexts  
contexts = await store.query(ContextFilter(namespaces=["users"]))
```

### 2. Episodic Client (Remote Client)

**New**: `Episodic` class (extends `ContextStoreClient`)

A dedicated client for connecting to remote Episodic servers:
- HTTP/HTTPS communication with REST API
- WebSocket support for real-time subscriptions
- Authentication via API keys
- Automatic reconnection and error handling

```python
from episodic import Episodic

# Connect to remote server
client = Episodic("https://your-server.com", api_key="your-key")

# Store context remotely
await client.store("session_456", {"user": "bob"})

# Subscribe to updates
@client.on_context_update(namespaces=["sessions"])
async def handle_update(update):
    print(f"Context updated: {update.context.id}")
```

### 3. FastAPI Server

**New**: Server implementation using FastAPI

Exposes SqliteContextStore via HTTP REST API:
- Full REST API for all context operations
- WebSocket endpoint for real-time subscriptions
- CORS support for web applications
- Health checks and diagnostics
- Configurable database backend

```python
from episodic.server import create_app
import uvicorn

# Create server app
app = create_app(db_path="./server.db")

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Command Line Interface

The new CLI provides comprehensive context management:

### Basic Usage

```bash
# Store a context
episodic store user_123 '{"name": "Alice", "role": "admin"}' --text "Alice is an admin"

# Get a context
episodic get user_123 --format json

# Query contexts
episodic query --namespaces users --tags admin --format table

# Search contexts
episodic search text "admin user" --limit 5
episodic search semantic "user management" --threshold 0.8

# Delete a context
episodic delete user_123

# Check health
episodic health
```

### Server Management

```bash
# Run server locally
episodic server --host 0.0.0.0 --port 8000 --db-path ./production.db

# Run with auto-reload for development
episodic server --reload
```

### Configuration

Use environment variables for configuration:

```bash
export EPISODIC_ENDPOINT="https://your-server.com"
export EPISODIC_API_KEY="your-api-key"
export EPISODIC_NAMESPACE="production"

# Now CLI commands use these settings
episodic store context_id '{"data": "value"}'
```

## API Endpoints

The FastAPI server exposes these endpoints:

### Context Management
- `POST /contexts` - Store a new context
- `GET /contexts/{context_id}` - Get context by ID
- `DELETE /contexts/{context_id}` - Delete context
- `POST /contexts/query` - Query contexts with filters

### Search
- `POST /contexts/search/text` - Text-based search
- `POST /contexts/search/semantic` - Semantic similarity search
- `POST /contexts/search/hybrid` - Hybrid search with filters

### Real-time
- `WebSocket /ws/{client_id}` - WebSocket for subscriptions

### System
- `GET /health` - Health check
- `GET /diagnostics` - System diagnostics

## Migration Guide

### Clean Client-Server Separation

```python
# For local storage - use SqliteContextStore directly
from episodic import SqliteContextStore
store = SqliteContextStore(db_path="./contexts.db")

# For remote servers - use Episodic client (recommended)
from episodic import Episodic
client = Episodic("https://your-server.com", api_key="your-key")

# Backward compatibility - ContextStore is exactly the same as Episodic
from episodic import ContextStore
client = ContextStore("https://your-server.com", api_key="your-key")
```

### Using Remote Servers

```python
# Use Episodic client for remote servers (recommended)
from episodic import Episodic
client = Episodic("https://server.com", api_key="key")

# Backward compatible usage
from episodic import ContextStore
client = ContextStore("https://server.com", api_key="key")  # Same as Episodic

# Use SqliteContextStore for local storage
from episodic import SqliteContextStore
store = SqliteContextStore(db_path="./local.db")
```

## Installation

Install with CLI support:

```bash
pip install episodic
```

The `episodic` command will be available after installation.

## Development

Run the example:

```bash
cd examples
python basic_usage.py
```

Start a development server:

```bash
episodic server --reload --port 8000
```

Test the CLI:

```bash
episodic store test_context '{"message": "hello"}' --text "Test message"
episodic get test_context
episodic query --limit 5
```

## Backward Compatibility

For users migrating from older versions, we provide full backward compatibility:

### ContextStore Alias

The `ContextStore` class is provided as an exact alias for `Episodic`:

```python
# Old code continues to work
from episodic import ContextStore
client = ContextStore("https://server.com", api_key="key")

# This is exactly equivalent to:
from episodic import Episodic  
client = Episodic("https://server.com", api_key="key")
```

### Migration Path

```python
# Before: Factory function (no longer available)
# store = ContextStore("sqlite://")  # Would return LocalContextStore
# client = ContextStore("https://server.com")  # Would return RemoteContextStore

# After: Explicit classes
from episodic import SqliteContextStore, ContextStore

# For local storage
store = SqliteContextStore(db_path="./local.db")

# For remote clients (both work identically)
client = ContextStore("https://server.com")  # Backward compatible
client = Episodic("https://server.com")      # Recommended
```

## Benefits of New Architecture

1. **Clear Separation**: Distinct components for local storage, remote clients, and servers
2. **Better Naming**: `SqliteContextStore` clearly indicates the storage backend
3. **Dedicated Client**: `Episodic` class provides a clean remote client interface
4. **Full Server**: FastAPI server with comprehensive REST API and WebSocket support
5. **CLI Tool**: Complete command-line interface for all operations
6. **Backward Compatible**: `ContextStore` alias ensures existing code works unchanged
7. **Extensible**: Easy to add new storage backends or client types in the future

## File Structure

The codebase is now organized with clear separation:

- `episodic/local.py` - SqliteContextStore (local storage backend)
- `episodic/client.py` - ContextStoreClient and Episodic (remote client classes)
- `episodic/server/` - FastAPI server implementation
- `episodic/cli.py` - Command-line interface
- `episodic/core.py` - Core data structures (Context, ContextFilter, etc.)
- `episodic/base.py` - Abstract base classes
- `episodic/subscriptions.py` - Subscription and notification system

This architecture provides a solid foundation for building context-oriented AI applications with both local and distributed deployment options. 