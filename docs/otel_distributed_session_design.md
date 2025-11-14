# OpenTelemetry Distributed Session Tracking - Design Document

**Author**: Claude (AI Assistant)
**Date**: 2025-01-14
**Status**: Proposed
**Related Issues**: Distributed tracing limitations in multi-process/microservice scenarios

## Executive Summary

**Problem**: The current `ContextVarSession` uses Python's `contextvars` which are process-local. This breaks session context propagation in distributed scenarios (Ray workers, microservices, multiprocessing).

**Solution**: Implement `OTelSession` as an alternative session implementation using OpenTelemetry's baggage mechanism for automatic distributed context propagation.

**Impact**:
- ✅ **Zero breaking changes** - Opt-in alternative to ContextVarSession
- ✅ **HTTP auto-propagation** - Context automatically flows across microservices
- ✅ **Same storage backends** - Compatible with InMemoryStorage and SqliteSessionStorage
- ⚠️ **Manual Ray propagation** - Still requires explicit context passing (better than current)

---

## Problem Analysis

### Current Architecture

```
User Code
    ↓
ContextVarSession (contextvars - process-local!)
    ↓
ProxyTrackedChatClient._scoped_client()
    ↓
assemble_routing_metadata() ← reads from contextvars
    ↓
build_proxied_base_url() ← encodes into URL slug
    ↓
HTTP Request with /meta/{slug}/v1/chat/completions
    ↓
MetadataRoutingMiddleware ← extracts slug from URL
    ↓
LiteLLM Callbacks ← reads from request.state
    ↓
SqliteTracer.log_llm_call() ← writes to storage
```

### Where It Breaks

**File**: `rllm/sdk/proxy/metadata_slug.py:16-30`

```python
def assemble_routing_metadata(extra=None):
    """Return the metadata dict that should be routed through the proxy slug."""
    payload = dict(get_current_metadata())  # ← Reads from contextvars
    session_name = get_current_session_name()  # ← Reads from contextvars
    # ...
```

**Problem**: `contextvars` are process-local and don't cross boundaries:
- ❌ Ray workers (new process → empty contextvars)
- ❌ Multiprocessing (separate process → empty contextvars)
- ❌ HTTP microservices (different process/machine → empty contextvars)

### Real-World Failure Example

```python
from rllm.sdk import session, get_chat_client
import ray

llm = get_chat_client(api_key="...", model="gpt-4")

@ray.remote
def train_episode(task):
    # ❌ session context is LOST here!
    # session_name = None, metadata = {}
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": task}]
    )
    return response

# Main process
with session(experiment="v1", run_id="123"):
    # This works - has session context
    llm.chat.completions.create(...)

    # ❌ Ray workers lose context!
    results = ray.get([train_episode.remote(task) for task in tasks])
    # All traces in Ray workers have no session_name or metadata
```

### Current Workaround

`ContextVarSession` provides manual serialization (`contextvar.py:255-311`):

```python
# Main process
with ContextVarSession(storage=storage) as session:
    context = session.to_context()  # Manual serialization
    send_to_ray_worker(context)

# Ray worker (Process 2)
context = receive_from_parent()
with ContextVarSession.from_context(context, storage) as session:
    llm.call()  # Context restored manually
```

**Problems**:
- Requires explicit `to_context()` / `from_context()` calls everywhere
- Error-prone (easy to forget)
- Doesn't work for HTTP calls (no automatic header injection)
- Pollutes function signatures

---

## Solution: OpenTelemetry Baggage-Based Propagation

### How OTel Solves It

**OpenTelemetry Baggage**: Key-value pairs that automatically propagate across:
- Process boundaries (when serialized/deserialized)
- HTTP boundaries (via `baggage` header - **automatic with instrumentation**)
- gRPC boundaries (via metadata - **automatic with instrumentation**)

**W3C Standard**: Uses W3C Trace Context and Baggage specifications

### With OTelSession

```python
from rllm.sdk.session import OTelSession
from rllm.sdk import get_chat_client
import ray
import requests

llm = get_chat_client(api_key="...", model="gpt-4")

@ray.remote
def train_episode(task, otel_ctx):
    # Restore context in Ray worker
    with OTelSession.from_otel_context(otel_ctx) as session:
        # ✅ Context restored! session_name and metadata available
        response = llm.chat.completions.create(
            messages=[{"role": "user", "content": task}]
        )
        return response

# Main process
with OTelSession(experiment="v1", run_id="123") as session:
    # ✅ Works - has session context
    llm.chat.completions.create(...)

    # Get context for Ray propagation
    otel_ctx = session.to_otel_context()

    # ✅ Manually pass to Ray workers (better than ContextVarSession)
    results = ray.get([
        train_episode.remote(task, otel_ctx)
        for task in tasks
    ])

    # ✅ HTTP calls work AUTOMATICALLY (with instrumentation)!
    response = requests.post("http://tool-service/execute", json={...})
    # Baggage header auto-injected by OTel instrumentation
```

**For HTTP Microservices** (fully automatic):

```http
# Main service → Tool service
POST /execute HTTP/1.1
Host: tool-service
traceparent: 00-4bf92f35-00f067aa-01
baggage: rllm_session_name=sess_abc123,rllm_experiment=v1,rllm_run_id=123

# Tool service automatically extracts context from headers!
# No code changes needed in the service
```

---

## Detailed Implementation Plan

### Phase 1: Core OTelSession Class

**Difficulty**: 🟢 **LOW** (2-3 days)

**New File**: `rllm/sdk/session/otel.py` (~300 lines)

**Key Components**:

1. **OTelSession class** - Implements `SessionProtocol`
2. **Baggage management** - Set/get from OTel context
3. **Serialization helpers** - `to_otel_context()` / `from_otel_context()`
4. **Helper functions** - `get_current_otel_session()`, `get_otel_metadata()`

**Dependencies** (add to `pyproject.toml`):
```toml
[project]
dependencies = [
    # ... existing deps
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
]

[project.optional-dependencies]
otel = [
    "opentelemetry-instrumentation-requests>=0.41b0",
    "opentelemetry-instrumentation-httpx>=0.41b0",
]
```

**API**:

```python
from rllm.sdk.session import OTelSession
from rllm.sdk.session.storage import SqliteSessionStorage

# Basic usage
with OTelSession(experiment="v1") as session:
    llm.chat.completions.create(...)

# With persistent storage
storage = SqliteSessionStorage("traces.db")
with OTelSession(name="task-123", storage=storage, experiment="v1") as session:
    llm.chat.completions.create(...)

# Serialization for Ray/multiprocessing
with OTelSession(experiment="v1") as session:
    ctx = session.to_otel_context()  # Dict for manual propagation

# Restoration
restored = OTelSession.from_otel_context(ctx, storage=storage)
with restored:
    llm.chat.completions.create(...)  # Has session context!
```

**Implementation Details**:

```python
class OTelSession(SessionProtocol):
    def __enter__(self):
        # Start OTel span
        self._span = self.tracer.start_span(f"session:{self.name}")
        self._span.__enter__()

        # Get current OTel context
        ctx = context.get_current()

        # Set session metadata in baggage
        ctx = baggage.set_baggage("rllm_session_name", self.name, context=ctx)
        ctx = baggage.set_baggage("rllm_session_uid", self._uid, context=ctx)

        # Set all metadata fields (with rllm_ prefix)
        for key, value in self.metadata.items():
            ctx = baggage.set_baggage(f"rllm_{key}", str(value), context=ctx)

        # Store list of metadata keys for retrieval
        metadata_keys = ",".join(self.metadata.keys())
        ctx = baggage.set_baggage("rllm_metadata_keys", metadata_keys, context=ctx)

        # Attach OTel context
        self._otel_token = context.attach(ctx)

        # Also set in process-local contextvars for get_current_otel_session()
        self._session_token = _current_otel_session.set(self)

        return self
```

**Storage Compatibility**:
- ✅ Works with `InMemoryStorage`
- ✅ Works with `SqliteSessionStorage`
- Uses same `session_uid_chain` mechanism for trace hierarchy

### Phase 2: Dual-Mode Metadata Assembly

**Difficulty**: 🟡 **MEDIUM** (1 day)

**Modified File**: `rllm/sdk/proxy/metadata_slug.py`

**Change**: Make `assemble_routing_metadata()` auto-detect session type

```python
def assemble_routing_metadata(extra=None):
    """Auto-detect session type and assemble metadata."""

    # Try OTel first (check if OTelSession is active)
    try:
        from rllm.sdk.session.otel import get_current_otel_session
        otel_session = get_current_otel_session()
    except ImportError:
        otel_session = None

    if otel_session:
        # OTelSession is active - read from OTel baggage
        from opentelemetry import baggage, context

        payload = {}
        ctx = context.get_current()

        # Get session name
        session_name = baggage.get_baggage("rllm_session_name", context=ctx)
        if session_name:
            payload["session_name"] = session_name

        # Get session UID
        session_uid = baggage.get_baggage("rllm_session_uid", context=ctx)
        if session_uid:
            payload["session_uids"] = [session_uid]

        # Get all metadata keys
        metadata_keys_str = baggage.get_baggage("rllm_metadata_keys", context=ctx) or ""
        for key in metadata_keys_str.split(","):
            if key.strip():
                value = baggage.get_baggage(f"rllm_{key}", context=ctx)
                if value:
                    payload[key] = value
    else:
        # ContextVarSession is active - use existing logic
        payload = dict(get_current_metadata())
        session_name = get_current_session_name()
        if session_name and "session_name" not in payload:
            payload["session_name"] = session_name

        # Add session UIDs from active sessions
        active_sessions = get_active_sessions()
        if active_sessions:
            payload["session_uids"] = [s._uid for s in active_sessions]

    # Merge extra metadata
    if extra:
        payload.update(dict(extra))

    return payload
```

**API Impact**: ✅ **NO BREAKING CHANGES** - Auto-detection is transparent

### Phase 3: HTTP Auto-Instrumentation

**Difficulty**: 🟡 **MEDIUM** (2 days)

**New Function in** `rllm/sdk/session/otel.py`:

```python
def setup_otel_http_instrumentation():
    """
    Configure OpenTelemetry HTTP auto-instrumentation.

    Call this once at application startup to enable automatic
    baggage propagation for HTTP requests.

    Example:
        >>> from rllm.sdk.session.otel import setup_otel_http_instrumentation
        >>> setup_otel_http_instrumentation()
        >>>
        >>> # Now HTTP calls auto-propagate baggage!
        >>> import requests
        >>> with OTelSession(experiment="v1"):
        ...     requests.post("http://service/api", ...)
        ...     # ✅ baggage header automatically injected!
    """
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # Instrument requests library
    RequestsInstrumentor().instrument()

    # Instrument httpx library
    HTTPXClientInstrumentor().instrument()

    logger.info("OpenTelemetry HTTP instrumentation enabled")
```

**User Setup** (one-time, optional):

```python
# In main.py or __init__.py
from rllm.sdk.session.otel import setup_otel_http_instrumentation

# Enable HTTP auto-instrumentation for microservices
setup_otel_http_instrumentation()

# Now OTelSession automatically propagates via HTTP headers!
```

**What This Enables**:

```python
# Service A
with OTelSession(experiment="v1", user="alice"):
    # ✅ HTTP request automatically includes baggage header
    response = requests.post("http://service-b/api", json={"task": "..."})

# Service B (separate machine/process)
# OTel middleware automatically extracts baggage from headers
@app.post("/api")
def handler():
    # ✅ Context automatically available!
    from rllm.sdk.session.otel import get_otel_session_name, get_otel_metadata

    session_name = get_otel_session_name()  # "sess_abc123"
    metadata = get_otel_metadata()  # {"experiment": "v1", "user": "alice"}

    llm.chat.completions.create(...)  # Trace has full context!
```

### Phase 4: Middleware Baggage Extraction

**Difficulty**: 🟢 **LOW** (1 day)

**Modified File**: `rllm/sdk/proxy/middleware.py`

**Add**: Extract metadata from `baggage` HTTP header in addition to URL slug

```python
class MetadataRoutingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        metadata: dict[str, Any] = {}

        # Method 1: Extract from URL slug (existing)
        extracted = extract_metadata_from_path(request.url.path)
        if extracted is not None:
            clean_path, metadata = extracted
            # ... existing code ...

        # Method 2: Extract from OTel baggage header (NEW)
        try:
            from opentelemetry import baggage, context
            from opentelemetry.propagate import extract

            # Extract W3C baggage from HTTP headers
            carrier = dict(request.headers)
            otel_ctx = extract(carrier)  # Extracts traceparent + baggage

            # Read rllm_* baggage entries
            otel_metadata = {}
            session_name = baggage.get_baggage("rllm_session_name", context=otel_ctx)
            if session_name:
                otel_metadata["session_name"] = session_name

            session_uid = baggage.get_baggage("rllm_session_uid", context=otel_ctx)
            if session_uid:
                otel_metadata["session_uids"] = [session_uid]

            # Get all metadata keys
            metadata_keys_str = baggage.get_baggage("rllm_metadata_keys", context=otel_ctx) or ""
            for key in metadata_keys_str.split(","):
                if key.strip():
                    value = baggage.get_baggage(f"rllm_{key}", context=otel_ctx)
                    if value:
                        otel_metadata[key] = value

            # Merge: baggage takes precedence over URL slug
            metadata = {**metadata, **otel_metadata}
        except ImportError:
            # OTel not installed, skip baggage extraction
            pass

        # Store in request.state (existing)
        request.state.rllm_metadata = metadata

        # ... rest of existing code ...
```

**API Impact**: ✅ **NO BREAKING CHANGES** - Additional extraction method

### Phase 5: Ray Integration Helpers

**Difficulty**: 🔴 **HIGH** (2 days)

**Challenge**: Ray doesn't automatically serialize OTel context across remote calls

**Solution**: Provide manual propagation helpers (similar to ContextVarSession but cleaner)

**New Helpers in** `rllm/sdk/session/otel.py`:

```python
# Helper for Ray remote functions
def ray_entrypoint(func):
    """
    Decorator to enable OTel context restoration in Ray workers.

    Usage:
        >>> @ray.remote
        >>> @ray_entrypoint
        >>> def train_episode(task):
        ...     # Context automatically restored if _otel_ctx passed
        ...     llm.chat.completions.create(...)
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        otel_ctx = kwargs.pop("_otel_ctx", None)

        if otel_ctx:
            from rllm.sdk.session.storage import SqliteSessionStorage
            storage = SqliteSessionStorage()

            with OTelSession.from_otel_context(otel_ctx, storage=storage):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
```

**User Code**:

```python
import ray
from rllm.sdk.session.otel import OTelSession, ray_entrypoint
from rllm.sdk.session.storage import SqliteSessionStorage

@ray.remote
@ray_entrypoint  # Automatic context restoration
def train_episode(task):
    # Context automatically available if _otel_ctx was passed
    llm.chat.completions.create(...)

# Main process
storage = SqliteSessionStorage("traces.db")
with OTelSession(experiment="v1", storage=storage) as session:
    otel_ctx = session.to_otel_context()

    # Pass _otel_ctx as kwarg (automatically handled by decorator)
    results = ray.get([
        train_episode.remote(task, _otel_ctx=otel_ctx)
        for task in tasks
    ])
```

**Alternative** (explicit restoration):

```python
@ray.remote
def train_episode(task, otel_ctx):
    storage = SqliteSessionStorage("traces.db")
    with OTelSession.from_otel_context(otel_ctx, storage=storage):
        llm.chat.completions.create(...)
```

**API Impact**: ⚠️ **NEW API** - Optional helpers, doesn't break existing code

---

## API Changes Summary

### ✅ Zero Breaking Changes

**All existing code continues to work:**

```python
# ContextVarSession (existing)
from rllm.sdk import session

with session(experiment="v1"):
    llm.chat.completions.create(...)  # ✅ Still works exactly as before!
```

### ✅ New Opt-In APIs

**1. OTelSession** (alternative to ContextVarSession):
```python
from rllm.sdk.session import OTelSession

with OTelSession(experiment="v1"):
    llm.chat.completions.create(...)  # ✅ New, opt-in
```

**2. HTTP auto-instrumentation** (optional):
```python
from rllm.sdk.session.otel import setup_otel_http_instrumentation

setup_otel_http_instrumentation()  # ✅ New, opt-in, one-time setup
```

**3. Ray helpers** (optional):
```python
from rllm.sdk.session.otel import OTelSession, ray_entrypoint

# Helper decorator
@ray_entrypoint  # ✅ New, opt-in
def train_episode(task):
    ...

# Manual serialization
otel_ctx = session.to_otel_context()  # ✅ New, opt-in
```

### Export from Main Package

**Modified File**: `rllm/sdk/__init__.py`

```python
from rllm.sdk.session import (
    ContextVarSession,
    SessionContext,  # Alias for ContextVarSession
    OTelSession,  # NEW
    # ... existing exports
)

__all__ = [
    # ... existing exports
    "OTelSession",  # NEW
]
```

---

## Comparison: ContextVarSession vs OTelSession

| Feature | ContextVarSession | OTelSession |
|---------|-------------------|-------------|
| **Single process** | ✅ Automatic | ✅ Automatic |
| **Threading** | ✅ Automatic | ✅ Automatic |
| **Async/await** | ✅ Automatic | ✅ Automatic |
| **Ray/multiprocessing** | ⚠️ Manual `to_context()` | ⚠️ Manual `to_otel_context()` |
| **HTTP microservices** | ❌ Manual headers | ✅ **Automatic** (with instrumentation) |
| **gRPC** | ❌ Manual | ✅ **Automatic** (with instrumentation) |
| **Setup complexity** | 🟢 Low (no deps) | 🟡 Medium (OTel SDK) |
| **Dependencies** | None | `opentelemetry-api`, `opentelemetry-sdk` |
| **Storage backends** | ✅ InMemory, SQLite | ✅ Same (InMemory, SQLite) |
| **Tracer compatibility** | ✅ All tracers | ✅ Same tracers |
| **Baggage size limit** | N/A | ⚠️ ~8KB total |

---

## Difficulty & Timeline Assessment

| Phase | Component | Difficulty | Estimate | Dependencies |
|-------|-----------|------------|----------|--------------|
| **1** | OTelSession class | 🟢 LOW | 2-3 days | None |
| **2** | Dual-mode metadata assembly | 🟡 MEDIUM | 1 day | Phase 1 |
| **3** | HTTP auto-instrumentation | 🟡 MEDIUM | 2 days | Phase 1 |
| **4** | Middleware baggage extraction | 🟢 LOW | 1 day | Phase 3 |
| **5** | Ray integration helpers | 🔴 HIGH | 2 days | Phase 1 |
| **6** | Testing (unit + integration) | 🟡 MEDIUM | 2 days | All phases |
| **7** | Documentation | 🟢 LOW | 1 day | All phases |

**Total Estimate**: **11-14 days** for full implementation and testing

**Can be delivered incrementally**:
- Week 1: Phases 1-2 (Core OTelSession + metadata assembly)
- Week 2: Phases 3-5 (HTTP instrumentation + middleware + Ray)
- Week 3: Phases 6-7 (Testing + documentation)

---

## Testing Strategy

### Unit Tests

**File**: `rllm/sdk/session/test_otel.py`

1. **Basic functionality**
   - OTelSession creation
   - Metadata setting/getting via baggage
   - Nested sessions (metadata inheritance)
   - Session UID chain construction

2. **Serialization**
   - `to_otel_context()` produces correct dict
   - `from_otel_context()` restores session correctly
   - Metadata preserved across serialization

3. **Dual-mode metadata assembly**
   - Auto-detects ContextVarSession correctly
   - Auto-detects OTelSession correctly
   - Returns correct metadata for each type

### Integration Tests

**File**: `rllm/sdk/session/test_otel_integration.py`

1. **HTTP propagation** (requires `requests` instrumentation)
   ```python
   def test_http_propagation():
       """Test baggage propagates via HTTP headers."""
       from rllm.sdk.session.otel import OTelSession, setup_otel_http_instrumentation

       setup_otel_http_instrumentation()

       with OTelSession(experiment="v1") as session:
           # Make HTTP request (mock server extracts baggage)
           response = requests.post("http://mock-server/api", json={})

           # Verify baggage was sent in headers
           assert "baggage" in response.request.headers
           assert "rllm_experiment=v1" in response.request.headers["baggage"]
   ```

2. **Ray propagation** (requires Ray)
   ```python
   def test_ray_propagation():
       """Test manual context propagation to Ray workers."""
       import ray
       from rllm.sdk.session.otel import OTelSession

       @ray.remote
       def worker(otel_ctx):
           with OTelSession.from_otel_context(otel_ctx) as session:
               # Verify metadata available
               assert session.metadata["experiment"] == "v1"
               return session.name

       with OTelSession(experiment="v1") as session:
           otel_ctx = session.to_otel_context()
           result = ray.get(worker.remote(otel_ctx))
           assert result == session.name
   ```

3. **Proxy round-trip** (full stack test)
   ```python
   def test_proxy_with_otel_session():
       """Test OTelSession → ProxyClient → Middleware → Tracer."""
       from rllm.sdk import get_chat_client
       from rllm.sdk.session.otel import OTelSession

       llm = get_chat_client(
           api_key="test",
           base_url="http://localhost:4000/v1",
           model="gpt-4"
       )

       with OTelSession(experiment="v1", run_id="123") as session:
           # Mock LLM call
           response = llm.chat.completions.create(
               messages=[{"role": "user", "content": "test"}]
           )

           # Verify trace has correct metadata
           traces = session.llm_calls
           assert len(traces) == 1
           assert traces[0].metadata["experiment"] == "v1"
           assert traces[0].metadata["run_id"] == "123"
   ```

### End-to-End Tests

1. **Microservices scenario**
   - Service A uses OTelSession
   - Makes HTTP call to Service B
   - Service B extracts context from baggage header
   - Both services write traces with correct session context

2. **Ray training scenario**
   - Main process creates OTelSession
   - Spawns Ray workers with serialized context
   - Workers restore context and make LLM calls
   - All traces have correct session metadata

3. **Hybrid scenario**
   - Uses both ContextVarSession and OTelSession
   - Metadata assembly works for both
   - No interference between the two

---

## Migration Guide

### When to Use OTelSession

**✅ Use OTelSession if you need:**
- HTTP microservices with automatic context propagation
- gRPC services with automatic context propagation
- Distributed tracing across multiple services
- W3C standard compliance
- Integration with existing OTel infrastructure

**✅ Keep using ContextVarSession if:**
- Single-process application
- No microservices or distributed architecture
- Want minimal dependencies
- Don't need automatic HTTP/gRPC propagation
- Current manual `to_context()` / `from_context()` works fine

### Migration Steps

**Step 1: Install OTel dependencies**
```bash
pip install "rllm[otel]"
```

**Step 2: Replace session import (optional)**
```python
# Before
from rllm.sdk import session

# After (for distributed scenarios)
from rllm.sdk.session import OTelSession as session
```

**Step 3: Enable HTTP instrumentation (for microservices)**
```python
# In main.py or __init__.py
from rllm.sdk.session.otel import setup_otel_http_instrumentation

# One-time setup
setup_otel_http_instrumentation()
```

**Step 4: Update Ray code (if using Ray)**
```python
# Before (with ContextVarSession)
@ray.remote
def train(task, ctx):
    with ContextVarSession.from_context(ctx, storage) as session:
        llm.call(...)

# After (with OTelSession)
@ray.remote
def train(task, otel_ctx):
    with OTelSession.from_otel_context(otel_ctx, storage) as session:
        llm.call(...)
```

**Step 5: Test - no other code changes needed!**

### Rollback Plan

If issues arise, simply revert to ContextVarSession:
```python
# Instant rollback - no code changes needed
from rllm.sdk import session  # Uses ContextVarSession by default
```

---

## Limitations & Constraints

### OTel Baggage Size Limit

**W3C Specification**: Total baggage size limited to ~8KB

**Impact**: Cannot store large metadata objects in baggage

**Mitigation**:
- Store only essential metadata in baggage (session_name, session_uid, small values)
- Store full metadata in SQLite using session_uid as key
- For large metadata, retrieve from storage rather than propagating

### Ray Doesn't Auto-Propagate OTel Context

**Problem**: Ray's serialization doesn't include OTel context automatically

**Current Solution**: Manual propagation via `to_otel_context()` / `from_otel_context()`

**Future Enhancement**: Custom Ray integration using Ray's runtime_context

**Comparison**:
- ContextVarSession: Requires `to_context()` / `from_context()`
- OTelSession: Also requires `to_otel_context()` / `from_otel_context()`
- **Neither is automatic for Ray** - same manual burden

### Performance Overhead

**OTel overhead**: ~0.1-0.5ms per request for baggage operations

**LLM call latency**: 100-1000ms typical

**Verdict**: **Negligible** - OTel overhead is <0.5% of total latency

**Mitigation**: Disable OTel if not needed (use ContextVarSession)

---

## Open Questions & Future Work

### 1. Automatic Ray Propagation

**Challenge**: Ray doesn't serialize OTel context

**Potential Solution**:
- Use Ray's `runtime_context` API
- Custom serializer that captures OTel context
- Monkey-patch Ray's `remote()` decorator

**Complexity**: HIGH

**Timeline**: Phase 2 (future enhancement)

### 2. Baggage Compression

**Challenge**: 8KB limit may be tight for large metadata

**Potential Solution**:
- Compress metadata before setting baggage
- Use shortened keys (e.g., `e` instead of `experiment`)
- Store only session_uid in baggage, retrieve full metadata from storage

**Complexity**: MEDIUM

### 3. Performance Benchmarks

**Needed**: Quantify OTel overhead vs ContextVarSession

**Tests**:
- Single-process throughput (contextvars vs baggage)
- HTTP propagation overhead (with/without instrumentation)
- Memory usage comparison

**Timeline**: After Phase 1 implementation

### 4. gRPC Support

**Similar to HTTP**: Should work with `opentelemetry-instrumentation-grpc`

**Testing needed**: Verify baggage propagates via gRPC metadata

**Timeline**: Phase 3 (alongside HTTP instrumentation)

---

## Success Criteria

### Phase 1 Success

- ✅ OTelSession works in single-process scenario
- ✅ Metadata stored correctly in baggage
- ✅ `to_otel_context()` / `from_otel_context()` work
- ✅ Compatible with InMemoryStorage and SqliteSessionStorage
- ✅ All unit tests pass

### Phase 2-3 Success

- ✅ `assemble_routing_metadata()` auto-detects session type
- ✅ HTTP instrumentation propagates baggage headers
- ✅ Middleware extracts metadata from baggage
- ✅ End-to-end test: Service A → Service B with context

### Phase 4-5 Success

- ✅ Ray manual propagation works
- ✅ Ray helpers simplify context passing
- ✅ Full integration test with Ray + SQLite storage

### Production Ready

- ✅ Zero breaking changes confirmed
- ✅ Documentation complete (migration guide + examples)
- ✅ All tests passing (unit + integration + e2e)
- ✅ Performance benchmarks acceptable (<1ms overhead)
- ✅ Users can opt-in without disrupting existing code

---

## References

### W3C Standards

- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [W3C Baggage](https://www.w3.org/TR/baggage/)

### OpenTelemetry Documentation

- [OTel Python SDK](https://opentelemetry.io/docs/languages/python/)
- [OTel Baggage API](https://opentelemetry.io/docs/concepts/signals/baggage/)
- [OTel Instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/)

### Related Code

- `rllm/sdk/session/contextvar.py` - Current ContextVarSession implementation
- `rllm/sdk/proxy/metadata_slug.py` - Metadata assembly logic
- `rllm/sdk/proxy/middleware.py` - Proxy middleware for metadata extraction
- `rllm/sdk/session/storage.py` - Storage protocol and implementations

---

## Appendix: Code Examples

### Example 1: Basic OTelSession Usage

```python
from rllm.sdk.session import OTelSession
from rllm.sdk import get_chat_client

llm = get_chat_client(api_key="...", model="gpt-4")

# Single-process usage (same as ContextVarSession)
with OTelSession(experiment="v1", user="alice") as session:
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )

    print(f"Session: {session.name}")
    print(f"Metadata: {session.metadata}")
    print(f"Traces: {len(session.llm_calls)}")
```

### Example 2: HTTP Microservices (Automatic Propagation)

```python
# Service A (main.py)
from rllm.sdk.session import OTelSession
from rllm.sdk.session.otel import setup_otel_http_instrumentation
import requests

# One-time setup
setup_otel_http_instrumentation()

# Use OTelSession
with OTelSession(experiment="v1", user="alice") as session:
    # ✅ Baggage auto-propagates via HTTP header
    response = requests.post(
        "http://service-b:8000/api",
        json={"task": "analyze"}
    )

# Service B (separate service)
from rllm.sdk import get_chat_client
from rllm.sdk.session.otel import get_otel_session_name, get_otel_metadata
from fastapi import FastAPI

app = FastAPI()
llm = get_chat_client(api_key="...", model="gpt-4")

@app.post("/api")
def handler(request):
    # ✅ Context automatically extracted from baggage header
    session_name = get_otel_session_name()
    metadata = get_otel_metadata()

    print(f"Session: {session_name}")  # Same as Service A
    print(f"Metadata: {metadata}")  # {"experiment": "v1", "user": "alice"}

    # LLM call has full context
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": request.task}]
    )

    return {"result": response.choices[0].message.content}
```

### Example 3: Ray with Manual Propagation

```python
import ray
from rllm.sdk.session import OTelSession
from rllm.sdk.session.storage import SqliteSessionStorage
from rllm.sdk import get_chat_client

# Initialize Ray
ray.init()

# Shared storage across processes
storage = SqliteSessionStorage("traces.db")
llm = get_chat_client(api_key="...", model="gpt-4")

@ray.remote
def train_episode(task, otel_ctx):
    """Ray worker function - runs in separate process."""
    # Restore OTelSession from serialized context
    with OTelSession.from_otel_context(otel_ctx, storage=storage):
        response = llm.chat.completions.create(
            messages=[{"role": "user", "content": task}]
        )
        return response.choices[0].message.content

# Main process
def train():
    with OTelSession(experiment="v1", run_id="123", storage=storage) as session:
        # Serialize context for Ray workers
        otel_ctx = session.to_otel_context()

        # Spawn 100 Ray workers
        tasks = [f"Task {i}" for i in range(100)]
        futures = [
            train_episode.remote(task, otel_ctx)
            for task in tasks
        ]

        # Wait for results
        results = ray.get(futures)

        # ✅ All traces have correct session context!
        print(f"Total traces: {len(session.llm_calls)}")  # 100+
        for trace in session.llm_calls:
            assert trace.metadata["experiment"] == "v1"
            assert trace.metadata["run_id"] == "123"

if __name__ == "__main__":
    train()
```

### Example 4: Nested Sessions with Metadata Inheritance

```python
from rllm.sdk.session import OTelSession
from rllm.sdk import get_chat_client

llm = get_chat_client(api_key="...", model="gpt-4")

# Outer session
with OTelSession(experiment="v1", dataset="train") as outer:
    llm.chat.completions.create(...)  # Has experiment=v1, dataset=train

    # Inner session (inherits + adds metadata)
    with OTelSession(batch="0") as inner:
        llm.chat.completions.create(...)  # Has experiment=v1, dataset=train, batch=0

        # Even deeper nesting
        with OTelSession(task="math") as deep:
            llm.chat.completions.create(...)  # Has all 4 metadata fields

    # Back to outer
    llm.chat.completions.create(...)  # Has experiment=v1, dataset=train

# All traces accessible from outer session
print(f"Total traces: {len(outer.llm_calls)}")  # 4 traces
```

---

## Conclusion

This design provides a **pragmatic solution** to the distributed session tracking problem:

✅ **Zero breaking changes** - Existing ContextVarSession code unchanged
✅ **Opt-in enhancement** - Users choose OTelSession when needed
✅ **HTTP auto-propagation** - Major improvement for microservices
✅ **Same storage backends** - Compatible with existing infrastructure
✅ **Incremental adoption** - Can be deployed in phases
✅ **Clear migration path** - Simple steps for users to upgrade

**Timeline**: 2-3 weeks for full implementation and testing

**Next Steps**:
1. Review and approve design
2. Create implementation issues/tasks
3. Begin Phase 1 (Core OTelSession class)
