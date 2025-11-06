# How OpenTelemetry Handles Distributed Tracing

**Date**: 2025-11-06
**Topic**: Understanding OTel's distributed tracing mechanism

## The Problem: Tracing Across Services

Imagine this scenario:

```
User Request
    ↓
[API Gateway] --HTTP--> [LLM Service] --HTTP--> [vLLM Backend]
                             ↓
                        [Vector DB]
```

**Question**: How do you connect traces from all 4 services into a single logical request?

**Without distributed tracing**:
- Each service logs independently
- No way to correlate logs across services
- Can't see end-to-end latency
- Debugging is a nightmare

**With OTel distributed tracing**:
- Single trace ID spans all services
- Parent-child span relationships
- End-to-end visibility
- Automatic context propagation

---

## Core Concepts

### 1. Trace Context

Every trace has a **Trace Context** that gets propagated between services:

```python
{
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",     # Unique ID for entire request
    "span_id": "00f067aa0ba902b7",                      # Current span ID
    "trace_flags": "01",                                # Sampling decision
    "trace_state": "vendor1=value1,vendor2=value2"      # Vendor-specific data
}
```

**Key principle**: This context is passed from service to service via **HTTP headers**.

### 2. W3C Trace Context Standard

OTel uses the **W3C Trace Context** standard for propagation:

```http
# HTTP Request Headers
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate: vendor1=value1,vendor2=value2
```

**Format breakdown** (`traceparent`):
```
00-{trace_id}-{parent_span_id}-{flags}
│  │          │                 │
│  │          │                 └─ Sampling flags (01 = sampled)
│  │          └─────────────────── Parent span ID (64-bit hex)
│  └────────────────────────────── Trace ID (128-bit hex)
└───────────────────────────────── Version (currently 00)
```

### 3. Spans (Parent-Child Relationships)

A **trace** is composed of multiple **spans**:

```
Trace: 4bf92f3577b34da6a3ce929d0e0e4736
│
├─ Span: [API Gateway]
│   ├─ Span: [LLM Service]
│   │   ├─ Span: [vLLM Backend]
│   │   └─ Span: [Vector DB Query]
│   └─ Span: [Response Processing]
```

Each span has:
- **trace_id**: Same across entire trace
- **span_id**: Unique to this span
- **parent_span_id**: Links to parent (creates hierarchy)
- **attributes**: Metadata (model, tokens, etc.)
- **events**: Timestamped logs within span
- **start/end time**: Duration

### 4. Baggage (Metadata Propagation)

**Baggage** allows you to propagate **custom metadata** across services:

```python
# Service A sets baggage
baggage.set_baggage("user_id", "alice")
baggage.set_baggage("experiment", "v2")

# HTTP headers automatically include:
# baggage: user_id=alice,experiment=v2

# Service B reads baggage
user_id = baggage.get_baggage("user_id")  # "alice"
experiment = baggage.get_baggage("experiment")  # "v2"
```

**Use cases**:
- Session IDs
- User IDs
- A/B test variants
- Feature flags
- Custom metadata (like rLLM's session metadata!)

---

## How It Works: Step-by-Step

### Scenario: User asks a question through your LLM app

```
[User Browser] → [API Gateway] → [LLM Service] → [OpenAI API]
                                      ↓
                                 [Vector DB]
```

### Step 1: Request Starts (API Gateway)

**Service**: API Gateway
**Action**: Creates root span

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# User makes request
@app.route("/ask")
def ask_question():
    # OTel automatically creates root span
    with tracer.start_as_current_span("api_gateway.ask") as span:
        span.set_attribute("http.method", "POST")
        span.set_attribute("http.route", "/ask")

        # Make HTTP call to LLM service
        response = requests.post(
            "http://llm-service/generate",
            json={"question": "What is AI?"},
            # OTel auto-injects headers here! ↓
        )
        return response.json()
```

**OTel automatically injects headers**:
```http
POST /generate HTTP/1.1
Host: llm-service
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate:
```

**Trace so far**:
```
Trace: 4bf92f3577b34da6a3ce929d0e0e4736
└─ Span: api_gateway.ask (span_id: 00f067aa0ba902b7)
```

### Step 2: LLM Service Receives Request

**Service**: LLM Service
**Action**: Extracts parent context, creates child span

```python
from opentelemetry import trace
from opentelemetry.propagate import extract

tracer = trace.get_tracer(__name__)

@app.route("/generate")
def generate():
    # OTel automatically extracts traceparent from headers!
    # No manual code needed - middleware does this

    with tracer.start_as_current_span("llm_service.generate") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4")

        # 1. Query vector DB
        embeddings = query_vector_db(question)

        # 2. Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question}]
            # OTel auto-injects headers here too! ↓
        )

        span.set_attribute("gen_ai.usage.prompt_tokens", response.usage.prompt_tokens)
        return response
```

**Headers sent to OpenAI**:
```http
POST /v1/chat/completions HTTP/1.1
Host: api.openai.com
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-5e107d8a0ba902c8-01
                                               ↑ new span_id (child)
```

**Trace now**:
```
Trace: 4bf92f3577b34da6a3ce929d0e0e4736
└─ Span: api_gateway.ask (00f067aa0ba902b7)
    └─ Span: llm_service.generate (5e107d8a0ba902c8) ← NEW
        ├─ Span: vector_db.query (child of generate)
        └─ Span: openai.chat.completions.create (child of generate)
```

### Step 3: Vector DB Query (Parallel)

```python
def query_vector_db(query):
    # This automatically becomes a child span
    with tracer.start_as_current_span("vector_db.query") as span:
        span.set_attribute("db.system", "pinecone")
        span.set_attribute("db.operation", "search")

        results = pinecone_client.query(query)
        span.set_attribute("db.results.count", len(results))
        return results
```

### Step 4: OpenAI API Call

```python
# OpenLLMetry auto-instruments this
response = openai.chat.completions.create(...)

# Creates span automatically:
# - Parent: llm_service.generate
# - Name: openai.chat.completions.create
# - Attributes: model, tokens, latency, etc.
```

### Final Trace Structure

```
Trace ID: 4bf92f3577b34da6a3ce929d0e0e4736

Timeline (left to right):
|─────────────────────────────────────────────────────|
  api_gateway.ask (350ms)
  |───────────────────────────────────────────────|
    llm_service.generate (320ms)
    |──────────|            |──────────────────|
      vector_db.query        openai.chat.completions
      (50ms)                 (250ms)

Hierarchy:
└─ api_gateway.ask
    └─ llm_service.generate
        ├─ vector_db.query
        └─ openai.chat.completions.create
```

**What you see in the UI** (Jaeger/Grafana):
```
┌─────────────────────────────────────────────────────┐
│ Trace: 4bf92f3577b34da6a3ce929d0e0e4736              │
│ Duration: 350ms                                      │
├─────────────────────────────────────────────────────┤
│ ▼ api_gateway.ask (350ms)                           │
│   ▼ llm_service.generate (320ms)                    │
│     ▶ vector_db.query (50ms)                        │
│     ▶ openai.chat.completions.create (250ms)        │
│       • model: gpt-4                                 │
│       • prompt_tokens: 100                           │
│       • completion_tokens: 50                        │
└─────────────────────────────────────────────────────┘
```

---

## How Context Propagation Works

### 1. Automatic Injection (Outgoing Requests)

OTel **automatically injects** trace context into HTTP headers:

```python
# You write this:
response = requests.get("http://service-b/api")

# OTel automatically does this behind the scenes:
headers = {
    "traceparent": f"00-{trace_id}-{span_id}-{flags}",
    "tracestate": f"{vendor_data}"
}
response = requests.get("http://service-b/api", headers=headers)
```

**How it works**:
1. OTel instruments HTTP libraries (requests, httpx, urllib, etc.)
2. Before sending request, extracts current trace context
3. Injects context into HTTP headers
4. Request sent with headers

### 2. Automatic Extraction (Incoming Requests)

OTel **automatically extracts** trace context from HTTP headers:

```python
# Incoming request has headers:
# traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

# You write this:
@app.route("/api")
def handler():
    with tracer.start_as_current_span("handler"):
        # ... your code ...
        pass

# OTel automatically:
# 1. Extracts traceparent from request headers
# 2. Sets trace_id = 4bf92f3577b34da6a3ce929d0e0e4736
# 3. Sets parent_span_id = 00f067aa0ba902b7
# 4. Creates new span_id for this span
# 5. Links spans in parent-child relationship
```

**How it works**:
1. OTel instruments web frameworks (Flask, FastAPI, Django, etc.)
2. Before handling request, extracts trace context from headers
3. Sets current context for this request
4. All spans created use this context

### 3. Context Storage (Thread-Local)

OTel uses **context variables** (similar to rLLM's approach!):

```python
# Under the hood (simplified):
import contextvars

_current_context = contextvars.ContextVar("otel_context")

def set_context(context):
    _current_context.set(context)

def get_context():
    return _current_context.get()

# When you create a span:
def start_span(name):
    parent_context = get_context()  # Get current context
    new_span = Span(
        trace_id=parent_context.trace_id,  # Same trace ID
        parent_span_id=parent_context.span_id,  # Link to parent
        span_id=generate_new_span_id()  # New span ID
    )
    set_context(new_span)  # Set as current
    return new_span
```

**Benefits of contextvars**:
- Thread-safe
- Async-safe
- No manual passing needed
- Works across function calls

---

## Propagators: How OTel Supports Multiple Formats

OTel supports multiple propagation formats via **Propagators**:

### 1. W3C Trace Context (Default)

```http
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
tracestate: vendor1=value1
```

### 2. B3 Propagation (Zipkin)

```http
X-B3-TraceId: 4bf92f3577b34da6a3ce929d0e0e4736
X-B3-SpanId: 00f067aa0ba902b7
X-B3-Sampled: 1
```

### 3. Jaeger Propagation

```http
uber-trace-id: 4bf92f3577b34da6a3ce929d0e0e4736:00f067aa0ba902b7:0:1
```

### Configuration

```python
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3Format
from opentelemetry.propagators.jaeger import JaegerPropagator

# Use multiple propagators
from opentelemetry.propagators.composite import CompositePropagator
set_global_textmap(CompositePropagator([
    W3CTraceContextPropagator(),
    B3Format(),
    JaegerPropagator()
]))
```

**Benefit**: Interoperability with existing tracing systems!

---

## Comparison with rLLM's Approach

### Current rLLM Implementation

```python
# Context stored in contextvars (similar to OTel!)
from rllm.sdk import RLLMClient

client = RLLMClient()

# Service A
with client.session("session-123", experiment="v1"):
    # Make HTTP call to Service B
    response = requests.post("http://service-b/api", json={...})
    # ❌ Session metadata NOT propagated to Service B
```

**Problem**: No automatic propagation between services!

### With OpenTelemetry

```python
from opentelemetry import trace, baggage

tracer = trace.get_tracer(__name__)

# Service A
with tracer.start_as_current_span("service_a"):
    baggage.set_baggage("session_id", "session-123")
    baggage.set_baggage("experiment", "v1")

    # Make HTTP call to Service B
    response = requests.post("http://service-b/api", json={...})
    # ✅ Session metadata automatically sent via baggage header!
    # baggage: session_id=session-123,experiment=v1

# Service B (separate process/machine)
@app.route("/api")
def handler():
    # ✅ Can access session metadata!
    session_id = baggage.get_baggage("session_id")  # "session-123"
    experiment = baggage.get_baggage("experiment")  # "v1"
```

---

## Real-World Example: LLM Agent System

### Architecture

```
[User] → [API Gateway] → [Agent Orchestrator] → [LLM Service]
                              ↓                       ↓
                         [Tool Executor]        [Vector DB]
                              ↓
                         [Web Search API]
```

### Trace Flow

```python
# 1. API Gateway receives request
@app.post("/agent/run")
async def run_agent(question: str):
    with tracer.start_as_current_span("api.run_agent") as span:
        span.set_attribute("question", question)

        # Set user context in baggage
        baggage.set_baggage("user_id", "alice")
        baggage.set_baggage("session_id", "sess-123")

        # Call agent orchestrator
        result = await agent_client.execute(question)
        return result

# 2. Agent Orchestrator (separate service)
async def execute(question: str):
    # OTel automatically extracts parent context from headers!
    with tracer.start_as_current_span("agent.execute") as span:
        # Baggage is automatically available!
        user_id = baggage.get_baggage("user_id")  # "alice"
        session_id = baggage.get_baggage("session_id")  # "sess-123"

        span.set_attribute("user_id", user_id)

        # Step 1: Query vector DB
        context = await query_vector_db(question)

        # Step 2: Generate LLM response
        response = await llm_service.generate(question, context)

        # Step 3: Execute tool if needed
        if response.needs_tool:
            tool_result = await tool_executor.run(response.tool_call)
            response = await llm_service.generate(question, tool_result)

        return response

# 3. Tool Executor (separate service)
async def run(tool_call):
    with tracer.start_as_current_span("tool.execute") as span:
        # Session context still available!
        session_id = baggage.get_baggage("session_id")

        span.set_attribute("tool.name", tool_call.name)
        span.set_attribute("session_id", session_id)

        if tool_call.name == "web_search":
            result = await web_search_api.search(tool_call.args)

        return result
```

### Resulting Trace

```
Trace: 7a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p

api.run_agent (1200ms)
├─ agent.execute (1150ms)
│   ├─ vector_db.query (100ms)
│   ├─ llm_service.generate (500ms)
│   │   └─ openai.chat.completions.create (480ms)
│   ├─ tool.execute (400ms)
│   │   └─ web_search_api.search (380ms)
│   └─ llm_service.generate (300ms)
│       └─ openai.chat.completions.create (280ms)

Baggage (propagated to ALL spans):
- user_id: alice
- session_id: sess-123
```

**What you see**:
- **End-to-end visibility**: See entire request flow across services
- **Latency breakdown**: 480ms in first LLM call, 380ms in web search
- **User context**: All spans tagged with user_id and session_id
- **Easy debugging**: Click on any span to see details

---

## Benefits Over Custom Implementation

### 1. Cross-Service Visibility

**Current rLLM**:
```
Service A: [Trace 1] → ❌ Lost context → Service B: [Trace 2]
```

**With OTel**:
```
Service A: [Trace 1] → ✅ Propagates → Service B: [Trace 1, Span 2]
```

### 2. Automatic Instrumentation

**Current rLLM**:
```python
# Must manually log everywhere
def my_function():
    tracer.log_llm_call(...)
    result = requests.get(...)  # ❌ Not traced
    return result
```

**With OTel**:
```python
# HTTP calls auto-traced
def my_function():
    with tracer.start_as_current_span("my_function"):
        result = requests.get(...)  # ✅ Auto-traced!
        return result
```

### 3. Standard Format (Interoperability)

**Current rLLM**:
- Custom format
- Only works with Episodic
- Can't integrate with other tools

**With OTel**:
- Standard W3C format
- Works with 50+ backends
- Integrates with existing monitoring tools

### 4. Multi-Service Debugging

**Scenario**: LLM call is slow (5 seconds)

**Current rLLM**:
- Check logs in each service manually
- Hard to correlate across services
- No visibility into which service is slow

**With OTel**:
- Single trace shows entire flow
- See exactly where the 5 seconds was spent:
  - API Gateway: 10ms
  - Agent Orchestrator: 50ms
  - LLM Service: 4900ms ← **This is the bottleneck!**
    - Vector DB: 100ms
    - OpenAI API: 4800ms ← **OpenAI is slow!**

---

## Implementation Example: Adding Distributed Tracing to rLLM

### Option 1: Keep Custom + Add OTel Headers

```python
# Add to rllm/sdk/session.py
from opentelemetry.propagate import inject

class SessionContext:
    def __enter__(self):
        # Current: Set contextvars
        self.s_token = _session_id.set(self.session_id)
        self.m_token = _metadata.set(merged)

        # NEW: Also inject OTel context
        carrier = {}
        inject(carrier)  # Adds traceparent/tracestate
        self.otel_headers = carrier

        return self

# Usage: HTTP client can now propagate context
import requests
from rllm.sdk.context import get_current_session_headers

with client.session("sess-123"):
    # Get OTel headers
    headers = get_current_session_headers()

    # Make request with headers
    response = requests.post(
        "http://service-b/api",
        headers=headers  # Propagates trace context!
    )
```

### Option 2: Full OTel Integration (Hybrid)

```python
# rllm/sdk/client.py
from opentelemetry import trace, baggage

class RLLMClient:
    def session(self, session_id=None, **metadata):
        # Create OTel span
        tracer = trace.get_tracer(__name__)
        span = tracer.start_span("rllm.session")

        # Set baggage for metadata
        ctx = baggage.set_baggage("session_id", session_id)
        for key, value in metadata.items():
            ctx = baggage.set_baggage(key, value, context=ctx)

        # Also set contextvars (for backwards compat)
        return SessionContext(session_id, **metadata, otel_span=span)
```

---

## Summary: How OTel Distributed Tracing Works

### Key Mechanisms

1. **W3C Trace Context Headers**: Standard format for propagation
   ```
   traceparent: 00-{trace_id}-{parent_span_id}-{flags}
   baggage: key1=value1,key2=value2
   ```

2. **Automatic Injection/Extraction**: OTel instruments HTTP libraries
   - Outgoing: Injects headers automatically
   - Incoming: Extracts headers automatically

3. **Context Variables**: Thread-safe storage (like rLLM!)
   - Current trace_id, span_id, baggage
   - Propagates through function calls

4. **Parent-Child Spans**: Builds trace hierarchy
   - Same trace_id across all services
   - Each span has parent_span_id
   - Creates tree structure

### Benefits

✅ **End-to-end visibility**: Single trace across all services
✅ **Automatic**: No manual header management
✅ **Standard**: W3C format, works everywhere
✅ **Debugging**: See exactly where time is spent
✅ **Metadata propagation**: Baggage carries session/user data

### Complexity Cost

⚠️ **Must instrument all services** with OTel
⚠️ **Need OTel collector** to aggregate traces
⚠️ **Need backend** (Jaeger/Tempo/etc.) to store/visualize
⚠️ **Learning curve**: Spans, context, propagators

---

## Conclusion

**How OTel handles distributed tracing**:
1. **Propagates context** via standard HTTP headers (traceparent)
2. **Links spans** via parent_span_id to build trace tree
3. **Uses contextvars** to store current context (like rLLM!)
4. **Auto-instruments** HTTP clients/servers to inject/extract headers

**Why it's powerful**:
- Single trace spans multiple services/machines
- Automatic context propagation (no manual passing)
- Standard format (interoperability)
- Rich ecosystem (50+ backends)

**Why it's complex**:
- Need OTel SDK + instrumentation + collector + backend
- Must instrument all services
- Operational overhead
- Learning curve

**For rLLM**: Could add OTel support **optionally** to enable distributed tracing for users who need it, while keeping the simpler custom implementation for single-service RL training.
