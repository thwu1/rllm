# Distributed Tracing Limitations: Current rLLM Implementation

**Date**: 2025-11-06
**Status**: Critical Gap Analysis
**Related**: `opentelemetry_comparison.md`, `otel_distributed_tracing_explained.md`

## Executive Summary

**Problem**: The current rLLM tracer **does not support distributed tracing**. When agent code runs across multiple processes or services, **session context is lost** and traces become **fragmented**.

**Impact**:
- ❌ Multi-process training loses session context
- ❌ Remote tool execution not traced
- ❌ Microservices lose trace correlation
- ❌ Can't debug cross-service issues

**Solutions**:
1. **Short-term**: Document limitations, provide workarounds
2. **Medium-term**: Add manual context propagation APIs
3. **Long-term**: Integrate OpenTelemetry for automatic distributed tracing

---

## The Problem: Context Doesn't Cross Process Boundaries

### Current Architecture

```python
# rllm/sdk/context.py
import contextvars

_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
```

**Key limitation**: Python's `contextvars` are **process-local**.

- ✅ Works within a single process (thread-safe, async-safe)
- ❌ **Does NOT work across processes**
- ❌ **Does NOT work across network calls**
- ❌ **Does NOT work with multiprocessing**
- ❌ **Does NOT work with Ray actors**

---

## Scenario 1: Multi-Process Training (Ray/Multiprocessing)

### Common Use Case

```python
from rllm.sdk import RLLMClient
import ray

client = RLLMClient()

@ray.remote
def run_agent_episode(task):
    """This runs in a separate Ray worker process."""
    llm = client.get_chat_client()
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": task}]
    )
    return response

# Main process
with client.session("training-run-1", experiment="v2"):
    # Spawn 100 parallel episodes
    futures = [run_agent_episode.remote(task) for task in tasks]
    results = ray.get(futures)
```

### What Happens

**Expected behavior**:
```
All traces tagged with:
- session_id: "training-run-1"
- experiment: "v2"
```

**Actual behavior**:
```
Main process:
  ✅ session_id: "training-run-1"
  ✅ experiment: "v2"

Ray worker processes (separate Python processes):
  ❌ session_id: None
  ❌ experiment: None
  ❌ Context is lost!
```

**Why**:
1. Ray spawns **new Python processes** for workers
2. Each process has its own `contextvars` state
3. Parent process context **is not inherited**
4. Workers have **empty context**

### Impact

```python
# Traces in episodic store:
{
    "session_id": None,  # ❌ Lost!
    "metadata": {},      # ❌ Lost!
    "model": "gpt-4o-mini",
    "input": {...},
    "output": {...}
}
```

**Problems**:
- ❌ Can't group traces by session
- ❌ Can't filter by experiment
- ❌ Can't correlate traces from same training run
- ❌ Can't calculate metrics per session
- ❌ Training data is fragmented

---

## Scenario 2: Remote Tool Execution

### Common Use Case

```python
# Main agent service
from rllm.sdk import RLLMClient
import requests

client = RLLMClient()

def run_agent(question):
    with client.session("agent-session-123", user="alice"):
        # 1. Call LLM (logged with session context)
        llm = client.get_chat_client()
        response = llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}]
        )

        # 2. Agent decides to use a tool
        if response.tool_calls:
            # Call remote tool service
            tool_result = requests.post(
                "http://tool-service/execute",
                json={"tool": "web_search", "query": "..."}
            )

            # 3. Call LLM again with tool result
            final_response = llm.chat.completions.create(...)
```

```python
# Remote tool service (separate service/machine)
from rllm.sdk import RLLMClient

client = RLLMClient()

@app.post("/execute")
def execute_tool(tool_request):
    # This service also calls LLMs for reasoning
    llm = client.get_chat_client()
    reasoning = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Analyze: " + tool_request.query}]
    )

    # Execute actual tool
    result = web_search(tool_request.query)
    return result
```

### What Happens

**Traces**:
```
Main agent service:
  Trace 1:
    ✅ session_id: "agent-session-123"
    ✅ user: "alice"
    model: gpt-4o
    content: "I need to search..."

  Trace 2:
    ✅ session_id: "agent-session-123"
    ✅ user: "alice"
    model: gpt-4o
    content: "Based on tool results..."

Tool service (separate machine):
  Trace 3:
    ❌ session_id: None
    ❌ user: None
    model: gpt-4o-mini
    content: "Analyze: ..."
```

**Problems**:
- ❌ Can't see that Trace 3 is part of same agent session
- ❌ Can't calculate total cost across all services
- ❌ Can't measure end-to-end latency
- ❌ Can't debug multi-service flows
- ❌ Traces are **fragmented across services**

### Workaround (Manual Propagation)

```python
# Main service
def run_agent(question):
    with client.session("agent-session-123", user="alice"):
        # Manually pass session metadata
        session_metadata = {
            "session_id": get_current_session(),
            "metadata": get_current_metadata()
        }

        tool_result = requests.post(
            "http://tool-service/execute",
            json={
                "tool": "web_search",
                "query": "...",
                "_session_metadata": session_metadata  # Pass manually
            }
        )

# Tool service
@app.post("/execute")
def execute_tool(tool_request):
    # Manually restore session context
    if "_session_metadata" in tool_request:
        session_id = tool_request["_session_metadata"]["session_id"]
        metadata = tool_request["_session_metadata"]["metadata"]

        with client.session(session_id, **metadata):
            # Now context is restored!
            llm = client.get_chat_client()
            reasoning = llm.chat.completions.create(...)
```

**Problems with workaround**:
- ⚠️ **Manual and error-prone**
- ⚠️ Must remember to pass metadata everywhere
- ⚠️ Pollutes request payloads
- ⚠️ Brittle (easy to forget)
- ⚠️ Doesn't work for 3rd-party libraries

---

## Scenario 3: Microservices Architecture

### Architecture

```
[API Gateway] → [Agent Orchestrator] → [LLM Service]
                        ↓                     ↓
                   [Tool Router]        [Vector DB Service]
                        ↓
                   [Web Search]
                   [Code Executor]
                   [File System]
```

### Problem

**With current rLLM**:
```
Each service has independent context:

API Gateway:
  ✅ session_id: "user-request-123"

Agent Orchestrator (HTTP call):
  ❌ session_id: None  (context lost!)

LLM Service (HTTP call):
  ❌ session_id: None  (context lost!)

Tool Router (HTTP call):
  ❌ session_id: None  (context lost!)
```

**Result**: **6 separate, unconnected trace fragments**

**Can't answer questions like**:
- ❓ What was the total latency for user request?
- ❓ Which service was the bottleneck?
- ❓ How many LLM calls happened for this request?
- ❓ What was the total cost?
- ❓ Which tools were used?

### With OpenTelemetry

```
All services connected in single trace:

API Gateway:
  trace_id: abc123
  span_id: span1

Agent Orchestrator (auto-propagated):
  trace_id: abc123  (same!)
  parent_span_id: span1
  span_id: span2

LLM Service (auto-propagated):
  trace_id: abc123  (same!)
  parent_span_id: span2
  span_id: span3

Tool Router (auto-propagated):
  trace_id: abc123  (same!)
  parent_span_id: span2
  span_id: span4
```

**Result**: **Single unified trace** showing entire request flow

---

## Scenario 4: Ray + ProcessPoolExecutor (Nested Parallelism)

### Code

```python
from rllm.sdk import RLLMClient
from concurrent.futures import ProcessPoolExecutor
import ray

client = RLLMClient()

def evaluate_episode(episode_id, task):
    """Runs in subprocess."""
    llm = client.get_chat_client()
    response = llm.chat.completions.create(...)
    return response

@ray.remote
class TrainingWorker:
    def run_batch(self, batch_id, tasks):
        """Runs in Ray worker process."""
        # Further parallelize with ProcessPool
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(evaluate_episode, i, task)
                for i, task in enumerate(tasks)
            ]
            results = [f.result() for f in futures]
        return results

# Main process
with client.session("training-run-42", experiment="v3"):
    workers = [TrainingWorker.remote() for _ in range(10)]
    futures = [w.run_batch.remote(i, batch) for i, batch in enumerate(batches)]
    results = ray.get(futures)
```

### Context Loss Layers

```
Main Process:
  ✅ session_id: "training-run-42"
  ✅ experiment: "v3"

Ray Worker Process (Layer 1):
  ❌ session_id: None  (lost at Ray boundary)
  ❌ experiment: None

ProcessPoolExecutor Subprocess (Layer 2):
  ❌ session_id: None  (lost at multiprocessing boundary)
  ❌ experiment: None
```

**Result**: **100% of traces have no session context**

---

## Scenario 5: Async/Await (Works!) vs Threading (Works!) vs Multiprocessing (Breaks!)

### ✅ Async/Await - Works!

```python
import asyncio

async def agent_task():
    with client.session("async-session"):
        llm = client.get_chat_client_async()

        # Parallel async tasks
        tasks = [
            llm.chat.completions.create(...),
            llm.chat.completions.create(...),
            llm.chat.completions.create(...)
        ]
        results = await asyncio.gather(*tasks)
        # ✅ All have session context!
```

**Why it works**: `asyncio` runs in **single process**, contextvars preserved

### ✅ Threading - Works!

```python
import threading

def agent_task():
    llm = client.get_chat_client()
    response = llm.chat.completions.create(...)
    # ✅ Has session context!

with client.session("threaded-session"):
    threads = [
        threading.Thread(target=agent_task)
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

**Why it works**: Threads share **same process**, contextvars preserved

### ❌ Multiprocessing - Breaks!

```python
import multiprocessing

def agent_task(task_id):
    llm = client.get_chat_client()
    response = llm.chat.completions.create(...)
    # ❌ No session context!

with client.session("multiprocess-session"):
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(agent_task, range(100))
        # ❌ All traces missing session context!
```

**Why it breaks**: Each subprocess is a **new Python process** with empty contextvars

---

## Impact on Common RL Workflows

### 1. VERL Training (Multi-Process)

VERL uses Ray for distributed training:

```python
from rllm.sdk import RLLMClient
from verl import PPOTrainer

client = RLLMClient()

with client.session("verl-training-run-1", algo="ppo"):
    trainer = PPOTrainer(
        num_workers=16,  # 16 Ray workers
        num_envs_per_worker=8  # 128 parallel environments
    )
    trainer.train()
```

**Impact**:
- ❌ **All 128 environment traces lose session context**
- ❌ Can't group traces by training run
- ❌ Can't compare episodes from same run
- ❌ Metrics are fragmented

### 2. Agent Evaluation (Parallel Episodes)

```python
from rllm.sdk import RLLMClient
import ray

client = RLLMClient()

@ray.remote
def evaluate_episode(episode_data):
    # Run agent on episode
    ...

with client.session("eval-run-5", dataset="test-split"):
    # Evaluate 1000 episodes in parallel
    futures = [
        evaluate_episode.remote(episode)
        for episode in test_episodes
    ]
    results = ray.get(futures)
    # ❌ All episodes lose session context
```

**Impact**:
- ❌ Can't compute per-run metrics
- ❌ Can't filter by dataset split
- ❌ Results are fragmented

### 3. Multi-Agent Systems

```python
@ray.remote
class Agent:
    def act(self, observation):
        llm = client.get_chat_client()
        action = llm.chat.completions.create(...)
        return action

with client.session("multi-agent-sim-1"):
    agents = [Agent.remote() for _ in range(10)]

    # Simulate 100 timesteps
    for t in range(100):
        actions = ray.get([a.act.remote(obs) for a in agents])
        # ❌ All actions lose session context
```

**Impact**:
- ❌ Can't track which traces belong to which simulation
- ❌ Can't correlate agent behaviors
- ❌ Can't compute simulation-level metrics

---

## Solutions & Workarounds

### Solution 1: Manual Context Passing (Current)

**Approach**: Explicitly pass session metadata

```python
@ray.remote
def run_episode(task, session_metadata):
    """session_metadata passed explicitly."""
    with client.session(**session_metadata):
        # Context restored!
        llm = client.get_chat_client()
        response = llm.chat.completions.create(...)

# Main process
with client.session("training-1", experiment="v2") as session:
    # Get session metadata
    metadata = {
        "session_id": session.session_id,
        **session.metadata
    }

    # Pass to workers
    futures = [
        run_episode.remote(task, metadata)
        for task in tasks
    ]
```

**Pros**:
- ✅ Works today (no changes needed)
- ✅ Explicit and clear

**Cons**:
- ❌ Manual and error-prone
- ❌ Pollutes function signatures
- ❌ Must remember everywhere
- ❌ Doesn't work for 3rd-party code

### Solution 2: Add Context Propagation API (Short-Term)

**Add helper functions**:

```python
# rllm/sdk/context.py
def get_propagation_context() -> dict[str, Any]:
    """Get context dict for manual propagation."""
    return {
        "session_id": _session_id.get(),
        "metadata": _metadata.get() or {}
    }

def set_propagation_context(context: dict[str, Any]) -> None:
    """Restore context from dict."""
    _session_id.set(context.get("session_id"))
    _metadata.set(context.get("metadata", {}))

def propagated_session(context: dict[str, Any]) -> SessionContext:
    """Create session from propagated context."""
    session_id = context.get("session_id")
    metadata = context.get("metadata", {})
    return SessionContext(session_id, **metadata)
```

**Usage**:

```python
# Main process
with client.session("training-1", experiment="v2"):
    # Get context for propagation
    ctx = get_propagation_context()

    # Pass to worker
    futures = [run_episode.remote(task, ctx) for task in tasks]

# Worker
@ray.remote
def run_episode(task, propagation_context):
    # Restore context
    with propagated_session(propagation_context):
        llm = client.get_chat_client()
        response = llm.chat.completions.create(...)
```

**Pros**:
- ✅ Official API (less error-prone)
- ✅ Clearer intent
- ✅ Works today (minimal changes)

**Cons**:
- ⚠️ Still manual
- ⚠️ Still pollutes signatures

### Solution 3: Integrate with Ray's Runtime Context (Medium-Term)

**Use Ray's runtime_context**:

```python
import ray
from ray import runtime_context

# rllm/sdk/context.py
def _get_ray_context() -> dict:
    """Get rLLM context from Ray runtime_context."""
    try:
        ctx = runtime_context.get_runtime_context()
        return ctx.get("rllm_session", {})
    except Exception:
        return {}

def _set_ray_context(context: dict) -> None:
    """Store rLLM context in Ray runtime_context."""
    try:
        ctx = runtime_context.get_runtime_context()
        ctx.set("rllm_session", context)
    except Exception:
        pass
```

**Usage**:

```python
# Automatically propagate via Ray
with client.session("training-1", experiment="v2"):
    # Store in Ray context
    _set_ray_context(get_propagation_context())

    # Ray workers automatically have access
    futures = [run_episode.remote(task) for task in tasks]

@ray.remote
def run_episode(task):
    # Automatically restore from Ray context
    ctx = _get_ray_context()
    with propagated_session(ctx):
        llm = client.get_chat_client()
        response = llm.chat.completions.create(...)
```

**Pros**:
- ✅ Automatic for Ray workflows
- ✅ No signature pollution
- ✅ Works with existing Ray code

**Cons**:
- ⚠️ Ray-specific (doesn't help with HTTP services)
- ⚠️ Still manual for other frameworks

### Solution 4: OpenTelemetry Integration (Long-Term)

**Use OTel's context propagation**:

```python
# rllm/sdk/client.py
from opentelemetry import trace, baggage
from opentelemetry.propagate import inject, extract

class RLLMClient:
    def session(self, session_id=None, **metadata):
        # Store in OTel baggage (auto-propagates!)
        ctx = baggage.set_baggage("rllm.session_id", session_id)
        for key, value in metadata.items():
            ctx = baggage.set_baggage(f"rllm.{key}", value, context=ctx)

        return SessionContext(session_id, **metadata)
```

**Usage** (automatic!):

```python
# Main process
with client.session("training-1", experiment="v2"):
    # OTel automatically propagates via HTTP headers
    response = requests.post("http://service-b/api")

    # OTel automatically propagates via Ray
    futures = [run_episode.remote(task) for task in tasks]

# Worker (separate process/machine)
@ray.remote
def run_episode(task):
    # OTel automatically restores context!
    session_id = baggage.get_baggage("rllm.session_id")
    experiment = baggage.get_baggage("rllm.experiment")

    # Or use helper:
    with client.restore_session_from_context():
        llm = client.get_chat_client()
        response = llm.chat.completions.create(...)
```

**Pros**:
- ✅ **Automatic** for HTTP calls
- ✅ Standard W3C headers
- ✅ Works across any service
- ✅ Rich ecosystem

**Cons**:
- ⚠️ Requires OTel dependency
- ⚠️ More complex setup
- ⚠️ Ray integration requires custom setup

---

## Comparison: Current vs OTel

### Context Propagation

| Scenario | Current rLLM | With OTel |
|----------|--------------|-----------|
| **Single process** | ✅ Automatic | ✅ Automatic |
| **Threading** | ✅ Automatic | ✅ Automatic |
| **Async/await** | ✅ Automatic | ✅ Automatic |
| **Multiprocessing** | ❌ Lost | ⚠️ Manual (subprocess args) |
| **Ray actors** | ❌ Lost | ⚠️ Needs Ray integration |
| **HTTP services** | ❌ Lost | ✅ **Automatic** (headers) |
| **gRPC services** | ❌ Lost | ✅ **Automatic** (metadata) |
| **Kafka/message queues** | ❌ Lost | ✅ **Automatic** (message headers) |

### Trace Correlation

| Feature | Current rLLM | With OTel |
|---------|--------------|-----------|
| **Session grouping** | ✅ session_id | ✅ trace_id |
| **Parent-child spans** | ❌ Not supported | ✅ Automatic |
| **Cross-service correlation** | ❌ Not supported | ✅ Automatic |
| **End-to-end latency** | ❌ Manual | ✅ Automatic |
| **Distributed debugging** | ❌ Hard | ✅ Easy |

---

## Recommendations

### For Current Users

**If you're using multi-process/service architectures**:

1. **Document the limitation** ✅ (this doc)
2. **Use manual propagation** (Solution 1 or 2)
3. **Consider OTel** for production systems

**If you're using single-process training**:
- ✅ Current implementation works fine
- No action needed

### For rLLM Project

**Short-term** (0-3 months):
1. ✅ Add `get_propagation_context()` / `propagated_session()` APIs
2. ✅ Document manual propagation patterns
3. ✅ Add examples for Ray/multiprocessing

**Medium-term** (3-6 months):
1. 🔄 Add optional OTel baggage integration
2. 🔄 Provide Ray runtime_context helper
3. 🔄 Add HTTP header propagation helpers

**Long-term** (6-12 months):
1. 🔄 Full OTel integration (optional)
2. 🔄 Automatic distributed tracing
3. 🔄 Hybrid mode: custom for RL data, OTel for observability

---

## Code Examples: Handling Each Scenario

### Example 1: Ray Training with Manual Propagation

```python
from rllm.sdk import RLLMClient
from rllm.sdk.context import get_propagation_context, propagated_session
import ray

client = RLLMClient()

@ray.remote
def train_episode(episode_id, task, session_context):
    """Worker function - restores context."""
    with propagated_session(session_context):
        llm = client.get_chat_client()
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}]
        )
        return response

def train():
    """Main training loop."""
    with client.session("verl-run-1", algo="ppo", lr=0.0001):
        # Get context for propagation
        ctx = get_propagation_context()

        # Spawn workers with context
        futures = [
            train_episode.remote(i, task, ctx)
            for i, task in enumerate(tasks)
        ]
        results = ray.get(futures)
        # ✅ All traces have session context!
```

### Example 2: HTTP Services with Manual Propagation

```python
# Service A (API Gateway)
@app.post("/agent/run")
def run_agent(request):
    with client.session("agent-123", user=request.user_id):
        # Get context
        ctx = get_propagation_context()

        # Pass via custom header
        response = requests.post(
            "http://agent-service/execute",
            json={"task": request.task},
            headers={"X-RLLM-Context": json.dumps(ctx)}
        )
        return response.json()

# Service B (Agent Service)
@app.post("/execute")
def execute(request):
    # Extract context from header
    ctx_header = request.headers.get("X-RLLM-Context")
    if ctx_header:
        ctx = json.loads(ctx_header)
        with propagated_session(ctx):
            # ✅ Context restored!
            llm = client.get_chat_client()
            response = llm.chat.completions.create(...)
    else:
        # ❌ No context
        response = llm.chat.completions.create(...)
```

### Example 3: Multiprocessing with Manual Propagation

```python
from multiprocessing import Pool
from rllm.sdk.context import get_propagation_context, propagated_session

def evaluate_task(args):
    """Worker function."""
    task_id, task_data, session_context = args

    with propagated_session(session_context):
        llm = client.get_chat_client()
        result = llm.chat.completions.create(...)
        return result

def evaluate_dataset():
    with client.session("eval-run-10", split="test"):
        ctx = get_propagation_context()

        # Prepare args with context
        args = [(i, task, ctx) for i, task in enumerate(tasks)]

        with Pool(processes=10) as pool:
            results = pool.map(evaluate_task, args)
            # ✅ All traces have session context!
```

---

## Testing Distributed Tracing

### Test 1: Verify Context Loss

```python
import multiprocessing
from rllm.sdk import RLLMClient
from rllm.sdk.context import get_current_session

client = RLLMClient()

def worker():
    # Should return None (context lost)
    session = get_current_session()
    print(f"Worker session: {session}")

with client.session("test-session"):
    parent_session = get_current_session()
    print(f"Parent session: {parent_session}")  # "test-session"

    p = multiprocessing.Process(target=worker)
    p.start()
    p.join()
    # Prints: Worker session: None
```

### Test 2: Verify Manual Propagation Works

```python
from rllm.sdk.context import get_propagation_context, propagated_session

def worker(ctx):
    with propagated_session(ctx):
        session = get_current_session()
        print(f"Worker session: {session}")

with client.session("test-session"):
    ctx = get_propagation_context()

    p = multiprocessing.Process(target=worker, args=(ctx,))
    p.start()
    p.join()
    # Prints: Worker session: test-session
```

---

## Conclusion

### Critical Findings

1. **Current rLLM tracer does NOT support distributed tracing**
2. **Context is lost** across:
   - Process boundaries (Ray, multiprocessing)
   - Network boundaries (HTTP services, gRPC)
   - Message queues (Kafka, RabbitMQ)

3. **Impact on RL workflows**:
   - Multi-process training loses session context
   - Parallel evaluation loses metadata
   - Microservices can't correlate traces

### Solutions

**Short-term**:
- Use manual propagation APIs
- Document patterns
- Provide examples

**Long-term**:
- Integrate OpenTelemetry for automatic propagation
- Support hybrid mode (custom + OTel)

### The Trade-off

**Current approach**:
- ✅ Simple for single-process
- ❌ Breaks for multi-process/service

**With OTel**:
- ✅ Automatic distributed tracing
- ⚠️ More complexity

For **RL training** (often multi-process), this is a **significant limitation** that should be addressed.
