# Context Variables and Ray Execution

## The Problem

When you pass a function to Ray for remote execution, **context variables do NOT automatically propagate** to the Ray worker process.

## Example Scenario

```python
from contextvars import ContextVar
import ray

# Context variable
session_id_var: ContextVar[str] = ContextVar("session_id", default=None)

def func_a():
    """Function that depends on context variable."""
    session_id = session_id_var.get()
    print(f"func_a sees session_id: {session_id}")
    return session_id

def execute_with_ray(func, session_id: str):
    """Execute function using Ray, trying to set context before execution."""

    # Set context variable in the main process
    session_id_var.set(session_id)
    print(f"Main process set session_id: {session_id}")

    # Execute via Ray
    @ray.remote
    def ray_wrapper():
        # Try to set context in Ray worker
        session_id_var.set(session_id)
        return func()

    result = ray.get(ray_wrapper.remote())
    return result

# Usage
ray.init()
execute_with_ray(func_a, "session-123")
```

## What Actually Happens

### ❌ Attempt 1: Set context in main process

```python
def execute_with_ray(func, session_id: str):
    # Set in main process
    session_id_var.set(session_id)  # ✅ Works in main process

    @ray.remote
    def ray_wrapper():
        return func()  # ❌ session_id_var.get() returns None!

    return ray.get(ray_wrapper.remote())
```

**Result**: `func_a` sees `session_id: None` because the Ray worker is a **different process** with its own context variable storage.

### ❌ Attempt 2: Set context in Ray wrapper

```python
def execute_with_ray(func, session_id: str):
    @ray.remote
    def ray_wrapper():
        session_id_var.set(session_id)  # Set in Ray worker
        return func()  # Still might not work!

    return ray.get(ray_wrapper.remote())
```

**Result**: This can work IF `func` directly calls `session_id_var.get()`. But if `func` calls other functions that expect the context to already be set, it breaks the call stack context.

## Why Context Variables Don't Cross Processes

Python's `contextvars` module uses **thread-local storage** that's tied to the current **OS process**:

1. **Main process** has its own context variable storage
2. **Ray worker process** has its own separate context variable storage
3. When Ray serializes and sends your function to the worker, **context variables are NOT serialized**

```
┌─────────────────────┐         ┌──────────────────────┐
│   Main Process      │         │   Ray Worker Process │
│                     │         │                      │
│  session_id_var =   │         │  session_id_var =    │
│    "session-123"    │  ----X  │    None (default)    │
│                     │         │                      │
│  execute_with_ray() │         │  ray_wrapper()       │
│         │           │         │       │              │
│         └──────────────────────────>  │              │
│            serialize func     │       ├─ func_a()    │
│                               │       │  └─ get() → None
└─────────────────────────────┘         └──────────────────────┘
     Context isolated                      Context isolated
```

## Solutions

### ✅ Solution 1: Explicit Parameter Passing

Pass the context value as an explicit parameter:

```python
def func_a(session_id: str):  # ← Explicit parameter
    print(f"func_a sees session_id: {session_id}")
    return session_id

@ray.remote
def ray_wrapper(func, session_id: str):
    return func(session_id)  # ← Pass explicitly

# Usage
result = ray.get(ray_wrapper.remote(func_a, "session-123"))
```

**Pros**: Simple, explicit, works reliably
**Cons**: Requires changing function signatures

### ✅ Solution 2: Use rLLM's `@entrypoint` Decorator + `_metadata`

The rLLM SDK provides a pattern for this:

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

@client.entrypoint
def func_a():
    """Function uses contextvars internally via SDK."""
    session_id = get_current_session()
    print(f"func_a sees session_id: {session_id}")
    return session_id

@ray.remote
def ray_wrapper(func, metadata: dict):
    # Call with special _metadata kwarg
    return func(_metadata=metadata)

# Usage
metadata = {"session_id": "session-123", "experiment": "v1"}
result = ray.get(ray_wrapper.remote(func_a, metadata))
```

**How it works**:
1. `@client.entrypoint` wraps `func_a` to accept `_metadata` kwarg
2. Inside the wrapper, it creates a new session context with that metadata
3. Then executes the original function within that context
4. All contextvars are properly set before function execution

**Code reference**: See `rllm/sdk/client.py:77-161` - the `entrypoint()` decorator implementation.

### ✅ Solution 3: Manual Context Restoration in Ray Worker

Set context variables at the start of the Ray function:

```python
@ray.remote
def ray_wrapper(func, session_id: str):
    # Restore context in worker process
    session_id_var.set(session_id)

    # Now execute function
    return func()

# Usage
result = ray.get(ray_wrapper.remote(func_a, "session-123"))
```

**Pros**: Works without changing inner function
**Cons**: Need to manually manage all context variables

### ✅ Solution 4: Ray Runtime Context (Advanced)

Use Ray's runtime context feature to propagate metadata:

```python
import ray
from ray import runtime_context

@ray.remote
def ray_wrapper(func):
    # Get metadata from Ray runtime context
    ctx = runtime_context.get_runtime_context()
    metadata = ctx.get("rllm_metadata", {})
    session_id = metadata.get("session_id")

    # Restore context
    session_id_var.set(session_id)
    return func()

# Set runtime context when submitting
ray.get(ray_wrapper.options(
    runtime_env={"env_vars": {"rllm_metadata": json.dumps({"session_id": "session-123"})}}
).remote(func_a))
```

**Pros**: Clean separation, Ray-native approach
**Cons**: More complex, requires Ray 2.0+

## Real-World rLLM Example

This is exactly the problem in RL training with Ray:

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

# User's agent function
def my_agent(task):
    llm = client.get_chat_client()
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content

# ❌ BROKEN: Context lost in Ray workers
with client.session("training-run-1", experiment="v2"):
    @ray.remote
    def train_episode(task):
        return my_agent(task)  # ← session_id is None!

    results = ray.get([train_episode.remote(task) for task in tasks])

# ✅ WORKING: Use @entrypoint with _metadata
@client.entrypoint
def my_agent(task):
    llm = client.get_chat_client()
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content

@ray.remote
def train_episode(task, metadata):
    return my_agent(task, _metadata=metadata)

# Prepare metadata once
metadata = {"session_id": "training-run-1", "experiment": "v2"}

# Execute with explicit metadata passing
results = ray.get([
    train_episode.remote(task, metadata)
    for task in tasks
])
```

## Summary

| Approach | Works? | Complexity | Recommended? |
|----------|--------|------------|--------------|
| Set context in main process | ❌ No | Low | Never |
| Set context in Ray wrapper | ⚠️ Maybe | Medium | Sometimes |
| Explicit parameters | ✅ Yes | Low | Yes (simple cases) |
| `@entrypoint` + `_metadata` | ✅ Yes | Medium | **Yes (rLLM SDK)** |
| Ray runtime context | ✅ Yes | High | Advanced use |

## Key Takeaway

**Context variables are process-local and do NOT automatically propagate to Ray workers.** You must explicitly pass metadata and restore context in the Ray worker process.

The rLLM SDK's `@entrypoint` decorator pattern is specifically designed to solve this problem by accepting `_metadata` kwargs that reconstruct the context in the worker process.

## References

- [distributed_tracing_limitations.md](./distributed_tracing_limitations.md) - Full analysis of distributed tracing limitations
- [rllm/sdk/client.py:77-161](../../client.py) - `@entrypoint` decorator implementation
- [rllm/sdk/test_entrypoint.py](../../test_entrypoint.py) - Tests showing `_metadata` usage
