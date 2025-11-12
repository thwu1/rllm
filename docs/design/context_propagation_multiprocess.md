# Context Variable Propagation in Multiprocess Execution

## Problem Statement

The Run Facade design proposes a `ProcessExecutor` for isolating GPU/long-running workloads. However, **Python's `contextvars` do NOT automatically propagate to child processes**, which breaks the session management system used by `RLLMClient`.

## Current Session Management Architecture

The RLLM SDK uses Python's `contextvars` for automatic session tracking:

```python
# rllm/sdk/context.py
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
```

This enables clean usage:

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

with client.session("my-session", experiment="v1"):
    # All LLM calls automatically get session_id="my-session"
    # and metadata={"experiment": "v1"}
    result = run_agent(tasks)
```

The tracer automatically injects session context:

```python
# episodic/tracing.py
def log_llm_call(..., session_id=None):
    if session_id is None:
        session_id = get_current_session()  # Auto-inject from context
```

## Multiprocessing Behavior

### Test Results

| Method | Behavior | Context Inherited? |
|--------|----------|-------------------|
| **spawn** (default on Windows/macOS) | Creates fresh Python interpreter | ❌ **NO** |
| **fork** (default on Unix) | Copies parent memory | ✅ **YES** (but unreliable) |
| **Explicit dict passing** | Pass context as argument | ✅ **YES** (reliable) |
| **copy_context()** | Pass Context object | ❌ **NOT PICKLABLE** |

### Spawn Method (Recommended for GPU isolation)

```python
# Parent process
session_var.set("parent-session-123")
print(session_var.get())  # "parent-session-123"

# Child process (spawn)
print(session_var.get())  # None ❌
```

**Result**: Context variables are **NOT inherited** with spawn.

### Fork Method (Unix only)

```python
# Parent process
session_var.set("parent-session-456")
print(session_var.get())  # "parent-session-456"

# Child process (fork)
print(session_var.get())  # "parent-session-456" ✅
```

**Result**: Context variables **ARE inherited** with fork, but:
- ⚠️ Fork is **not available on Windows**
- ⚠️ Fork is **unsafe with CUDA** (GPU state corruption)
- ⚠️ The design document specifies **spawn** for GPU isolation

## Impact on Run Facade Design

The Run Facade design document states:

> **ProcessExecutor (spawn)**: isolates CUDA/memory; ideal for long or GPU‑heavy jobs

This creates a **critical problem**:

1. User sets session context: `with client.session("my-session"):`
2. Facade spawns child process for GPU isolation
3. **Session context is lost** in child process
4. All LLM calls in child process have `session_id=None`
5. Traces are not grouped correctly ❌

## Solutions

### Option 1: Explicit Context Passing (Recommended)

**Approach**: Extract context in parent, pass as dict to child, restore in child.

```python
# In RunService.run()
def run(args: dict) -> RunResult:
    # Capture context before spawning
    context_snapshot = {
        'session_id': get_current_session(),
        'metadata': get_current_metadata(),
    }
    
    # Add to args
    args['_context'] = context_snapshot
    
    if executor == "process":
        return ProcessExecutor.execute(args)
    # ...

# In ProcessExecutor (child process)
def _run_in_child(args: dict):
    # Restore context
    ctx = args.pop('_context', {})
    session_id = ctx.get('session_id')
    metadata = ctx.get('metadata', {})
    
    # Set context variables
    if session_id:
        _session_id.set(session_id)
    if metadata:
        _metadata.set(metadata)
    
    # Now run the actual work
    result = _execute_engine(args)
    return result
```

**Pros**:
- ✅ Works reliably across all platforms
- ✅ Works with spawn (GPU-safe)
- ✅ Simple to implement
- ✅ Explicit and debuggable

**Cons**:
- ⚠️ Requires manual extraction/restoration
- ⚠️ Context must be picklable (dicts are fine)

### Option 2: Context Manager in Child

**Approach**: Use `SessionContext` in child process.

```python
# In ProcessExecutor (child process)
def _run_in_child(args: dict):
    ctx = args.pop('_context', {})
    session_id = ctx.get('session_id')
    metadata = ctx.get('metadata', {})
    
    # Use context manager to set context
    from rllm.sdk import SessionContext
    with SessionContext(session_id, **metadata):
        result = _execute_engine(args)
    
    return result
```

**Pros**:
- ✅ Reuses existing SessionContext infrastructure
- ✅ Automatic cleanup on exit
- ✅ Nested context support

**Cons**:
- ⚠️ Still requires explicit passing

### Option 3: Hybrid Approach (Fork for CPU, Spawn for GPU)

**Approach**: Use fork when safe, spawn when GPU is involved.

```python
def choose_executor(args: dict):
    if args.get('uses_gpu') or args.get('cuda_visible_devices'):
        # GPU workload: must use spawn (context lost)
        return 'spawn', True  # needs_context_passing=True
    else:
        # CPU workload: can use fork (context inherited)
        return 'fork', False  # needs_context_passing=False
```

**Pros**:
- ✅ Best of both worlds
- ✅ No context passing overhead for CPU workloads

**Cons**:
- ❌ Complex logic
- ❌ Platform-dependent behavior
- ❌ Fork still has issues (not recommended by Python docs)

### Option 4: Thread-Based Executor (No Multiprocessing)

**Approach**: Use threads instead of processes for isolation.

```python
# ThreadExecutor
def execute(args: dict):
    # Context variables automatically propagate to threads
    with ThreadPoolExecutor() as executor:
        future = executor.submit(_run_engine, args)
        return future.result()
```

**Pros**:
- ✅ Context variables automatically propagate
- ✅ No manual context passing needed
- ✅ Simpler implementation

**Cons**:
- ❌ **No GPU isolation** (CUDA context shared)
- ❌ **No memory isolation** (shared address space)
- ❌ **GIL contention** for CPU-bound work
- ❌ Defeats the purpose of ProcessExecutor

## Recommendation

**✅ SOLVED: Use `@client.entrypoint()` Decorator**

The cleanest solution is to use the `@client.entrypoint()` decorator, which wraps the function with automatic session context creation. This is **already part of the RLLM SDK API design** and solves the multiprocess problem elegantly.

### How It Works

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

# User wraps their agent function with entrypoint
@client.entrypoint(service="my-agent", experiment="v1", mode="training")
def my_agent_function(task):
    # The decorator automatically creates a session context
    # All LLM calls here get the metadata
    llm = client.get_chat_client(provider="openai")
    response = llm.chat.completions.create(...)
    return response

# When ProcessExecutor spawns child process and calls this function:
# 1. Child process starts with NO context (spawn method)
# 2. Entrypoint decorator creates NEW session context with metadata
# 3. All LLM calls inside get correct session_id and metadata
# 4. Context is cleaned up after function returns
```

### Why This Works

1. **No context inheritance needed**: The decorator creates context in the child process
2. **Metadata baked into function**: The metadata is part of the decorator, not contextvars
3. **Picklable**: Decorated functions can be pickled and sent to child processes
4. **User-friendly**: Users just wrap their function, no manual context passing
5. **Already in API**: This is the intended design from `RLLM_SDK_API.md`

### Implementation Status

✅ **IMPLEMENTED**: The `@client.entrypoint()` decorator is now implemented in `rllm/sdk/client.py`
✅ **TESTED**: Tests in `rllm/sdk/test_entrypoint.py` verify multiprocess context propagation works

### Alternative: Explicit Context Passing (Fallback)

If for some reason the entrypoint decorator cannot be used, **Option 1: Explicit Context Passing** is the fallback:

### Implementation Plan

1. **Modify RunArgs schema** to include internal `_context` field:
   ```python
   @dataclass
   class RunArgs:
       # ... existing fields
       _context: dict | None = None  # Internal: for context propagation
   ```

2. **Capture context in run()** before executor selection:
   ```python
   def run(args: dict) -> RunResult:
       # Capture current context
       args['_context'] = {
           'session_id': get_current_session(),
           'metadata': get_current_metadata(),
       }
       # ... rest of run logic
   ```

3. **Restore context in ProcessExecutor child**:
   ```python
   def _child_process_main(args: dict):
       # Restore context
       ctx = args.pop('_context', {})
       if ctx.get('session_id'):
           _session_id.set(ctx['session_id'])
       if ctx.get('metadata'):
           _metadata.set(ctx['metadata'])
       
       # Run engine
       return _execute_engine(args)
   ```

4. **Document the limitation** in the design doc:
   - Note that ProcessExecutor requires explicit context passing
   - Explain why (spawn method for GPU safety)
   - Show example usage

### Testing

Add tests to verify:
- ✅ Context propagates correctly with InlineExecutor
- ✅ Context propagates correctly with AsyncExecutor
- ✅ Context propagates correctly with ThreadExecutor
- ✅ Context propagates correctly with ProcessExecutor (via explicit passing)
- ✅ Nested contexts work correctly
- ✅ Missing context doesn't break execution

## Alternative: Document the Limitation

If explicit passing is too complex, **document that ProcessExecutor loses context**:

```python
# In design doc
## ProcessExecutor Limitations

⚠️ **Context Variable Limitation**: When using `ProcessExecutor` with `spawn` method
(required for GPU isolation), Python's `contextvars` are NOT inherited by child processes.

**Workaround**: Pass session_id explicitly in args:

```python
# Instead of relying on context:
with client.session("my-session"):
    result = run(args, executor="process")  # ❌ session lost

# Pass explicitly:
result = run({
    **args,
    'tracing': {'session_id': 'my-session', 'metadata': {...}}
}, executor="process")  # ✅ works
```

## Conclusion

The multiprocess executor design **must account for context variable propagation**.

**✅ SOLVED**: The `@client.entrypoint()` decorator provides an elegant, user-friendly solution:

- ✅ No manual context passing needed
- ✅ Works reliably across all platforms
- ✅ Compatible with spawn (GPU-safe)
- ✅ Already part of the RLLM SDK API design
- ✅ User just wraps their function with decorator
- ✅ Metadata is baked into the function, not contextvars

**Usage Pattern for Run Facade**:

```python
# User's code
@client.entrypoint(service="my-agent", mode="training")
def my_agent(task):
    # All LLM calls automatically get metadata
    return solve(task)

# Run Facade can safely use ProcessExecutor
result = run(
    engine="execution",
    tasks=[...],
    endpoint=my_agent,  # Decorated function
    executor="process"  # Spawns child process - context auto-created!
)
```

The entrypoint decorator solves the context propagation problem without requiring any changes to the Run Facade implementation.

