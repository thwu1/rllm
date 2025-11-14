# RLLM SDK - Context-Based Session Tracking

Automatic session and metadata propagation for LLM tracing using Python's `contextvars`.

## Installation

The SDK is part of the `rllm` package:

```python
from rllm.sdk import RLLMClient
```

## Quick Start

### Before: Manual session_name passing

```python
def run_agent(test_set, session_name):
    for task in test_set:
        result = solve_task(task, session_name)

def solve_task(task, session_name):
    tracer.log_llm_call(..., session_name=session_name)

run_agent(test_set, "session-123")  # Must pass everywhere!
```

### After: Automatic propagation

```python
from rllm.sdk import RLLMClient

client = RLLMClient()

def run_agent(test_set):
    for task in test_set:
        result = solve_task(task)  # No session_name!

def solve_task(task):
    tracer.log_llm_call(...)  # session_name auto-injected!

with client.session("session-123"):
    run_agent(test_set)  # Set once, propagates everywhere
```

## Features

### Simple Session
```python
with client.session("my-session"):
    tracer.log_llm_call(...)
    # All traces get session_name="my-session"
```

### Custom Metadata
```python
with client.session("my-session", experiment="v1", user="alice"):
    tracer.log_llm_call(...)
    # All traces get session_name + metadata
```

### Auto-Generated Session Name
```python
with client.session(experiment="v1"):
    tracer.log_llm_call(...)
    # session_name auto-generated
```

### Nested Contexts (Metadata Inheritance)
```python
with client.session("outer", experiment="v1"):
    tracer.log_llm_call(...)  # metadata: {experiment: "v1"}

    with client.session("inner", task="math"):
        tracer.log_llm_call(...)  # metadata: {experiment: "v1", task: "math"}

    tracer.log_llm_call(...)  # back to: {experiment: "v1"}
```

### Thread-Safe
```python
def thread1():
    with client.session("thread-1"):
        tracer.log_llm_call(...)  # Isolated context

def thread2():
    with client.session("thread-2"):
        tracer.log_llm_call(...)  # Isolated context
```

## Implementation

The SDK uses Python's `contextvars` for automatic context propagation:

1. **Context Variables** (`rllm/sdk/context.py`): Thread-safe storage
2. **Session Context Manager** (`rllm/sdk/session.py`): `__enter__`/`__exit__` handling

### Files Created

```
rllm/sdk/
├── __init__.py           # Package exports
├── context.py            # Context variables (10 lines)
├── session.py            # SessionContext class (20 lines)
├── client.py             # RLLMClient (15 lines)
├── test_context.py       # Unit tests
├── test_tracer_integration.py  # Integration tests
└── example_usage.py      # Usage examples
```

## API Reference

### RLLMClient

```python
class RLLMClient:
    def session(self, session_name: Optional[str] = None, **metadata) -> SessionContext:
        """Create session context manager."""
```

### Helper Functions

```python
def get_current_session() -> Optional[str]:
    """Get current session_name from context."""

def get_current_metadata() -> Dict[str, Any]:
    """Get current metadata from context."""
```

## Testing

Run tests:
```bash
pytest rllm/sdk/test_context.py -v
pytest rllm/sdk/test_tracer_integration.py -v
```

Run example:
```bash
python rllm/sdk/example_usage.py
```

## Design Principles

1. **Minimal code**: ~38 lines of new code total
2. **Clean imports**: No try/except, direct imports only
3. **Thread-safe**: Each thread gets isolated context
4. **Backward compatible**: Explicit session_name still works
5. **Flexible**: Support arbitrary metadata keys

## Benefits

✅ **No function signature changes** - Don't modify existing code
✅ **Automatic propagation** - Set once, available everywhere
✅ **Thread-safe** - Each thread has isolated context
✅ **Async-safe** - Works with async/await
✅ **Composable** - Nested contexts merge metadata
✅ **Flexible** - Arbitrary metadata keys supported
