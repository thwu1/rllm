# How to Patch vLLM to Return Token IDs (vLLM < 0.10.2)

## Quick Answer

**Difficulty: EASY** - Just 3 steps, ~5 minutes

```python
# Step 1: Import the patch
from agentlightning.instrumentation.vllm import instrument_vllm

# Step 2: Apply before starting vLLM
instrument_vllm()

# Step 3: Start vLLM server
# All responses now include prompt_token_ids and response_token_ids
```

## When Do You Need This?

### Check Your vLLM Version

```bash
python -c "import vllm; print(vllm.__version__)"
```

- **vLLM < 0.10.2**: You need this patch ✅
- **vLLM >= 0.10.2**: Use native `return_token_ids` parameter instead ❌

### What You Get

After patching, vLLM responses will include:

```json
{
  "choices": [...],
  "usage": {...},
  "prompt_token_ids": [1, 2, 3, 4, 5],           // ← NEW
  "response_token_ids": [[100, 101, 102, 103]]   // ← NEW
}
```

## Implementation

### Option 1: Integrate with VerlProxyManager (Recommended)

Update `rllm/engine/proxy_manager.py`:

```python
from agentlightning.instrumentation.vllm import instrument_vllm

class VerlProxyManager:
    def __init__(
        self,
        rollout_engine: VerlEngine,
        model_name: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        tracer: LLMTracer | None = None,
        instrument_vllm_for_token_ids: bool = True,  # NEW
    ):
        # Apply vLLM patch if needed
        if instrument_vllm_for_token_ids:
            try:
                instrument_vllm()
                logger.info("vLLM instrumented to return token IDs")
            except Exception as e:
                logger.warning(f"Failed to instrument vLLM: {e}")
        
        # ... rest of initialization
```

### Option 2: Manual Application

If you're starting vLLM servers manually:

```python
from agentlightning.instrumentation.vllm import instrument_vllm
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

# Apply patch BEFORE creating vLLM servers
instrument_vllm()

# Now create vLLM servers
server = AsyncvLLMServer(config)
await server.launch_servers()

# All responses will include token IDs
```

### Option 3: Standalone vLLM Server

If you're running vLLM as a standalone server:

```python
# server_with_patch.py
from agentlightning.instrumentation.vllm import instrument_vllm
import vllm.entrypoints.openai.api_server

# Apply patch
instrument_vllm()

# Start vLLM server
if __name__ == "__main__":
    vllm.entrypoints.openai.api_server.run_server()
```

Run with:
```bash
python server_with_patch.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

## Extracting Token IDs

### In LiteLLM Callbacks

Update `rllm/sdk/proxy/litellm_callbacks.py`:

```python
class TracingCallback(CustomLogger):
    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ):
        # Extract token IDs if available (vLLM < 0.10.2 with patch)
        prompt_token_ids = getattr(response_obj, 'prompt_token_ids', None)
        response_token_ids = getattr(response_obj, 'response_token_ids', None)
        
        # For vLLM >= 0.10.2, token IDs might be in different location
        if not prompt_token_ids and hasattr(response_obj, 'choices'):
            for choice in response_obj.choices:
                if hasattr(choice, 'prompt_token_ids'):
                    prompt_token_ids = choice.prompt_token_ids
                if hasattr(choice, 'response_token_ids'):
                    response_token_ids = choice.response_token_ids
        
        # Build metadata
        metadata = {
            "token_ids": {
                "prompt": prompt_token_ids,
                "completion": response_token_ids[0] if response_token_ids else None
            }
        }
        
        # Log to tracer
        if self.tracer:
            self.tracer.log_llm_call(
                session_id=session_id,
                metadata=metadata,
                ...
            )
```

### In OpenAI Client

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    logprobs=True,  # Also request logprobs
)

# Access token IDs
prompt_token_ids = response.prompt_token_ids
response_token_ids = response.response_token_ids[0]  # First choice

print(f"Prompt tokens: {prompt_token_ids}")
print(f"Response tokens: {response_token_ids}")
```

## Testing

### Basic Test

```python
import asyncio
from openai import AsyncOpenAI
from agentlightning.instrumentation.vllm import instrument_vllm

async def test_token_ids():
    # Apply patch
    instrument_vllm()
    
    # Create client (assumes vLLM server running on port 8000)
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )
    
    # Make request
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Say hello"}],
        logprobs=True,
    )
    
    # Verify token IDs are present
    assert hasattr(response, 'prompt_token_ids'), "Missing prompt_token_ids"
    assert hasattr(response, 'response_token_ids'), "Missing response_token_ids"
    assert response.prompt_token_ids is not None, "prompt_token_ids is None"
    assert response.response_token_ids is not None, "response_token_ids is None"
    
    print("✅ Token IDs patch working!")
    print(f"Prompt tokens: {len(response.prompt_token_ids)}")
    print(f"Response tokens: {len(response.response_token_ids[0])}")

if __name__ == "__main__":
    asyncio.run(test_token_ids())
```

### Integration Test with VERL

```python
from rllm.engine.proxy_manager import VerlProxyManager
from rllm.engine.rollout.verl_engine import VerlEngine
from openai import AsyncOpenAI

async def test_verl_with_token_ids():
    # Create VERL engine
    verl_engine = VerlEngine(config, rollout_manager, tokenizer)
    
    # Create proxy manager (automatically applies patch)
    proxy_mgr = VerlProxyManager(
        rollout_engine=verl_engine,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        proxy_port=4000,
        instrument_vllm_for_token_ids=True,  # Enable patch
    )
    
    # Start proxy
    proxy_mgr.start_proxy_server()
    
    # Create client
    client = AsyncOpenAI(
        base_url=proxy_mgr.get_proxy_url(),
        api_key="EMPTY"
    )
    
    # Make request
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        logprobs=True,
    )
    
    # Verify
    assert response.prompt_token_ids is not None
    assert response.response_token_ids is not None
    print("✅ VERL + Token IDs patch working!")
```

## How the Patch Works

### Technical Details

The patch uses **monkey patching** to intercept vLLM's response generation:

1. **Extends Response Schema**:
   - Adds `prompt_token_ids` field
   - Adds `response_token_ids` field

2. **Intercepts Generator**:
   - Wraps `OpenAIServingChat.chat_completion_full_generator`
   - Extracts token IDs from internal `RequestOutput` objects
   - Adds token IDs to response

3. **No Source Modification**:
   - Pure Python monkey patching
   - No need to rebuild vLLM
   - Works with any vLLM < 0.10.2

### Why It Works

vLLM's internal `RequestOutput` already contains:
- `prompt_token_ids`: The tokenized prompt
- `outputs[i].token_ids`: The generated tokens

The patch simply **exposes** this data in the OpenAI API response.

## Troubleshooting

### "vllm is already instrumented"

This warning is harmless - it means `instrument_vllm()` was called multiple times.

```python
# To avoid the warning, check first:
import vllm.entrypoints.openai.protocol
from agentlightning.instrumentation.vllm import ChatCompletionResponsePatched

if vllm.entrypoints.openai.protocol.ChatCompletionResponse is not ChatCompletionResponsePatched:
    instrument_vllm()
```

### Token IDs Not in Response

**Possible causes**:

1. **Patch not applied**: Make sure `instrument_vllm()` is called before starting vLLM
2. **vLLM >= 0.10.2**: Use `return_token_ids=True` parameter instead
3. **Wrong response object**: Check if you're accessing the right object

**Debug**:
```python
print(f"Response type: {type(response)}")
print(f"Has prompt_token_ids: {hasattr(response, 'prompt_token_ids')}")
print(f"Response dict: {response.model_dump()}")
```

### Performance Impact

**Minimal** - the patch only extracts data that vLLM already computes:

- **CPU overhead**: Negligible (just attribute access)
- **Memory overhead**: ~4 bytes per token (already in memory)
- **Latency overhead**: < 1ms (no additional computation)

## Migration to vLLM >= 0.10.2

When you upgrade to vLLM >= 0.10.2, you can remove the patch:

### Before (vLLM < 0.10.2)

```python
from agentlightning.instrumentation.vllm import instrument_vllm

instrument_vllm()

response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    logprobs=True,
)

# Token IDs in response object
token_ids = response.prompt_token_ids
```

### After (vLLM >= 0.10.2)

```python
# No patch needed!

response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    logprobs=True,
    return_token_ids=True,  # ← Native parameter
)

# Token IDs in response object (same location)
token_ids = response.prompt_token_ids
```

## Summary

| Aspect | Details |
|--------|---------|
| **Difficulty** | ⭐⭐☆☆☆ Very Easy |
| **Code Changes** | 1-3 lines |
| **Time Required** | 5 minutes |
| **Maintenance** | Low (already implemented) |
| **Performance Impact** | Negligible |
| **Compatibility** | vLLM < 0.10.2 |

**Bottom Line**: The patch is **production-ready**, **easy to use**, and **already implemented** in your codebase (agent-lightning). Just import and call `instrument_vllm()`!

