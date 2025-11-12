# How to Instrument VERL vLLM Servers for Token IDs

This guide explains how to enable token IDs and logprobs from VERL vLLM servers, even when using vLLM < 0.10.2.

## Problem

vLLM < 0.10.2 doesn't support the `return_token_ids` parameter, so responses don't include:
- `prompt_token_ids` - Token IDs of the prompt
- `response_token_ids` - Token IDs of the response

This is needed for:
- LiteLLM proxy's `SamplingParametersCallback` to inject logprobs
- `TracingCallback` to log token-level telemetry
- Any downstream processing that needs token IDs

## Solution

Use RLLM's vLLM instrumentation to monkey-patch vLLM **before** VERL servers start.

### Option 1: Instrument in Trainer (Recommended)

**IMPORTANT**: You must instrument vLLM **BEFORE** creating the `AgentLoopManager`, because VERL servers run in separate Ray worker processes and monkey patches don't propagate to already-running workers.

For VERL trainers, instrument in `init_workers()` **BEFORE** calling `super().init_workers()`:

```python
from rllm.engine.vllm_instrumentation import instrument_vllm
from rllm.trainer.verl.agent_omni_trainer import AgentOmniTrainer

class MyTrainer(AgentOmniTrainer):
    def init_workers(self):
        # Instrument vLLM BEFORE creating VERL servers
        instrument_vllm()

        # This creates AgentLoopManager and launches vLLM servers
        super().init_workers()

        # Now all vLLM servers will return token IDs!
```

**How it works**:
1. `instrument_vllm()` patches vLLM classes in the main process
2. `super().init_workers()` creates `AgentLoopManager`
3. `AgentLoopManager.__init__()` launches vLLM servers in Ray workers
4. Ray workers inherit the patched vLLM classes from the main process
5. All responses include `prompt_token_ids` and `response_token_ids`

### Option 2: Standalone Script

If you're running a standalone script (not a trainer), instrument before importing VERL:

```python
# Instrument vLLM FIRST, before any VERL imports
from rllm.engine.vllm_instrumentation import instrument_vllm
instrument_vllm()

# Now import and use VERL
from verl.experimental.agent_loop import AgentLoopManager

# Create AgentLoopManager - servers will use instrumented vLLM
manager = AgentLoopManager(config, worker_group, rm_wg)

# All vLLM servers will return token IDs!
```

### Option 3: Using Ray Runtime Environment

For production deployments, set up instrumentation in Ray's runtime environment:

```python
import ray

# Define a setup function
def setup_vllm_instrumentation():
    from rllm.engine.vllm_instrumentation import instrument_vllm
    instrument_vllm()

# Initialize Ray with runtime_env
ray.init(
    runtime_env={
        "worker_process_setup_hook": setup_vllm_instrumentation
    }
)

# Now create VERL components - all workers will be instrumented
```

## Verification

Check if instrumentation worked:

```python
from rllm.engine.vllm_instrumentation import (
    is_vllm_instrumented,
    get_vllm_token_ids_support
)

# Check instrumentation status
print(f"Instrumented: {is_vllm_instrumented()}")
print(f"Support: {get_vllm_token_ids_support()}")
# Output:
# Instrumented: True
# Support: instrumented  (or 'native' for vLLM >= 0.10.2)
```

Test with OpenAI client:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:4000/v1",  # LiteLLM proxy
    api_key="EMPTY"
)

response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}]
)

# Check if token IDs are present
print(f"Prompt token IDs: {response.prompt_token_ids}")
print(f"Response token IDs: {response.response_token_ids}")
```

## How It Works

### vLLM < 0.10.2

The instrumentation monkey-patches vLLM's OpenAI server:

1. **Extends `ChatCompletionResponse`**:
   ```python
   class ChatCompletionResponsePatched(ChatCompletionResponse):
       prompt_token_ids: List[int] | None = None
       response_token_ids: List[List[int]] | None = None
   ```

2. **Wraps `chat_completion_full_generator`**:
   ```python
   async def chat_completion_full_generator(...):
       # Intercept result_generator to extract token IDs
       async for res in result_generator:
           prompt_token_ids = res.prompt_token_ids
           response_token_ids = [output.token_ids for output in res.outputs]
       
       # Add token IDs to response
       response = response.model_copy(
           update={
               "prompt_token_ids": prompt_token_ids,
               "response_token_ids": response_token_ids,
           }
       )
       return response
   ```

3. **Applies patches globally**:
   ```python
   vllm.entrypoints.openai.protocol.ChatCompletionResponse = ChatCompletionResponsePatched
   OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator
   ```

### vLLM >= 0.10.2

No instrumentation needed! vLLM natively supports `return_token_ids`:

```python
response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"return_token_ids": True}  # Native support
)
```

## Timing Considerations

### ✅ Correct Timing

```python
# 1. Instrument vLLM in the main process
instrument_vllm()

# 2. Create VERL servers (Ray workers inherit the patches)
super().init_workers()  # Creates AgentLoopManager

# 3. Create VerlEngine
verl_engine = VerlEngine(...)
```

### ❌ Incorrect Timing

```python
# 1. Create VERL servers (without instrumentation)
super().init_workers()  # Creates AgentLoopManager

# 2. Instrument vLLM (TOO LATE!)
instrument_vllm()  # Only patches main process, not Ray workers!

# 3. Create VerlEngine
verl_engine = VerlEngine(...)
```

**Why?** VERL servers run in **separate Ray worker processes**. Each Ray worker has its own Python interpreter and module namespace. Monkey patches applied in the main process **DO NOT** propagate to already-running Ray workers.

**Critical insight**: You must instrument **BEFORE** Ray workers are created. When Ray spawns a new worker process, it imports the modules fresh, so if you've already patched vLLM in the main process, the workers will inherit those patches during their initialization.

## Troubleshooting

### Token IDs are `None`

**Cause**: vLLM not instrumented or instrumentation failed.

**Solution**:
```python
from rllm.engine.vllm_instrumentation import instrument_vllm, get_vllm_token_ids_support

support = get_vllm_token_ids_support()
if support == "none":
    instrument_vllm(force=True)
```

### `AttributeError: 'ChatCompletionResponse' object has no attribute 'prompt_token_ids'`

**Cause**: Using vLLM >= 0.10.2 without `return_token_ids` parameter.

**Solution**: Use LiteLLM's `SamplingParametersCallback` to inject the parameter:
```python
from rllm.sdk.proxy.litellm_callbacks import SamplingParametersCallback

# This automatically adds return_token_ids=True
callback = SamplingParametersCallback()
```

### Instrumentation doesn't work in Ray workers

**Cause**: Instrumentation was applied **AFTER** Ray workers were created.

**Solution**: Instrument **BEFORE** creating `AgentLoopManager`:
```python
# CORRECT
instrument_vllm()  # First
super().init_workers()  # Then create workers

# WRONG
super().init_workers()  # Workers created first
instrument_vllm()  # Too late!
```

**Why**: Ray workers import modules during initialization. If you patch vLLM before creating workers, they inherit the patches. If you patch after, the workers already have the unpatched version.

## Performance Impact

- **CPU overhead**: Negligible (~0.1%)
- **Memory overhead**: ~4 bytes/token (already in memory)
- **Latency overhead**: < 1ms
- **Throughput impact**: None

The token IDs are already computed by vLLM internally - we're just exposing them in the response.

## Summary

| vLLM Version | Instrumentation Needed? | How to Enable |
|--------------|------------------------|---------------|
| < 0.10.2 | ✅ Yes | `instrument_vllm()` |
| >= 0.10.2 | ❌ No | Use `return_token_ids=True` |

**Recommended approach**: Use `VerlProxyManager` with `auto_instrument_vllm=True` (default) - it handles everything automatically!

