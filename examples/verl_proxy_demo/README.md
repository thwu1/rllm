# VERL + LiteLLM Proxy Integration

This example demonstrates how to use RLLM's LiteLLM proxy with VERL's multiple vLLM instances for load-balanced inference.

## Overview

When using VERL as the rollout engine, you get multiple vLLM server instances (replicas) for data parallelism. The `VerlProxyManager` automatically:

1. **Extracts all vLLM server addresses** from VERL's `AgentLoopManager`
2. **Configures LiteLLM** with all replicas for automatic load balancing
3. **Provides a unified OpenAI-compatible endpoint** with metadata routing
4. **Enables session tracking** via URL-encoded metadata slugs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
│  @client.entrypoint                                          │
│  def my_agent(task):                                         │
│      response = openai_client.chat.completions.create(...)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ http://localhost:4000/meta/{slug}/v1
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LiteLLM Proxy (Port 4000)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MetadataRoutingMiddleware                            │   │
│  │  - Decodes /meta/{slug} → session_id, metadata       │   │
│  │  - Rewrites path to /v1/chat/completions             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SamplingParametersCallback                           │   │
│  │  - Injects logprobs=True, return_token_ids=True      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TracingCallback                                      │   │
│  │  - Logs all calls to LLMTracer with session context  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Load Balancer (simple-shuffle)                       │   │
│  │  - Round-robin across all vLLM replicas              │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
               ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  vLLM    │   │  vLLM    │   │  vLLM    │
       │ Replica  │   │ Replica  │   │ Replica  │
       │    0     │   │    1     │   │    2     │
       │ :8000    │   │ :8001    │   │ :8002    │
       └──────────┘   └──────────┘   └──────────┘
```

## Usage

### 1. Basic Setup

```python
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.engine.agent_omni_engine import AgentOmniEngine
from rllm.sdk import RLLMClient

# Initialize VERL engine (with multiple replicas)
verl_engine = VerlEngine(config, rollout_manager, tokenizer)

# Create AgentOmniEngine with proxy configuration
engine = AgentOmniEngine(
    agent_run_func=my_agent_function,
    rollout_engine=verl_engine,
    proxy_config={
        "model_name": "Qwen/Qwen2.5-7B-Instruct",  # Model name to expose
        "proxy_host": "127.0.0.1",                  # Proxy host
        "proxy_port": 4000,                         # Proxy port
        "auto_start": True,                         # Auto-start proxy server
    },
    tracer=client.tracer,  # Optional: for logging
)

# Get the unified endpoint
endpoint = engine.get_openai_endpoint()
print(f"OpenAI endpoint: {endpoint}")  # http://127.0.0.1:4000/v1

# Get all vLLM server addresses (for debugging)
servers = engine.get_server_addresses()
print(f"vLLM replicas: {servers}")  # ['192.168.1.100:8000', '192.168.1.101:8001', ...]
```

### 2. Using with OpenAI Client

```python
from openai import AsyncOpenAI
from rllm.sdk import RLLMClient

# Initialize RLLM client
client = RLLMClient(
    context_store_endpoint="http://localhost:8000",
    project="verl-demo"
)

# Create OpenAI client pointing to the proxy
openai_client = AsyncOpenAI(
    base_url=engine.get_openai_endpoint(),
    api_key="EMPTY"  # VERL doesn't require authentication
)

# Define agent function with @entrypoint decorator
@client.entrypoint
async def my_agent(task: str):
    # This call will:
    # 1. Encode session_id + metadata into URL slug
    # 2. Route through LiteLLM proxy
    # 3. Load balance across vLLM replicas
    # 4. Log to tracer with session context
    response = await openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": task}],
    )
    return response.choices[0].message.content

# Run with session context
async def main():
    # The _metadata kwarg is passed by the Run Facade or your orchestrator
    result = await my_agent(
        "Solve this problem: 2+2=?",
        _metadata={
            "session_id": "run-123",
            "experiment": "math-eval",
            "split": "test"
        }
    )
    print(result)
```

### 3. Manual Proxy Setup (Advanced)

If you want more control over the proxy lifecycle:

```python
from rllm.engine.proxy_manager import VerlProxyManager

# Create proxy manager manually
proxy_mgr = VerlProxyManager(
    rollout_engine=verl_engine,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    proxy_host="0.0.0.0",  # Bind to all interfaces
    proxy_port=4000,
    tracer=client.tracer,
)

# Get configuration
config = proxy_mgr.get_litellm_config()
print(f"LiteLLM config: {config}")

# Write config to file
config_path = proxy_mgr.write_config_file("/tmp/litellm_verl.yaml")

# Start proxy server
proxy_mgr.start_proxy_server(config_path)

# Use the endpoint
endpoint = proxy_mgr.get_proxy_url()
print(f"Proxy running at: {endpoint}")

# Later: stop the proxy
proxy_mgr.stop_proxy_server()
```

### 4. Production Deployment

For production, use a proper process manager instead of the built-in thread-based server:

```python
# 1. Generate config file
proxy_mgr = VerlProxyManager(
    rollout_engine=verl_engine,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    proxy_port=4000,
)
config_path = proxy_mgr.write_config_file("litellm_verl.yaml")

# 2. Run proxy with uvicorn (in separate process or container)
# See examples/proxy_demo/proxy_app.py for full setup
```

## LiteLLM Configuration

The generated LiteLLM config looks like this:

```yaml
model_list:
  - model_name: Qwen/Qwen2.5-7B-Instruct
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
      api_base: http://192.168.1.100:8000/v1
      drop_params: true
    model_info:
      id: verl-replica-0
      replica_rank: 0
  
  - model_name: Qwen/Qwen2.5-7B-Instruct
    litellm_params:
      model: hosted_vllm/Qwen/Qwen2.5-7B-Instruct
      api_base: http://192.168.1.101:8001/v1
      drop_params: true
    model_info:
      id: verl-replica-1
      replica_rank: 1
  
  # ... one entry per vLLM replica

litellm_settings:
  drop_params: true
  num_retries: 3
  routing_strategy: simple-shuffle  # Round-robin load balancing
```

## Metadata Routing

The proxy supports URL-encoded metadata for session tracking:

```python
# Without metadata (plain URL)
http://localhost:4000/v1/chat/completions

# With metadata (encoded in URL)
http://localhost:4000/meta/rllm1:eyJzZXNzaW9uX2lkIjoicnVuLTEyMyJ9/v1/chat/completions
                           └─────────────────────────────────┘
                                  base64({"session_id": "run-123"})
```

The middleware automatically:
1. Decodes the slug
2. Rewrites the path to `/v1/chat/completions`
3. Injects metadata into request state
4. Passes metadata to tracer callbacks

## Benefits

1. **Automatic Load Balancing**: LiteLLM distributes requests across all vLLM replicas
2. **Session Tracking**: All LLM calls are logged with session context
3. **Token-Level Telemetry**: Proxy injects `logprobs=True` and `return_token_ids=True`
4. **Unified Interface**: Single OpenAI-compatible endpoint for all replicas
5. **Metadata Routing**: URL-encoded metadata for flexible context propagation
6. **Observability**: Full tracing via LLMTracer integration

## Troubleshooting

### Proxy won't start

```python
# Check if VERL servers are initialized
servers = engine.get_server_addresses()
if not servers:
    print("VERL servers not initialized yet")
```

### Load balancing not working

```python
# Verify all replicas are in the config
config = proxy_mgr.get_litellm_config()
print(f"Number of replicas: {len(config['model_list'])}")
```

### Session context not propagating

```python
# Make sure you're using @client.entrypoint decorator
# and passing _metadata kwarg when calling the function
@client.entrypoint
def my_agent(task):
    # ...

# Call with metadata
result = my_agent(task, _metadata={"session_id": "run-123"})
```

## See Also

- [LiteLLM Proxy Documentation](https://docs.litellm.ai/docs/proxy/configs)
- [VERL Documentation](https://github.com/volcengine/verl)
- [RLLM SDK API](../../RLLM_SDK_API.md)
- [Proxy Plan](../../rllm/sdk/proxy/plan.md)

