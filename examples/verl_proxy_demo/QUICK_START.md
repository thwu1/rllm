# VERL Proxy Quick Start

## 30-Second Setup

```python
from rllm.engine.agent_omni_engine import AgentOmniEngine
from openai import AsyncOpenAI

# 1. Create engine with VERL
engine = AgentOmniEngine(
    agent_run_func=my_agent,
    rollout_engine=verl_engine,  # Your VERL engine
    proxy_config={
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "proxy_port": 4000,
        "auto_start": True,
    },
)

# 2. Get endpoint
endpoint = engine.get_openai_endpoint()

# 3. Use with OpenAI client
client = AsyncOpenAI(base_url=endpoint, api_key="EMPTY")
response = await client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Done! All vLLM replicas are now load-balanced automatically.

## What You Get

✅ **Single endpoint** for all vLLM replicas  
✅ **Automatic load balancing** (round-robin)  
✅ **Session tracking** via metadata slugs  
✅ **Full tracing** with LLMTracer  
✅ **Token-level telemetry** (logprobs, token_ids)  

## Common Patterns

### Pattern 1: With @entrypoint decorator

```python
from rllm.sdk import RLLMClient

client = RLLMClient(project="my-project")

@client.entrypoint
async def my_agent(task: str):
    openai_client = AsyncOpenAI(
        base_url=engine.get_openai_endpoint(),
        api_key="EMPTY"
    )
    response = await openai_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": task}]
    )
    return response.choices[0].message.content

# Call with metadata
result = await my_agent(
    "Solve this problem",
    _metadata={"session_id": "run-123", "experiment": "v1"}
)
```

### Pattern 2: Manual proxy control

```python
from rllm.engine.proxy_manager import VerlProxyManager

# Create manager
proxy_mgr = VerlProxyManager(
    rollout_engine=verl_engine,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    proxy_port=4000,
)

# Start proxy
proxy_mgr.start_proxy_server()

# Get endpoint
endpoint = proxy_mgr.get_proxy_url()

# Stop when done
proxy_mgr.stop_proxy_server()
```

### Pattern 3: Export config for external use

```python
# Generate config file
proxy_mgr = VerlProxyManager(
    rollout_engine=verl_engine,
    model_name="Qwen/Qwen2.5-7B-Instruct",
)

# Write to file
config_path = proxy_mgr.write_config_file("litellm_verl.yaml")

# Run externally
# $ litellm --config litellm_verl.yaml --port 4000
```

## Debugging

### Check server addresses

```python
# Get all vLLM replica addresses
servers = engine.get_server_addresses()
print(f"vLLM replicas: {servers}")
# Output: ['192.168.1.100:8000', '192.168.1.101:8001', '192.168.1.102:8002']
```

### Inspect config

```python
import yaml

config = proxy_mgr.get_litellm_config()
print(yaml.dump(config, default_flow_style=False))
```

### Check proxy status

```python
if proxy_mgr.is_running():
    print(f"Proxy running at {proxy_mgr.get_proxy_url()}")
else:
    print("Proxy not running")
```

## Configuration Options

```python
proxy_config = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",  # Required
    "proxy_host": "127.0.0.1",                  # Default: 127.0.0.1
    "proxy_port": 4000,                         # Default: 4000
    "auto_start": True,                         # Default: False
}

engine = AgentOmniEngine(
    agent_run_func=my_agent,
    rollout_engine=verl_engine,
    proxy_config=proxy_config,
    tracer=client.tracer,  # Optional: for logging
)
```

## Troubleshooting

### "No server addresses found"

```python
# Make sure VERL engine is initialized
# Check if rollout_manager has server_addresses
servers = verl_engine.rollout_manager.server_addresses
if not servers:
    print("VERL servers not initialized yet")
```

### "Proxy won't start"

```python
# Check if port is already in use
# Try a different port
proxy_config = {"model_name": "...", "proxy_port": 4001}
```

### "Session context not propagating"

```python
# Make sure you're using @client.entrypoint
# and passing _metadata kwarg

@client.entrypoint  # ← Must use decorator
async def my_agent(task):
    # ...

# Must pass _metadata
result = await my_agent(task, _metadata={"session_id": "..."})
```

## See Also

- [Full README](./README.md) - Complete documentation
- [Examples](./example_usage.py) - Code examples
- [Design Doc](../../docs/design/verl_litellm_integration.md) - Architecture details

