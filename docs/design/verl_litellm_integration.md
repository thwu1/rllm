# VERL + LiteLLM Proxy Integration Design

## Overview

This document describes how RLLM integrates VERL's multiple vLLM instances with LiteLLM proxy to provide:

1. **Automatic load balancing** across all vLLM replicas
2. **Unified OpenAI-compatible endpoint** with session tracking
3. **Metadata routing** via URL-encoded slugs
4. **Full observability** through LLMTracer integration

## Problem Statement

When using VERL as the rollout engine, you get multiple vLLM server instances (replicas) for data parallelism:

```python
# VERL creates multiple replicas
num_replicas = world_size // rollout_world_size

# Each replica exposes an OpenAI-compatible HTTP endpoint
server_addresses = [
    "192.168.1.100:8000",  # Replica 0
    "192.168.1.101:8001",  # Replica 1
    "192.168.1.102:8002",  # Replica 2
]
```

**Challenges:**

1. **How to use all replicas?** Using only the first server wastes resources
2. **How to load balance?** Need to distribute requests across replicas
3. **How to track sessions?** Need to propagate session_id and metadata
4. **How to integrate with existing proxy?** RLLM already has LiteLLM proxy infrastructure

## Solution Architecture

### Components

1. **VerlProxyManager** (`rllm/engine/proxy_manager.py`)
   - Extracts all vLLM server addresses from VERL's `AgentLoopManager`
   - Generates LiteLLM configuration with all replicas
   - Manages proxy server lifecycle
   - Provides unified endpoint URL

2. **AgentOmniEngine Integration** (`rllm/engine/agent_omni_engine.py`)
   - Auto-detects VERL engines
   - Initializes `VerlProxyManager` with config
   - Exposes `get_openai_endpoint()` for clients
   - Optional auto-start of proxy server

3. **LiteLLM Proxy** (existing infrastructure)
   - `MetadataRoutingMiddleware`: Decodes URL slugs
   - `SamplingParametersCallback`: Injects logprobs/token_ids
   - `TracingCallback`: Logs to LLMTracer
   - Load balancing: Round-robin across replicas

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Code                                                 │
│    @client.entrypoint                                        │
│    def my_agent(task):                                       │
│        response = openai_client.chat.completions.create(...) │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Metadata from contextvars
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ProxyTrackedChatClient                                    │
│    - Reads session_id from get_current_session()            │
│    - Reads metadata from get_current_metadata()             │
│    - Encodes to slug: rllm1:base64(json)                    │
│    - Builds URL: /meta/{slug}/v1/chat/completions           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP POST /meta/{slug}/v1/chat/completions
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LiteLLM Proxy (Port 4000)                                 │
│    ┌────────────────────────────────────────────────────┐   │
│    │ MetadataRoutingMiddleware                          │   │
│    │  - Decodes slug → {"session_id": "...", ...}      │   │
│    │  - Rewrites path → /v1/chat/completions           │   │
│    │  - Stores in request.state.rllm_metadata          │   │
│    └────────────────────────────────────────────────────┘   │
│    ┌────────────────────────────────────────────────────┐   │
│    │ SamplingParametersCallback                         │   │
│    │  - Injects logprobs=True                           │   │
│    │  - Injects return_token_ids=True (for vLLM)       │   │
│    └────────────────────────────────────────────────────┘   │
│    ┌────────────────────────────────────────────────────┐   │
│    │ Load Balancer (simple-shuffle)                     │   │
│    │  - Round-robin across model_list entries          │   │
│    │  - All entries have same model_name                │   │
│    └────────────────────────────────────────────────────┘   │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
               ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  vLLM    │   │  vLLM    │   │  vLLM    │
       │ Replica  │   │ Replica  │   │ Replica  │
       │    0     │   │    1     │   │    2     │
       └──────────┘   └──────────┘   └──────────┘
               │              │              │
               └──────────────┴──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TracingCallback                                           │
│    - Extracts metadata from request.state                   │
│    - Logs to LLMTracer with session_id                      │
│    - Records: prompt, completion, tokens, logprobs, latency │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. VerlProxyManager

**Responsibilities:**
- Extract server addresses from VERL
- Generate LiteLLM config
- Manage proxy lifecycle

**Key Methods:**

```python
class VerlProxyManager:
    def __init__(
        self,
        rollout_engine: VerlEngine,
        model_name: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        tracer: LLMTracer | None = None,
    ):
        # Extract addresses from VERL
        self._server_addresses = rollout_engine.rollout_manager.server_addresses
        
        # Generate config
        self._config = self._generate_litellm_config()
    
    def _generate_litellm_config(self) -> dict:
        """Generate LiteLLM config with all replicas."""
        model_list = []
        for idx, server_address in enumerate(self._server_addresses):
            model_list.append({
                "model_name": self.model_name,
                "litellm_params": {
                    "model": f"hosted_vllm/{self.model_name}",
                    "api_base": f"http://{server_address}/v1",
                    "drop_params": True,
                },
                "model_info": {
                    "id": f"verl-replica-{idx}",
                    "replica_rank": idx,
                }
            })
        
        return {
            "model_list": model_list,
            "litellm_settings": {
                "drop_params": True,
                "num_retries": 3,
                "routing_strategy": "simple-shuffle",  # Round-robin
            }
        }
    
    def get_proxy_url(self) -> str:
        """Get unified endpoint URL."""
        return f"http://{self.proxy_host}:{self.proxy_port}/v1"
    
    def start_proxy_server(self):
        """Start LiteLLM proxy with middleware and callbacks."""
        # Setup FastAPI app
        # Add MetadataRoutingMiddleware
        # Add SamplingParametersCallback
        # Add TracingCallback
        # Mount LiteLLM
        # Start uvicorn
```

### 2. AgentOmniEngine Integration

**Auto-detection and setup:**

```python
class AgentOmniEngine:
    def __init__(
        self,
        agent_run_func: callable,
        rollout_engine: RolloutEngine,
        proxy_config: Optional[dict] = None,
        tracer: Optional[LLMTracer] = None,
        **kwargs
    ):
        self.rollout_engine = rollout_engine
        self.proxy_manager = None
        self.rollout_engine_endpoint = None
        
        # Auto-setup for VERL engines
        if isinstance(rollout_engine, VerlEngine):
            self._setup_verl_proxy(proxy_config or {}, tracer)
    
    def _setup_verl_proxy(self, proxy_config: dict, tracer):
        """Setup proxy for VERL engine."""
        model_name = proxy_config.get("model_name")
        if not model_name:
            logger.warning("No model_name in proxy_config, skipping proxy setup")
            return
        
        self.proxy_manager = VerlProxyManager(
            rollout_engine=self.rollout_engine,
            model_name=model_name,
            proxy_host=proxy_config.get("proxy_host", "127.0.0.1"),
            proxy_port=proxy_config.get("proxy_port", 4000),
            tracer=tracer,
        )
        
        self.rollout_engine_endpoint = self.proxy_manager.get_proxy_url()
        
        if proxy_config.get("auto_start", False):
            self.proxy_manager.start_proxy_server()
    
    def get_openai_endpoint(self) -> Optional[str]:
        """Get OpenAI-compatible endpoint."""
        return self.rollout_engine_endpoint
```

### 3. LiteLLM Configuration

**Generated config structure:**

```yaml
model_list:
  # One entry per vLLM replica
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

litellm_settings:
  drop_params: true
  num_retries: 3
  routing_strategy: simple-shuffle  # LiteLLM round-robins across entries
```

**Load Balancing:**
- LiteLLM sees multiple entries with the same `model_name`
- Uses `simple-shuffle` strategy to round-robin
- Each request goes to a different replica
- Sticky sessions NOT used (stateless vLLM)

## Usage Patterns

### Pattern 1: Auto-configured with AgentOmniEngine

```python
engine = AgentOmniEngine(
    agent_run_func=my_agent,
    rollout_engine=verl_engine,
    proxy_config={
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "proxy_port": 4000,
        "auto_start": True,
    },
    tracer=client.tracer,
)

# Use the endpoint
endpoint = engine.get_openai_endpoint()  # http://127.0.0.1:4000/v1
```

### Pattern 2: Manual proxy management

```python
proxy_mgr = VerlProxyManager(
    rollout_engine=verl_engine,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    proxy_port=4000,
)

# Write config for external use
proxy_mgr.write_config_file("litellm_verl.yaml")

# Or start embedded server
proxy_mgr.start_proxy_server()
```

### Pattern 3: Production deployment

```python
# Generate config
proxy_mgr.write_config_file("/etc/litellm/verl.yaml")

# Run proxy in separate process
# $ uvicorn proxy_app:app --host 0.0.0.0 --port 4000
```

## Benefits

1. **Automatic Load Balancing**: All vLLM replicas are utilized
2. **Unified Interface**: Single endpoint for all replicas
3. **Session Tracking**: Metadata propagates through URL slugs
4. **Observability**: Full tracing with session context
5. **Token-Level Telemetry**: Automatic injection of logprobs/token_ids
6. **Flexible Deployment**: Embedded or standalone proxy

## Comparison with Direct Ray RPC

| Aspect | Direct Ray RPC (VerlEngine) | LiteLLM Proxy |
|--------|----------------------------|---------------|
| **Interface** | Token-in-token-out | OpenAI-compatible HTTP |
| **Load Balancing** | AsyncLLMServerManager (internal) | LiteLLM (external) |
| **Use Case** | Training/internal rollout | Inference/external clients |
| **Overhead** | Minimal (direct memory) | HTTP serialization |
| **Observability** | Limited | Full tracing |
| **Compatibility** | VERL-specific | Standard OpenAI SDK |

**When to use each:**
- **Direct Ray RPC**: Training loops, internal rollout (VerlEngine default)
- **LiteLLM Proxy**: Inference, external clients, session tracking, observability

## Future Enhancements

1. **Health Checks**: Monitor replica health and remove unhealthy servers
2. **Weighted Load Balancing**: Route based on replica load/latency
3. **Sticky Sessions**: For prefix caching (if needed)
4. **Dynamic Replica Updates**: Add/remove replicas without restart
5. **Metrics Dashboard**: Real-time load balancing metrics

## See Also

- [LiteLLM Proxy Documentation](https://docs.litellm.ai/docs/proxy/configs)
- [VERL Documentation](https://github.com/volcengine/verl)
- [Proxy Plan](../../rllm/sdk/proxy/plan.md)
- [Run Facade Design](./run_facade.md)

