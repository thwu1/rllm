# RLLM LiteLLM Proxy

OpenAI-compatible proxy server with episodic tracing for RLLM (Reinforcement Learning Language Model) workloads.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI client libraries
- **Multi-Provider Support**: Route requests to OpenAI, Anthropic, Cohere, Google, and more via LiteLLM
- **Episodic Tracing**: Automatic capture of prompts, completions, token IDs, logprobs, and timing
- **Context Propagation**: Session-based tracking through HTTP headers
- **Telemetry Augmentation**: Automatic request enrichment with logprobs and token IDs
- **Production Ready**: FastAPI-based with health checks, error handling, and logging

## Architecture

```
┌─────────────────┐
│  RLLM SDK/App   │
│  (Client)       │
└────────┬────────┘
         │ HTTP with
         │ X-RLLM-Session headers
         ▼
┌─────────────────────────┐
│   RLLM Proxy Server     │
│  ┌──────────────────┐   │
│  │   Middleware     │   │
│  │  - Context       │   │
│  │  - Latency       │   │
│  │  - Tracing       │   │
│  └──────────────────┘   │
│  ┌──────────────────┐   │
│  │   LiteLLM        │   │
│  │   Router         │   │
│  └──────────────────┘   │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────┐
│   LLM Providers      │
│  - OpenAI            │
│  - Anthropic         │
│  - Cohere, etc.      │
└──────────────────────┘
```

## Quick Start

### Using Docker Compose (Recommended)

1. **Set up environment variables**:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

2. **Start the proxy**:

```bash
docker-compose up -d
```

3. **Test the proxy**:

```bash
curl http://localhost:8000/health
```

### Manual Installation

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Configure environment**:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

3. **Run the server**:

```bash
python -m sdk.proxy.server
# Or with uvicorn
uvicorn sdk.proxy.server:create_app --factory --host 0.0.0.0 --port 8000
```

## Usage

### Basic Request (without tracing)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Proxy uses configured keys
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Request with Episodic Tracing

```python
import openai
import json

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    default_headers={
        "X-RLLM-Session": "episode-123",
        "X-RLLM-Metadata": json.dumps({
            "task_id": "task-456",
            "agent_name": "agent-1",
            "step_idx": 0
        })
    }
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Trace is automatically logged to ./logs/llm_traces/episode-123/
```

### Session Headers

The proxy recognizes these headers for tracing:

| Header | Description | Example |
|--------|-------------|---------|
| `X-RLLM-Session` | Episode/trajectory ID | `"episode-123"` |
| `X-RLLM-Metadata` | JSON metadata | `'{"task_id": "task-456"}'` |
| `X-RLLM-Request-ID` | Request identifier (auto-generated) | `"uuid-..."` |

### Trace Storage

Traces are stored in JSON format:

```
logs/llm_traces/
├── episode-123/
│   ├── request-uuid-1.json
│   ├── request-uuid-2.json
│   └── ...
└── episode-456/
    └── ...
```

Each trace contains:

```json
{
  "request_id": "uuid-...",
  "session_id": "episode-123",
  "timestamp": "2024-01-01T12:00:00",
  "model": "gpt-3.5-turbo",
  "messages": [...],
  "response_text": "...",
  "prompt_tokens": [...],
  "completion_tokens": [...],
  "logprobs": [...],
  "prompt_length": 10,
  "completion_length": 20,
  "latency_ms": 1234.56,
  "finish_reason": "stop",
  "metadata": {...},
  "provider": "openai"
}
```

## Configuration

### config.yaml

See `config.yaml` for full configuration options:

- **Server settings**: host, port, workers
- **Tracing settings**: enable_logprobs, log_dir
- **Model routing**: map model names to providers
- **Security**: API key validation, rate limiting

### Environment Variables

- `RLLM_PROXY_CONFIG`: Path to config.yaml
- `RLLM_LOG_DIR`: Directory for trace logs
- `RLLM_ENABLE_LOGPROBS`: Enable logprobs by default (true/false)
- `RLLM_ENABLE_PROMPT_TOKENS`: Request prompt tokens (true/false)

Provider API keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `COHERE_API_KEY`
- `GOOGLE_API_KEY`

## Deployment

### Kubernetes with Helm

See `helm/` directory for Helm charts:

```bash
cd helm/rllm-proxy
helm install rllm-proxy . \
  --set env.OPENAI_API_KEY="sk-..." \
  --set env.ANTHROPIC_API_KEY="sk-ant-..."
```

### Production Considerations

1. **Cost Management**: Logprobs increase API costs. Set `enable_logprobs: false` if not needed.

2. **Security**: Use API key authentication in production:
   ```yaml
   security:
     require_api_key: true
     api_keys: ["key1", "key2"]
   ```

3. **Scaling**: Run multiple workers:
   ```bash
   uvicorn sdk.proxy.server:create_app --workers 4
   ```

4. **Monitoring**:
   - Health check: `GET /health`
   - Metrics: Enable Prometheus metrics (TODO)
   - Logs: JSON structured logging

5. **Rate Limiting**: Enable in config.yaml to prevent abuse

## Integration with RLLM

### Creating a LiteLLM RolloutEngine

```python
from rllm.engine.rollout import RolloutEngine
import openai

class LiteLLMEngine(RolloutEngine):
    def __init__(self, proxy_url: str = "http://localhost:8000/v1"):
        self.client = openai.AsyncOpenAI(base_url=proxy_url)

    async def get_model_response(
        self,
        messages: list[dict],
        session_id: str,
        metadata: dict,
        **kwargs
    ):
        # Set session headers
        headers = {
            "X-RLLM-Session": session_id,
            "X-RLLM-Metadata": json.dumps(metadata)
        }

        # Call proxy
        response = await self.client.chat.completions.create(
            messages=messages,
            extra_headers=headers,
            **kwargs
        )

        # Convert to ModelOutput
        return self._to_model_output(response)
```

### Using in AgentWorkflowEngine

```python
from rllm.engine import AgentWorkflowEngine

# Create engine with LiteLLM proxy
engine = LiteLLMEngine(proxy_url="http://localhost:8000/v1")

# Use in workflow
workflow_engine = AgentWorkflowEngine(
    rollout_engine=engine,
    # ... other config
)
```

## API Reference

### Endpoints

#### `GET /health`
Health check endpoint.

**Response**: `{"status": "healthy", "service": "rllm-proxy"}`

#### `GET /v1/models`
List available models.

**Response**: OpenAI-compatible models list

#### `POST /v1/chat/completions`
Chat completions endpoint (OpenAI-compatible).

**Headers**:
- `X-RLLM-Session` (optional): Session ID for tracing
- `X-RLLM-Metadata` (optional): JSON metadata

**Body**: OpenAI chat completion request

**Response**: OpenAI chat completion response + tracing

#### `POST /v1/completions`
Text completions endpoint (OpenAI-compatible).

Similar to chat completions.

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Custom Middleware

Edit `middleware.py` and add your middleware to the app in `server.py`:

```python
from .custom_middleware import MyMiddleware

app.add_middleware(MyMiddleware)
```

## Open Questions & TODOs

1. **Logprobs Cost**: Should logprobs be enabled by default? Consider provider costs.

2. **Streaming Traces**: Current implementation collects streaming chunks. Consider memory usage for long generations.

3. **API Key Security**: In shared proxy environments, ensure secure key passthrough without logging.

4. **Metrics**: Add Prometheus metrics for request counts, latencies, errors.

5. **Integration with EpisodeLogger**: Bridge proxy traces with RLLM's existing episode logging system.

## License

See main RLLM repository license.

## Support

For issues and questions:
- GitHub Issues: [rllm/issues](https://github.com/togethercomputer/rllm/issues)
- Documentation: [RLLM Docs](https://rllm.readthedocs.io)
