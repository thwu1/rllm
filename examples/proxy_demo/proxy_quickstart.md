# Proxy Integration Quickstart

This walkthrough shows how to plug the rLLM SDK into a LiteLLM-style proxy so that chat traffic carries rich session metadata and every call is logged through `LLMTracer`.

## 1. Install prerequisites

```bash
pip install fastapi uvicorn litellm episodic
```

## 2. Start a proxy with the rLLM middleware

Create `proxy_app.py`:

```python
from fastapi import FastAPI
from litellm.proxy.proxy_server import app as litellm_app
from episodic import ContextStore, LLMTracer
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

# Set up LiteLLM's FastAPI app
app = FastAPI()
app.mount("/", litellm_app)

# Configure the episodic tracer (point to your context store)
context_store = ContextStore(endpoint="http://localhost:8000", api_key="your-cs-key")
tracer = LLMTracer(context_store, project="proxy-demo")

# Attach the rLLM middleware so every request is decoded + logged.
app.add_middleware(MetadataRoutingMiddleware, tracer=tracer)
```

Run the proxy:

```bash
uvicorn proxy_app:app --port 4000
```

LiteLLM reads its configuration from `litellm_proxy_config.yaml`. Make sure that file points to your upstream model providers.

## 3. Call the proxy from the SDK

Inside your python client:

```python
from rllm.sdk import RLLMClient

client = RLLMClient(
    api_key="openai-key",
    base_url="http://localhost:4000/v1",  # proxy base URL
    cs_endpoint="http://localhost:8000",
    cs_api_key="your-cs-key",
    project="proxy-demo",
)

with client.session(session_id="demo-session", job="nightly"):
    chat = client.get_chat_client()
    response = chat.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
```

The wrapped OpenAI client serialises the session metadata, rewrites the base URL to `/meta/{slug}/…`, and your proxy middleware restores the metadata before logging the call with `LLMTracer`.

## 4. Inspect traces

Traces are inserted into the episodic context store under the project you supplied. You can query them with:

```python
records = client.get_session_traces("demo-session")
```

Each trace contains the original request/response payloads, latency, token counts, and any metadata captured by `client.session(...)`.

