"""FastAPI app mounting LiteLLM proxy with rLLM metadata middleware.

Run with:
  export OPENAI_API_KEY="sk-your-openai-key"
  export LITELLM_CONFIG="examples/proxy_demo/litellm_proxy_config.yaml"
  python examples/proxy_demo/proxy_app.py

Or via uvicorn (from repo root):
  export OPENAI_API_KEY="sk-your-openai-key"
  export LITELLM_CONFIG="examples/proxy_demo/litellm_proxy_config.yaml"
  uvicorn examples.proxy_demo.proxy_app:app --host 127.0.0.1 --port 4000
"""

from __future__ import annotations

import os

# CRITICAL: Set LiteLLM config path BEFORE importing proxy_server
# LiteLLM reads this at import time to initialize models
default_config = os.path.join(os.path.dirname(__file__), "litellm_proxy_config.yaml")
os.environ["LITELLM_CONFIG"] = default_config

print(f"Using LiteLLM config: {default_config}")

# ruff: noqa: E402
# Imports must come after setting LITELLM_CONFIG environment variable
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI
from litellm.proxy.proxy_server import app as litellm_app
from litellm.proxy.proxy_server import initialize

from episodic import ContextStore, LLMTracer
from rllm.sdk.proxy.litellm_callbacks import SamplingParametersCallback
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

# Configuration via environment variables for convenience
EPISODIC_ENDPOINT = os.getenv("EPISODIC_ENDPOINT", "http://localhost:8000")
EPISODIC_API_KEY = os.getenv("EPISODIC_API_KEY", "")
PROJECT = os.getenv("RLLM_PROJECT", "proxy-demo")

# Configure the episodic tracer (points to your context store)
context_store = ContextStore(endpoint=EPISODIC_ENDPOINT, api_key=EPISODIC_API_KEY)
tracer = LLMTracer(context_store, project=PROJECT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LiteLLM proxy on startup."""
    litellm.drop_params = True
    litellm.callbacks.append(SamplingParametersCallback())
    await initialize(config=default_config, telemetry=False)
    yield


# Set up our FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Attach the rLLM middleware so every request is decoded and logged
app.add_middleware(MetadataRoutingMiddleware, tracer=tracer)

# Mount LiteLLM at root after middleware is added
app.mount("/", litellm_app)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "4000"))
    uvicorn.run(app, host=host, port=port)
