"""Minimal proxy + client demo wiring the rLLM metadata middleware."""

from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI
from litellm.proxy.proxy_server import app as litellm_app

from rllm.sdk import RLLMClient, get_tracer
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

tracer = get_tracer(project="proxy-demo", endpoint="http://localhost:8000", api_key="your-api-key-here")


def build_proxy_app() -> FastAPI:
    app = FastAPI()
    app.mount("/", litellm_app)
    app.add_middleware(MetadataRoutingMiddleware, tracer=tracer)
    return app


async def run_client() -> None:
    client = RLLMClient(
        api_key="sk-demo",
        base_url="http://127.0.0.1:4000/v1",
        project="proxy-demo",
    )

    with client.session(session_id="demo-session", job="quickstart"):
        chat = client.get_chat_client()
        try:
            response = chat.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello proxy!"}],
            )
            print("Client response:", response)
        except Exception as exc:
            print("Upstream call failed (expected if backend is not configured):", exc)


async def main() -> None:
    app = build_proxy_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=4000, log_level="info")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.0)  # give the server time to boot

    try:
        await run_client()
    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
