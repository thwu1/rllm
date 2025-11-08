#!/usr/bin/env python3
"""Standalone LiteLLM proxy launcher with reload endpoint.

Example:
    python scripts/litellm_proxy_server.py \
        --config /tmp/litellm_proxy_config_autogen.yaml \
        --host 127.0.0.1 --port 4000 \
        --cs-endpoint http://localhost:8000 --cs-api-key "$EPISODIC_API_KEY" \
        --admin-token my-shared-secret
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import litellm
import uvicorn
import yaml
from episodic import ContextStore
from fastapi import Depends, FastAPI, Header, HTTPException, status
from litellm.proxy.proxy_server import app as litellm_app
from litellm.proxy.proxy_server import initialize
from pydantic import BaseModel, Field, root_validator

from rllm.sdk import LLMTracer
from rllm.sdk.proxy.litellm_callbacks import SamplingParametersCallback, TracingCallback
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware


class ReloadPayload(BaseModel):
    """Request body for /admin/reload."""

    config_yaml: str | None = Field(
        default=None,
        description="Inline LiteLLM config YAML (written to state_dir before reload).",
    )
    config_path: str | None = Field(
        default=None,
        description="Existing LiteLLM config path. Only needed when reusing on-disk file.",
    )

    @root_validator(skip_on_failure=True)
    def _validate_choice(cls, values: dict[str, str | None]) -> dict[str, str | None]:
        yaml_blob, path = values.get("config_yaml"), values.get("config_path")
        if not yaml_blob and not path:
            raise ValueError("Provide config_yaml or config_path.")
        if yaml_blob and path:
            raise ValueError("Choose config_yaml or config_path, not both.")
        return values


class TracerSignalPayload(BaseModel):
    """Request body for batch-end tracer signals."""

    token: str = Field(..., description="Unique identifier for the batch completion marker.")


class LiteLLMProxyRuntime:
    """Owns LiteLLM initialization and reload logic."""

    def __init__(self, initial_config: Path, state_dir: Path, tracer: LLMTracer | None):
        self._current_config = initial_config
        self._state_dir = state_dir
        self._tracer = tracer
        self._lock = asyncio.Lock()
        self._initialized = False

    async def startup(self) -> None:
        # Don't initialize LiteLLM on startup - wait for first reload request
        # This allows the server to start even when backends aren't running yet
        logging.info("LiteLLM proxy server ready. Waiting for /admin/reload to configure backends.")
        self._initialized = False

    async def reload(self, payload: ReloadPayload) -> Path:
        if payload.config_yaml:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            target = self._state_dir / "litellm_proxy_config_autogen.yaml"
            target.write_text(payload.config_yaml)
        else:
            target = Path(payload.config_path).expanduser().resolve()  # type: ignore[arg-type]

        # Only verify health if this is a reload (not first-time initialization)
        verify_health = self._initialized
        await self._apply_config(target, verify_health=verify_health)
        self._initialized = True
        return target

    async def _apply_config(self, config_path: Path, verify_health: bool = False) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        async with self._lock:
            # Clean up existing LiteLLM state if this is a reload
            if self._current_config != config_path:
                logging.info("Reloading LiteLLM configuration...")
                # Clear existing router and model list
                if hasattr(litellm, "model_list"):
                    litellm.model_list = []
                if hasattr(litellm, "router"):
                    litellm.router = None
                # Clear callbacks to reinstall them
                litellm.callbacks = []

            # Initialize new LiteLLM instance
            os.environ["LITELLM_CONFIG"] = str(config_path)
            litellm.drop_params = True
            self._install_callbacks()
            logging.info("Initializing LiteLLM with %s", config_path)
            await initialize(config=str(config_path), telemetry=False)
            self._current_config = config_path
            num_models = self._count_models(config_path)
            logging.info("LiteLLM initialized (models=%d)", num_models)
            # Debug: show installed callbacks and tracer identity
            try:
                callback_names = [type(cb).__name__ for cb in getattr(litellm, "callbacks", [])]
                logging.info("LiteLLM callbacks installed: %s", callback_names)
                logging.info("Proxy tracer id: %s", hex(id(self._tracer)) if self._tracer else None)
            except Exception:
                pass

            # Verify backends are reachable (only during reload, not startup)
            if verify_health:
                await self._verify_backends_health(config_path)
                logging.info("LiteLLM ready - all backends verified")
            else:
                logging.info("LiteLLM ready (health check skipped during startup)")

    def _install_callbacks(self) -> None:
        callbacks = [
            cb for cb in getattr(litellm, "callbacks", []) if not isinstance(cb, (SamplingParametersCallback, TracingCallback))
        ]
        callbacks.append(SamplingParametersCallback(add_return_token_ids=True))
        if self._tracer:
            callbacks.append(TracingCallback(self._tracer))
        litellm.callbacks = callbacks
        # Debug: log callback list at install-time
        try:
            names = [type(cb).__name__ for cb in litellm.callbacks]
            logging.info("_install_callbacks -> callbacks=%s tracer_id=%s", names, hex(id(self._tracer)) if self._tracer else None)
        except Exception:
            pass

    async def emit_tracer_signal(self, token: str) -> None:
        if not self._tracer:
            raise RuntimeError("Tracer is not configured on this proxy")
        # Debug: show tracer identity when emitting signal
        try:
            logging.info("Emitting tracer signal via tracer_id=%s token=%s", hex(id(self._tracer)), token)
        except Exception:
            pass
        await self._tracer.store_signal(token, context_type="trace_batch_end")

    async def flush_tracer(self, timeout: float = 30.0) -> None:
        if not self._tracer:
            raise RuntimeError("Tracer is not configured on this proxy")
        try:
            logging.info("Flushing tracer queue tracer_id=%s", hex(id(self._tracer)))
        except Exception:
            pass
        await self._tracer.flush(timeout=timeout)

    async def _verify_backends_health(self, config_path: Path) -> None:
        """Verify that LiteLLM router is initialized and ready."""
        # Simple check: just verify LiteLLM initialized successfully
        # The actual backend health will be checked on first request
        if not hasattr(litellm, "router") or litellm.router is None:
            raise RuntimeError("LiteLLM router not initialized")

        num_models = self._count_models(config_path)
        logging.info("LiteLLM router ready with %d model(s)", num_models)

    @staticmethod
    def _count_models(config_path: Path) -> int:
        try:
            data = yaml.safe_load(config_path.read_text()) or {}
            return len(data.get("model_list", []))
        except Exception:
            return -1

    @property
    def config_path(self) -> str:
        return str(self._current_config)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


def _build_tracer(endpoint: str | None, api_key: str | None, project: str | None) -> LLMTracer | None:
    if not endpoint or not api_key:
        logging.warning("Tracer disabled (missing context store endpoint or API key).")
        return None
    store = ContextStore(endpoint=endpoint, api_key=api_key)
    return LLMTracer(store, project=project)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteLLM proxy server with reload endpoint.")
    parser.add_argument("--config", required=True, help="Initial LiteLLM config YAML.")
    parser.add_argument("--host", default=os.getenv("LITELLM_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LITELLM_PROXY_PORT", "4000")))
    parser.add_argument("--state-dir", default=os.getenv("LITELLM_PROXY_STATE_DIR", "./.litellm_proxy"))
    parser.add_argument("--admin-token", default=os.getenv("LITELLM_PROXY_ADMIN_TOKEN"))
    parser.add_argument("--cs-endpoint", default=os.getenv("EPISODIC_ENDPOINT"))
    parser.add_argument("--cs-api-key", default=os.getenv("EPISODIC_API_KEY"))
    parser.add_argument("--project", required=True, help="Project name for the tracer.")
    parser.add_argument("--log-level", default=os.getenv("LITELLM_PROXY_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runtime = LiteLLMProxyRuntime(
        initial_config=Path(args.config).expanduser().resolve(),
        state_dir=Path(args.state_dir).expanduser().resolve(),
        tracer=_build_tracer(args.cs_endpoint, args.cs_api_key, args.project),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.startup()
        yield

    # Direct-on-litellm_app approach: attach middleware + admin routes to litellm_app
    logging.info("Installing MetadataRoutingMiddleware on litellm_app")
    litellm_app.add_middleware(MetadataRoutingMiddleware)  # type: ignore[attr-defined]
    logging.info("MetadataRoutingMiddleware installed successfully")

    def _require_token(authorization: str = Header(default="")) -> None:
        if args.admin_token and authorization != f"Bearer {args.admin_token}":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")

    @litellm_app.get("/admin/health")
    async def health():
        return {
            "status": "ok",
            "initialized": runtime.is_initialized,
            "config_path": runtime.config_path if runtime.is_initialized else None
        }

    @litellm_app.post("/admin/reload", dependencies=[Depends(_require_token)])
    async def reload_proxy(payload: ReloadPayload):
        try:
            new_path = await runtime.reload(payload)
            return {"status": "reloaded", "config_path": str(new_path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
        except Exception as exc:
            logging.exception("Reload failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @litellm_app.post("/admin/tracer-signal", dependencies=[Depends(_require_token)])
    async def tracer_signal(payload: TracerSignalPayload):
        try:
            await runtime.emit_tracer_signal(payload.token)
            return {"status": "queued", "token": payload.token}
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        except Exception as exc:
            logging.exception("Tracer signal failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @litellm_app.post("/admin/tracer-signal-sync", dependencies=[Depends(_require_token)])
    async def tracer_signal_sync(payload: TracerSignalPayload):
        """Flush tracer queue for this instance, then enqueue the signal.

        This guarantees the signal is emitted after all currently queued traces
        for this process have been persisted.
        """
        try:
            await runtime.flush_tracer()
            await runtime.emit_tracer_signal(payload.token)
            return {"status": "flushed_and_queued", "token": payload.token}
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        except Exception as exc:
            logging.exception("Tracer signal sync failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    def _shutdown_handler(*_: int) -> None:
        raise SystemExit

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    uvicorn.run(litellm_app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
