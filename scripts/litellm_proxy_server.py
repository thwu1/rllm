#!/usr/bin/env python3
"""Standalone LiteLLM proxy launcher with reload endpoint.

Example:
    python scripts/litellm_proxy_server.py \
        --config /tmp/litellm_proxy_config_autogen.yaml \
        --host 127.0.0.1 --port 4000 \
        --db-path ~/.rllm/traces.db --project my-app \
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
from fastapi import Depends, FastAPI, Header, HTTPException, status
from litellm.proxy.proxy_server import app as litellm_app
from litellm.proxy.proxy_server import initialize
from pydantic import BaseModel, Field, root_validator

from rllm.sdk.proxy.litellm_callbacks import SamplingParametersCallback, TracingCallback
from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware
from rllm.sdk.tracers import SqliteTracer


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


class FlushTracerPayload(BaseModel):
    """Request body for flushing tracer queue."""

    timeout: float = Field(default=30.0, description="Maximum time to wait for flush operation in seconds.")


class RewardPayload(BaseModel):
    """Request body for publishing reward scores."""

    context_id: str = Field(..., description="Unique identifier for the context (e.g., trace ID, step ID, or session ID).")
    reward: float = Field(..., description="Reward score to assign to this context.")
    metadata: dict | None = Field(default=None, description="Optional metadata associated with the reward.")


class LiteLLMProxyRuntime:
    """Owns LiteLLM initialization and reload logic."""

    def __init__(self, initial_config: Path, state_dir: Path, tracer: SqliteTracer | None):
        self._current_config = initial_config
        self._state_dir = state_dir
        self._tracer = tracer
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        # Don't initialize LiteLLM on startup - wait for first reload request
        # This allows the server to start even when backends aren't running yet
        logging.info("LiteLLM proxy server ready. Waiting for /admin/reload to configure backends.")

    async def reload(self, payload: ReloadPayload) -> Path:
        if payload.config_yaml:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            target = self._state_dir / "litellm_proxy_config_autogen.yaml"
            target.write_text(payload.config_yaml)
        else:
            target = Path(payload.config_path).expanduser().resolve()  # type: ignore[arg-type]

        # Only verify health if this is a reload (not first-time initialization)
        await self._apply_config(target)
        return target

    async def _apply_config(self, config_path: Path) -> None:
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

    def _install_callbacks(self) -> None:
        callbacks = [cb for cb in getattr(litellm, "callbacks", []) if not isinstance(cb, SamplingParametersCallback | TracingCallback)]
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

    async def flush_tracer(self, timeout: float = 30.0) -> bool:
        """Flush the tracer queue and return success status.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if flush succeeded, False if it timed out or failed

        Raises:
            RuntimeError: If tracer is not configured
        """
        if not self._tracer:
            raise RuntimeError("Tracer is not configured on this proxy")
        try:
            logging.info("Flushing tracer queue tracer_id=%s timeout=%s", hex(id(self._tracer)), timeout)
        except Exception:
            pass
        # Run synchronous flush in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(self._tracer.flush, timeout=timeout)

        # Treat None as success for backward compatibility with tracers that don't return bool
        # Only False explicitly indicates failure
        success = result is not False

        try:
            if success:
                logging.info("Tracer flush succeeded tracer_id=%s (result=%s)", hex(id(self._tracer)), result)
            else:
                logging.warning("Tracer flush failed or timed out tracer_id=%s", hex(id(self._tracer)))
        except Exception:
            pass
        return success

    async def publish_reward(self, context_id: str, reward: float, metadata: dict | None = None) -> None:
        """Publish a reward score to the context store.

        Args:
            context_id: Unique identifier for the context (trace ID, step ID, or session ID).
            reward: Reward score to assign.
            metadata: Optional metadata associated with the reward.
        """
        if not self._tracer:
            raise RuntimeError("Tracer is not configured on this proxy")

        # Store reward as a signal in the context store
        reward_data = {"reward": reward, **(metadata or {})}

        logging.info("Publishing reward=%s for context_id=%s metadata=%s", reward, context_id, metadata)

        await self._tracer.store_signal(context_id=context_id, context_type="reward", data=reward_data, tags=["reward"])

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


def _build_tracer(db_path: str | None, project: str | None) -> SqliteTracer | None:
    if not db_path:
        logging.warning("Tracer disabled (missing database path).")
        return None
    return SqliteTracer(db_path=db_path, namespace=project or "default")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiteLLM proxy server with reload endpoint.")
    parser.add_argument("--config", required=True, help="Initial LiteLLM config YAML.")
    parser.add_argument("--host", default=os.getenv("LITELLM_PROXY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LITELLM_PROXY_PORT", "4000")))
    parser.add_argument("--state-dir", default=os.getenv("LITELLM_PROXY_STATE_DIR", "./.litellm_proxy"))
    parser.add_argument("--admin-token", default=os.getenv("LITELLM_PROXY_ADMIN_TOKEN"))
    parser.add_argument("--db-path", default=os.getenv("SQLITE_DB_PATH", "~/.rllm/traces.db"), help="Path to SQLite database file.")
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME", "default"), help="Project name/namespace for the tracer.")
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
        tracer=_build_tracer(args.db_path, args.project),
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

    @litellm_app.post("/admin/reload", dependencies=[Depends(_require_token)])
    async def reload_proxy(payload: ReloadPayload):
        try:
            new_path = await runtime.reload(payload)
            return {"status": "reloaded", "config_path": str(new_path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("Reload failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @litellm_app.post("/admin/tracer-signal", dependencies=[Depends(_require_token)])
    async def tracer_signal(payload: TracerSignalPayload):
        try:
            await runtime.emit_tracer_signal(payload.token)
            return {"status": "queued", "token": payload.token}
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("Tracer signal failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @litellm_app.post("/admin/flush-tracer", dependencies=[Depends(_require_token)])
    async def flush_tracer(payload: FlushTracerPayload | None = None):
        """Flush the tracer queue to ensure all traces are persisted.

        This endpoint blocks until all queued traces are written to storage,
        ensuring synchronization between the tracer and storage.

        Args:
            payload: Request payload containing timeout parameter

        Returns:
            {"status": "flushed", "timeout": <timeout_used>} on success

        Raises:
            HTTPException 408: If flush times out
            HTTPException 503: If tracer is not configured
            HTTPException 500: If flush fails for other reasons
        """
        if payload is None:
            payload = FlushTracerPayload()
        try:
            success = await runtime.flush_tracer(timeout=payload.timeout)
            if not success:
                # Flush failed or timed out
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"Tracer flush timed out or failed after {payload.timeout}s. Some traces may not be persisted.",
                )
            return {"status": "flushed", "timeout": payload.timeout}
        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("Flush tracer failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @litellm_app.post("/admin/publish-reward", dependencies=[Depends(_require_token)])
    async def publish_reward(payload: RewardPayload):
        """Publish a reward score for a specific context.

        This endpoint allows external services to publish reward scores that can be
        associated with LLM traces, steps, or sessions. The reward is stored in the
        context store with context_type="reward" and can be retrieved later.

        Example:
            ```bash
            curl -X POST http://localhost:4000/admin/publish-reward \\
                -H "Authorization: Bearer my-shared-secret" \\
                -H "Content-Type: application/json" \\
                -d '{
                    "context_id": "task_123:0:1_solver_step0",
                    "reward": 1.0,
                    "metadata": {"is_correct": true, "step_type": "solver"}
                }'
            ```
        """
        try:
            await runtime.publish_reward(context_id=payload.context_id, reward=payload.reward, metadata=payload.metadata)
            return {"status": "published", "context_id": payload.context_id, "reward": payload.reward}
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logging.exception("Publish reward failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    def _shutdown_handler(*_: int) -> None:
        raise SystemExit

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    uvicorn.run(litellm_app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
