"""LiteLLM proxy manager for VERL rollout engines.

This module provides utilities to:
1. Extract vLLM server addresses from VERL rollout engines
2. Configure LiteLLM proxy with multiple vLLM backends for load balancing
3. Provide a unified OpenAI-compatible endpoint with metadata routing
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING, Any

import aiohttp
import requests
import yaml

from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine

if TYPE_CHECKING:
    from rllm.sdk.tracing import LLMTracer

logger = logging.getLogger(__name__)


class VerlProxyManager:
    """Manages LiteLLM proxy configuration for VERL rollout engines.

    This class:
    - Extracts all vLLM server addresses from VERL's AgentLoopManager
    - Generates LiteLLM config with load balancing across all replicas
    - Provides the unified proxy endpoint URL for OpenAI clients

    Example:
        ```python
        # Create VERL engine
        verl_engine = VerlEngine(config, rollout_manager, tokenizer)

        # Setup proxy manager
        proxy_mgr = VerlProxyManager(
            rollout_engine=verl_engine,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            proxy_port=4000
        )

        # Get the unified endpoint
        base_url = proxy_mgr.get_proxy_url()  # http://localhost:4000/v1

        # Use with OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        ```
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        model_name: str,
        proxy_host: str = "127.0.0.1",
        proxy_port: int = 4000,
        admin_token: str | None = None,
        tracer: LLMTracer | None = None,
        auto_instrument_vllm: bool = True,
        proxy_access_log: bool = False,
    ):
        """Initialize the proxy manager.

        Args:
            rollout_engine: The rollout engine (must be VerlEngine)
            model_name: Model name to expose via the proxy
            proxy_host: Host to bind the proxy server
            proxy_port: Port to bind the proxy server
            tracer: Optional LLMTracer for logging
            auto_instrument_vllm: Whether to automatically instrument vLLM for token IDs (default: True)
            proxy_access_log: Whether to emit uvicorn access logs for each request (default: False)
        """
        if not isinstance(rollout_engine, VerlEngine):
            raise TypeError(f"VerlProxyManager only supports VerlEngine, got {type(rollout_engine).__name__}")

        self.rollout_engine = rollout_engine
        self.model_name = model_name
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.tracer = tracer
        self.auto_instrument_vllm = auto_instrument_vllm
        self.proxy_access_log = proxy_access_log
        self.admin_token = admin_token

        # Instrument vLLM if needed (before extracting server addresses)
        if auto_instrument_vllm:
            self._instrument_vllm_servers()

        # Extract server addresses from VERL
        self._server_addresses = self._extract_server_addresses()
        logger.info(f"Extracted {len(self._server_addresses)} vLLM server addresses from VERL")

        # Generate LiteLLM config
        self._config = self._generate_litellm_config()
        self._config_snapshot_path: str | None = None
        self._config_snapshot_path = self._snapshot_config_to_file()

        # Proxy server state
        self._server_thread: threading.Thread | None = None
        self._config_file: str | None = None
        self._is_running = False

    async def emit_batch_end_signal(self, token: str) -> bool:
        """Ask LiteLLM proxy to enqueue a batch-end marker."""

        url = f"http://{self.proxy_host}:{self.proxy_port}/admin/tracer-signal"
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.admin_token}"

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json={"token": token}, headers=headers) as resp:
                    resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Failed to emit batch end signal via proxy: %s", exc)
            return False

    def _instrument_vllm_servers(self) -> None:
        """Instrument vLLM servers to return token IDs.

        WARNING: This method cannot instrument already-running VERL servers!

        VERL servers run in separate Ray worker processes, and monkey patches
        applied in the main process do NOT propagate to Ray workers.

        To enable token IDs for vLLM < 0.10.2, you must instrument BEFORE
        creating the AgentLoopManager. See docs/howto/instrument_verl_vllm_for_token_ids.md

        This method only logs a warning if instrumentation is needed but servers
        are already running.
        """
        try:
            from rllm.engine.vllm_instrumentation import check_vllm_instrumentation_status, get_vllm_token_ids_support

            support = get_vllm_token_ids_support()
            print(f"[PROXY_MANAGER] vLLM token IDs support: {support}")

            # Get detailed status for debugging
            status = check_vllm_instrumentation_status()
            print(f"[PROXY_MANAGER] Detailed instrumentation status: {status}")

            if support == "none":
                logger.warning("vLLM < 0.10.2 detected, but VERL servers are already running. Token IDs will NOT be available! To enable token IDs, call instrument_vllm() BEFORE creating AgentLoopManager. See docs/howto/instrument_verl_vllm_for_token_ids.md for details.")
            elif support == "native":
                logger.info("vLLM >= 0.10.2 detected, token IDs available via native support")
            elif support == "instrumented":
                logger.info("vLLM already instrumented, token IDs should be available")
            else:
                logger.debug("vLLM not available in main process (expected for Ray workers)")
        except Exception as e:
            logger.debug(f"Could not check vLLM instrumentation status: {e}")

    def _extract_server_addresses(self) -> list[str]:
        """Extract all vLLM server addresses from VERL's AgentLoopManager."""
        server_addresses = self.rollout_engine.rollout_manager.server_addresses
        return server_addresses

    def _generate_litellm_config(self) -> dict[str, Any]:
        """Generate LiteLLM configuration with all vLLM replicas.

        Creates a model_list with one entry per vLLM replica for load balancing.
        LiteLLM will automatically round-robin across all entries with the same model_name.
        """
        model_list = []

        for idx, server_address in enumerate(self._server_addresses):
            # Each replica gets its own entry in the model list
            # LiteLLM will load balance across all entries with the same model_name
            # Add "vllm/" prefix so SamplingParametersCallback knows to add return_token_ids
            model_list.append(
                {
                    "model_name": f"vllm/{self.model_name}",
                    "litellm_params": {
                        "model": f"hosted_vllm/{self.model_name}",
                        "api_base": f"http://{server_address}/v1",
                        "drop_params": True,
                    },
                    # Optional: Add replica identifier for debugging
                    "model_info": {
                        "id": f"verl-replica-{idx}",
                        "replica_rank": idx,
                    },
                }
            )

        config = {
            "model_list": model_list,
            "litellm_settings": {
                "drop_params": True,
                "num_retries": 3,
                # Enable load balancing across replicas
                "routing_strategy": "simple-shuffle",
                # # Cooldown policy - back    # for t
                # "allowed_fails": 5,  # Allow 5 failures per minute before cooldown
                # "cooldown_time": 10,  # Cooldown for 10 seconds, then auto-retry
            },
        }

        return config

    def _snapshot_config_to_file(self, directory: str | None = None) -> str | None:
        """Persist the auto-generated config to a readable path for debugging."""

        base_dir = directory or os.getenv("RLLM_PROXY_CONFIG_DIR") or os.getcwd()
        try:
            os.makedirs(base_dir, exist_ok=True)
            snapshot_path = os.path.join(base_dir, "litellm_proxy_config_autogen.yaml")
            with open(snapshot_path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
            self._config_snapshot_path = snapshot_path
            logger.info(f"📄 LiteLLM config snapshot written to {snapshot_path}")
            return snapshot_path
        except Exception as e:
            logger.warning(f"Failed to write LiteLLM config snapshot: {e}")
            return None

    def get_litellm_config(self) -> dict[str, Any]:
        """Get the generated LiteLLM configuration."""
        return self._config

    def get_config_snapshot_path(self) -> str | None:
        """Return the path of the last config snapshot written to disk."""
        return self._config_snapshot_path

    def reload_external_proxy(
        self,
        reload_url: str | None = None,
        inline_payload: bool = True,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Ask an external LiteLLM proxy (started via scripts/litellm_proxy_server.py) to reload config.

        Args:
            reload_url: Full URL to the reload endpoint (default: http://{host}:{port}/admin/reload).
            admin_token: Optional bearer token expected by the proxy.
            inline_payload: When True (default) send YAML directly in the request body.
                            Set to False to reference an on-disk config file.
            timeout: HTTP timeout in seconds.
        """

        url = reload_url or f"http://{self.proxy_host}:{self.proxy_port}/admin/reload"

        if inline_payload:
            payload = {"config_yaml": yaml.dump(self._config, default_flow_style=False)}
        else:
            snapshot_path = self._config_snapshot_path or self._snapshot_config_to_file()
            if not snapshot_path or not os.path.exists(snapshot_path):
                raise RuntimeError("LiteLLM config snapshot is unavailable on disk.")
            payload = {"config_path": snapshot_path}

        headers = {"Content-Type": "application/json"}
        if self.admin_token:
            token = self.admin_token if self.admin_token.lower().startswith("bearer ") else f"Bearer {self.admin_token}"
            headers["Authorization"] = token

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to reload LiteLLM proxy via {url}: {exc}") from exc

        try:
            return resp.json()
        except ValueError:
            return {"status": "ok", "raw": resp.text}

    def get_server_addresses(self) -> list[str]:
        """Get all vLLM server addresses."""
        return self._server_addresses.copy()

    def get_proxy_url(self, include_v1: bool = True) -> str:
        """Get the unified proxy endpoint URL.

        Args:
            include_v1: Whether to include /v1 suffix (default: True)

        Returns:
            Proxy URL (e.g., "http://localhost:4000/v1")
        """
        base = f"http://{self.proxy_host}:{self.proxy_port}"
        return f"{base}/v1" if include_v1 else base

    def get_litellm_model_name(self) -> str:
        """Get the LiteLLM model name (with vllm/ prefix).

        This is the model name that should be used when making API calls to the proxy.
        It includes the "vllm/" prefix so that SamplingParametersCallback knows to
        add return_token_ids=True to requests.

        Returns:
            Model name with vllm/ prefix (e.g., "vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        """
        return f"vllm/{self.model_name}"

    def write_config_file(self, path: str | None = None) -> str:
        """Write LiteLLM config to a YAML file.

        Args:
            path: Optional path to write config. If None, creates a temp file.

        Returns:
            Path to the written config file
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".yaml", prefix="litellm_verl_")
            os.close(fd)
            self._config_file = path
            logger.info(f"📄 Created temp config file: {path}")

        logger.info(f"💾 Writing config with {len(self._config.get('model_list', []))} models to {path}")

        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

        # Verify file was written
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            logger.info(f"✅ Wrote LiteLLM config to {path} ({file_size} bytes)")
        else:
            logger.error(f"❌ Failed to write config file: {path}")

        return path

    def start_proxy_server(self, config_path: str | None = None) -> None:
        """Start the LiteLLM proxy server in a background thread.

        Args:
            config_path: Optional path to config file. If None, writes a temp file.

        Note:
            This is a basic implementation. For production, consider using
            the full proxy setup from examples/proxy_demo/proxy_app.py with
            middleware and callbacks.
        """
        if self._is_running:
            logger.warning("⚠️ Proxy server is already running")
            return

        logger.info("🚀 Starting LiteLLM proxy server...")

        if config_path is None:
            logger.info("📝 No config path provided, creating temp file...")
            config_path = self.write_config_file()

        logger.info(f"📂 Using config file: {config_path}")

        # Import here to avoid hard dependency
        try:
            from contextlib import asynccontextmanager

            import litellm
            import uvicorn
            from fastapi import FastAPI
            from litellm.proxy.proxy_server import app as litellm_app
            from litellm.proxy.proxy_server import initialize
        except ImportError as e:
            raise ImportError("LiteLLM proxy dependencies not installed. Install with: pip install litellm uvicorn fastapi") from e

        logger.info(f"📝 Writing LiteLLM config to {config_path}")

        # Verify config file exists and is valid
        if not os.path.exists(config_path):
            logger.error(f"❌ Config file does not exist: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config to verify and log
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        logger.info(f"📋 Config loaded: {len(config_data.get('model_list', []))} models configured")
        for idx, model in enumerate(config_data.get("model_list", [])):
            logger.info(f"   Model {idx}: {model.get('model_name')} -> {model.get('litellm_params', {}).get('api_base')}")

        # Setup LiteLLM
        litellm.drop_params = True

        # Add callbacks if tracer is provided
        if self.tracer:
            logger.info("🔍 Adding LiteLLM callbacks for tracing")
            from rllm.sdk.proxy.litellm_callbacks import SamplingParametersCallback, TracingCallback

            litellm.callbacks.append(SamplingParametersCallback(add_return_token_ids=True))
            litellm.callbacks.append(TracingCallback(self.tracer))

        # Initialize LiteLLM with lifespan
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f"🚀 Initializing LiteLLM proxy with config: {config_path}")
            # Pass the file path, not the dict!
            await initialize(config=config_path, telemetry=False)
            logger.info("✅ LiteLLM proxy initialized successfully")
            yield
            logger.info("🛑 Shutting down LiteLLM proxy")

        # Create FastAPI app with lifespan
        logger.info("🔧 Creating FastAPI app with lifespan")
        app = FastAPI(lifespan=lifespan)

        # Add metadata routing middleware
        logger.info("🔌 Adding MetadataRoutingMiddleware")
        from rllm.sdk.proxy.middleware import MetadataRoutingMiddleware

        app.add_middleware(MetadataRoutingMiddleware)

        # Mount LiteLLM
        logger.info("📦 Mounting LiteLLM app")
        app.mount("/", litellm_app)

        # Start server in background thread
        logger.info(f"🌐 Starting proxy server at {self.proxy_host}:{self.proxy_port}")

        def run_server():
            logger.info("🔄 Server thread started, launching uvicorn...")
            try:
                uvicorn.run(
                    app,
                    host=self.proxy_host,
                    port=self.proxy_port,
                    log_level="info",
                    access_log=self.proxy_access_log,
                )
            except Exception as e:
                logger.exception(f"❌ Server thread failed: {e}")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self._is_running = True

        # Give the server time to start up
        import time

        logger.info("⏳ Waiting 3 seconds for server to start...")
        time.sleep(3.0)

        logger.info(f"✅ LiteLLM proxy started at {self.get_proxy_url()}")
        logger.info(f"📊 Proxy serving {len(self._server_addresses)} vLLM replicas:")
        for idx, addr in enumerate(self._server_addresses):
            logger.info(f"   [{idx}] {addr}")

    def stop_proxy_server(self) -> None:
        """Stop the proxy server and clean up temp files."""
        if not self._is_running:
            return

        # Note: uvicorn doesn't provide a clean shutdown mechanism in thread mode
        # For production, use a proper process manager
        logger.warning("Proxy server shutdown not fully implemented in thread mode")

        if self._config_file and os.path.exists(self._config_file):
            os.unlink(self._config_file)
            self._config_file = None

        self._is_running = False

    def is_running(self) -> bool:
        """Check if the proxy server is running."""
        return self._is_running

    def __repr__(self) -> str:
        return f"VerlProxyManager(model={self.model_name}, replicas={len(self._server_addresses)}, proxy={self.get_proxy_url()}, running={self._is_running})"
