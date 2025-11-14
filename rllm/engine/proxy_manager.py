"""LiteLLM proxy manager for VERL rollout engines.

This module provides utilities to:
1. Extract vLLM server addresses from VERL rollout engines
2. Configure LiteLLM proxy with multiple vLLM backends for load balancing
3. Provide a unified OpenAI-compatible endpoint with metadata routing
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import requests
import yaml

from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine

if TYPE_CHECKING:
    from rllm.sdk.tracers import TracerProtocol

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
        tracer: "TracerProtocol | None" = None,
        auto_instrument_vllm: bool = True,
        proxy_access_log: bool = False,
    ):
        """Initialize the proxy manager.

        Args:
            rollout_engine: The rollout engine (must be VerlEngine)
            model_name: Model name to expose via the proxy
            proxy_host: Host to bind the proxy server
            proxy_port: Port to bind the proxy server
            tracer: Optional tracer for logging
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

        # Subprocess state
        self._proxy_process: subprocess.Popen | None = None

    async def flush_tracer(self, timeout: float = 30.0) -> bool:
        """Ask LiteLLM proxy to flush the tracer queue.

        This ensures all queued traces are persisted to storage before returning.
        Useful for synchronization before collecting traces from the database.

        Args:
            timeout: Maximum time to wait for flush operation (default: 30.0 seconds)

        Returns:
            True if flush succeeds, False otherwise
        """
        url = f"http://{self.proxy_host}:{self.proxy_port}/admin/flush-tracer"
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.admin_token}"

        try:
            request_timeout = aiohttp.ClientTimeout(total=timeout + 5.0)  # Add buffer for network overhead
            async with aiohttp.ClientSession(timeout=request_timeout) as session:
                async with session.post(url, json={"timeout": timeout}, headers=headers) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    logger.info("Tracer flush succeeded: %s", result)
            return True
        except Exception as exc:
            logger.warning("Failed to flush tracer via proxy: %s", exc)
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
            # SamplingParametersCallback will detect vLLM from litellm_params (hosted_vllm prefix)
            model_list.append(
                {
                    "model_name": self.model_name,
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

    def reload_external_proxy(
        self,
        reload_url: str | None = None,
        inline_payload: bool = True,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Ask an external LiteLLM proxy (started via launch_litellm.sh or python -m rllm.sdk.proxy.litellm_server) to reload config.

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

    def start_proxy_subprocess(self, db_path: str | None = None, project: str | None = None) -> None:
        """Start LiteLLM proxy as subprocess (no GIL contention).

        Args:
            db_path: Path to SQLite database for tracer
            project: Project name for tracer namespace
        """
        if self._proxy_process is not None:
            logger.warning("Proxy subprocess already running")
            return

        if not self._config_snapshot_path or not os.path.exists(self._config_snapshot_path):
            raise RuntimeError("Config snapshot not available. Cannot start proxy.")

        # Build command to run proxy server as module
        cmd = [
            sys.executable,
            "-m",
            "rllm.sdk.proxy.litellm_server",
            "--config",
            self._config_snapshot_path,
            "--host",
            self.proxy_host,
            "--port",
            str(self.proxy_port),
        ]

        # Only add admin token if one is provided
        if self.admin_token:
            cmd.extend(["--admin-token", self.admin_token])

        if db_path:
            cmd.extend(["--db-path", db_path])
        if project:
            cmd.extend(["--project", project])

        # Start subprocess
        logger.info(f"Starting proxy subprocess: {' '.join(cmd)}")
        self._proxy_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr to avoid pipe blocking
        )

        # Wait for server to start, then send config via reload
        try:
            self._wait_for_server_start(timeout=10.0)
            logger.info("Proxy server started, sending configuration...")
            self.reload_external_proxy(inline_payload=True)
            logger.info("Proxy configuration loaded successfully")
        except Exception:
            # Cleanup on failure
            self.shutdown_proxy()
            raise

        # Register cleanup handler
        atexit.register(self.shutdown_proxy)

        logger.info(f"✅ Proxy subprocess ready (PID: {self._proxy_process.pid})")

    def _wait_for_server_start(self, timeout: float = 10.0) -> None:
        """Wait for proxy server process to start accepting connections.

        The server starts but doesn't initialize LiteLLM until we call reload.
        We just need to wait for the basic server to be listening.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If proxy process dies during startup
            TimeoutError: If server doesn't start within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self._proxy_process.poll() is not None:
                exit_code = self._proxy_process.returncode
                raise RuntimeError(f"Proxy process died during startup with exit code {exit_code}")

            # Try to connect to the server (any endpoint will do)
            try:
                resp = requests.get(f"http://{self.proxy_host}:{self.proxy_port}/", timeout=0.5)
                # If we get any response, server is up (even 404 is fine)
                logger.info(f"Proxy server accepting connections")
                return
            except requests.RequestException:
                pass

            time.sleep(0.3)

        raise TimeoutError(f"Proxy server did not start within {timeout}s")

    def shutdown_proxy(self) -> None:
        """Gracefully shutdown proxy subprocess."""
        if self._proxy_process is None:
            return

        logger.info("Shutting down proxy subprocess...")

        # Try graceful shutdown
        self._proxy_process.terminate()
        try:
            self._proxy_process.wait(timeout=5.0)
            logger.info("Proxy shutdown gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't respond
            logger.warning("Proxy did not terminate gracefully, forcing kill")
            self._proxy_process.kill()
            self._proxy_process.wait()

        self._proxy_process = None

    def __repr__(self) -> str:
        mode = "subprocess" if self._proxy_process else "external"
        return f"VerlProxyManager(model={self.model_name}, replicas={len(self._server_addresses)}, proxy={self.get_proxy_url()}, mode={mode})"
