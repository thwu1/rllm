#!/usr/bin/env python3
"""Standalone test for LiteLLM proxy tracer integration with OpenAI.

This test:
1. Starts LiteLLM proxy server in subprocess
2. Reloads it with OpenAI configuration
3. Tests trace collection through proxy
4. Validates flush mechanism
5. Cleans up automatically

No manual proxy startup needed!

Prerequisites:
    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/sdk/test_proxy_tracer_standalone.py [--db-path PATH]

Example:
    python examples/sdk/test_proxy_tracer_standalone.py \
        --db-path /tmp/test_proxy.db
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from rllm.sdk import get_chat_client_async, session
from rllm.sdk.store import SqliteTraceStore


class ProxyManager:
    """Manages LiteLLM proxy server lifecycle for testing."""

    def __init__(self, db_path: str, proxy_port: int = 4000, admin_token: str = "test-admin-token"):
        self.db_path = db_path
        self.proxy_port = proxy_port
        self.admin_token = admin_token
        self.proxy_url = f"http://127.0.0.1:{proxy_port}"
        self.process = None
        self.config_path = None

    def _create_openai_config(self) -> str:
        """Create OpenAI configuration for LiteLLM."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        config = {
            "model_list": [
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": api_key,
                    },
                }
            ]
        }

        # Write config to temp file
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="litellm_test_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)

        self.config_path = path
        return path

    async def start(self):
        """Start LiteLLM proxy server."""
        config_path = self._create_openai_config()

        print(f"Starting LiteLLM proxy on port {self.proxy_port}...")
        print(f"  - Config: {config_path}")
        print(f"  - Database: {self.db_path}")

        # Start proxy subprocess
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rllm.sdk.proxy.litellm_server",
                "--config",
                config_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.proxy_port),
                "--db-path",
                self.db_path,
                "--project",
                "test-project",
                "--admin-token",
                self.admin_token,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for proxy to be ready
        await self._wait_for_ready(timeout=30.0)
        print("✓ Proxy started successfully")

    async def _wait_for_ready(self, timeout: float = 30.0):
        """Wait for proxy server to be ready."""
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(f"{self.proxy_url}/health", timeout=1.0)
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        raise TimeoutError(f"Proxy did not become ready within {timeout}s")

    async def reload_config(self, config_yaml: str):
        """Reload proxy configuration."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.proxy_url}/admin/reload",
                headers={"Authorization": f"Bearer {self.admin_token}"},
                json={"config_yaml": config_yaml},
                timeout=10.0,
            )
            response.raise_for_status()
            print("✓ Proxy configuration reloaded")

    async def flush_tracer(self, timeout: float = 30.0):
        """Flush tracer via proxy endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.proxy_url}/admin/tracer/flush",
                headers={"Authorization": f"Bearer {self.admin_token}"},
                json={"timeout": timeout},
                timeout=timeout + 5.0,
            )
            response.raise_for_status()
            return response.json()

    def stop(self):
        """Stop proxy server."""
        if self.process:
            print("Stopping proxy server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✓ Proxy stopped")

        # Clean up config file
        if self.config_path and os.path.exists(self.config_path):
            os.unlink(self.config_path)


async def test_basic_request(proxy_url: str):
    """Test basic chat completion request."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Chat Completion")
    print("=" * 60)

    client = get_chat_client_async(base_url=f"{proxy_url}/v1", api_key="EMPTY")

    with session(test="basic_request") as sess:
        print(f"Session UID: {sess._uid}")

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=5,
        )

        print(f"✓ Request successful")
        print(f"  - Response ID: {response.id}")
        print(f"  - Content: {response.choices[0].message.content}")

        return response.id, sess._uid


async def test_multiple_requests(proxy_url: str):
    """Test multiple requests in one session."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Requests")
    print("=" * 60)

    client = get_chat_client_async(base_url=f"{proxy_url}/v1", api_key="EMPTY")

    with session(test="multi_request") as sess:
        session_uid = sess._uid
        print(f"Session UID: {session_uid}")

        for i in range(3):
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Count: {i}"}],
                max_tokens=5,
            )
            print(f"  - Request {i+1}: {response.id}")

        print(f"✓ Completed 3 requests")
        return session_uid


async def test_trace_persistence(db_path: str, session_uid: str, proxy_manager: ProxyManager):
    """Test trace persistence after flush."""
    print("\n" + "=" * 60)
    print("TEST 3: Trace Persistence")
    print("=" * 60)

    # Flush tracer
    print("Flushing tracer...")
    result = await proxy_manager.flush_tracer(timeout=30.0)
    print(f"✓ Flush result: {result}")

    # Check database
    await asyncio.sleep(1.0)  # Give SQLite a moment

    store = SqliteTraceStore(db_path=db_path)
    traces = await store.get_by_session_uid(session_uid)

    if traces:
        print(f"✓ Found {len(traces)} trace(s) in database")
        for i, trace in enumerate(traces, 1):
            print(f"  - Trace {i}: {trace.id}")
        return True
    else:
        print("✗ No traces found")
        return False


async def run_all_tests(db_path: str, proxy_port: int):
    """Run all proxy tracer tests."""
    # Clean up existing database
    db_file = Path(db_path)
    if db_file.exists():
        print(f"Removing existing database: {db_path}")
        db_file.unlink()

    print("=" * 60)
    print("LiteLLM Proxy Tracer Standalone Test")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Proxy port: {proxy_port}")
    print()

    # Start proxy
    proxy_manager = ProxyManager(db_path=db_path, proxy_port=proxy_port)

    try:
        await proxy_manager.start()

        proxy_url = proxy_manager.proxy_url
        results = []

        # Test 1: Basic request
        try:
            response_id, session_uid_1 = await test_basic_request(proxy_url)
            results.append(("Basic Request", response_id is not None))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Basic Request", False))

        # Test 2: Multiple requests
        try:
            session_uid_2 = await test_multiple_requests(proxy_url)
            results.append(("Multiple Requests", session_uid_2 is not None))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Multiple Requests", False))

        # Test 3: Trace persistence (use session from test 1)
        try:
            if session_uid_1:
                persist_ok = await test_trace_persistence(db_path, session_uid_1, proxy_manager)
                results.append(("Trace Persistence", persist_ok))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Trace Persistence", False))

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, success in results if success)
        total = len(results)

        for test_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status:<8} {test_name}")

        print()
        print(f"Results: {passed}/{total} tests passed")

        return passed == total

    finally:
        proxy_manager.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test LiteLLM proxy tracer with OpenAI")
    parser.add_argument("--db-path", type=str, default="/tmp/test_proxy_tracer.db", help="SQLite database path")
    parser.add_argument("--proxy-port", type=int, default=4000, help="Proxy port")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        success = await run_all_tests(args.db_path, args.proxy_port)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
