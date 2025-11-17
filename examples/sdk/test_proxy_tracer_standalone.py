#!/usr/bin/env python3
"""Standalone test for LiteLLM proxy tracer integration with OpenAI.

This test:
1. Starts LiteLLM proxy server using ProxyManager base class
2. Configures it with OpenAI models
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
import sys
from pathlib import Path

from rllm.engine.proxy_manager import ProxyManager
from rllm.sdk import get_chat_client_async, session
from rllm.sdk.store import SqliteTraceStore


def create_openai_config() -> dict:
    """Create OpenAI configuration for LiteLLM.

    Returns:
        Configuration dict with OpenAI model setup.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return {
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


async def test_basic_request(proxy_manager: ProxyManager):
    """Test basic chat completion request."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Chat Completion")
    print("=" * 60)

    proxy_url = proxy_manager.get_proxy_url(include_v1=True)
    client = get_chat_client_async(base_url=proxy_url, api_key="EMPTY")

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


async def test_multiple_requests(proxy_manager: ProxyManager):
    """Test multiple requests in one session."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Requests")
    print("=" * 60)

    proxy_url = proxy_manager.get_proxy_url(include_v1=True)
    client = get_chat_client_async(base_url=proxy_url, api_key="EMPTY")

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


async def run_all_tests(db_path: str, proxy_port: int, admin_token: str):
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

    # Create proxy manager with base class
    proxy_manager = ProxyManager(
        proxy_host="127.0.0.1",
        proxy_port=proxy_port,
        admin_token=admin_token,
    )

    try:
        # Create OpenAI configuration
        config = create_openai_config()

        # Start proxy subprocess with config
        print(f"Starting LiteLLM proxy on port {proxy_port}...")
        print(f"  - Database: {db_path}")

        config_path = proxy_manager.start_proxy_subprocess(
            config=config,
            db_path=db_path,
            project="test-project",
        )
        print(f"  - Config: {config_path}")
        print("✓ Proxy started successfully")

        results = []

        # Test 1: Basic request
        try:
            response_id, session_uid_1 = await test_basic_request(proxy_manager)
            results.append(("Basic Request", response_id is not None))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append(("Basic Request", False))

        # Test 2: Multiple requests
        try:
            session_uid_2 = await test_multiple_requests(proxy_manager)
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
        proxy_manager.shutdown_proxy()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test LiteLLM proxy tracer with OpenAI")
    parser.add_argument("--db-path", type=str, default="/tmp/test_proxy_tracer.db", help="SQLite database path")
    parser.add_argument("--proxy-port", type=int, default=4000, help="Proxy port")
    parser.add_argument("--admin-token", type=str, default="test-admin-token", help="Admin token for proxy")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        success = await run_all_tests(args.db_path, args.proxy_port, args.admin_token)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
