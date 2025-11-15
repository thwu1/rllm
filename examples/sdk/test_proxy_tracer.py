#!/usr/bin/env python3
"""Test script for LiteLLM proxy tracer integration.

This script tests the tracer integration with the LiteLLM proxy server,
including the TracingCallback and flush endpoint.

Prerequisites:
    1. Start LiteLLM proxy with OpenAI config:
       export OPENAI_API_KEY="sk-..."
       cd examples/omni_trainer
       ./start_proxy_openai.sh

Usage:
    python examples/omni_trainer/test_proxy_tracer.py [--proxy-url URL] [--db-path PATH]

Example:
    python examples/omni_trainer/test_proxy_tracer.py \
        --proxy-url http://localhost:4000 \
        --db-path /tmp/rllm_test.db
"""

import argparse
import asyncio
import sys
import time

import httpx

from rllm.sdk import get_chat_client_async, session
from rllm.sdk.store import SqliteTraceStore


async def test_proxy_health(proxy_url: str):
    """Test proxy server health."""
    print("\n" + "=" * 60)
    print("TEST 1: Proxy Server Health")
    print("=" * 60)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{proxy_url}/health")
            if response.status_code == 200:
                print(f"✓ Proxy is healthy at {proxy_url}")
                print(f"  - Status: {response.status_code}")
                return True
            else:
                print(f"✗ Proxy health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Failed to connect to proxy: {e}")
        print(f"  - Make sure proxy is running at {proxy_url}")
        return False


async def test_simple_request(proxy_url: str):
    """Test a simple chat completion request."""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Chat Completion")
    print("=" * 60)

    client = get_chat_client_async(base_url=f"{proxy_url}/v1", api_key="EMPTY")

    try:
        with session(test="proxy_tracer") as sess:
            print(f"Session UID: {sess._uid}")
            print(f"Session name: {sess._name}")

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                max_tokens=10,
            )

            print("✓ Request successful")
            print(f"  - Response ID: {response.id}")
            print(f"  - Model: {response.model}")
            print(f"  - Content: {response.choices[0].message.content}")
            print(f"  - Tokens: {response.usage.total_tokens}")

            return response.id, sess._uid

    except Exception as e:
        print(f"✗ Request failed: {e}")
        return None, None


async def test_multiple_requests(proxy_url: str):
    """Test multiple sequential requests."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Sequential Requests")
    print("=" * 60)

    client = get_chat_client_async(base_url=f"{proxy_url}/v1", api_key="EMPTY")
    num_requests = 5
    session_uid = None

    try:
        with session(test="multiple_requests", batch="1") as sess:
            session_uid = sess._uid
            print(f"Session UID: {session_uid}")
            print(f"Sending {num_requests} requests...")

            start_time = time.time()
            for i in range(num_requests):
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Count: {i}"}],
                    max_tokens=5,
                )
                print(f"  - Request {i+1}: {response.id}")

            elapsed = time.time() - start_time
            print(f"✓ Completed {num_requests} requests in {elapsed:.2f}s")

            return session_uid

    except Exception as e:
        print(f"✗ Requests failed: {e}")
        return None


async def test_flush_endpoint(proxy_url: str, admin_token: str = "my-shared-secret"):
    """Test the proxy flush endpoint."""
    print("\n" + "=" * 60)
    print("TEST 4: Proxy Flush Endpoint")
    print("=" * 60)

    try:
        async with httpx.AsyncClient() as client:
            print("Calling flush endpoint...")
            start_time = time.time()

            response = await client.post(
                f"{proxy_url}/admin/tracer/flush",
                headers={"Authorization": f"Bearer {admin_token}"},
                json={"timeout": 30.0},
                timeout=35.0,
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Flush succeeded in {elapsed:.2f}s")
                print(f"  - Response: {result}")
                return True
            else:
                print(f"✗ Flush failed: {response.status_code}")
                print(f"  - Response: {response.text}")
                return False

    except Exception as e:
        print(f"✗ Flush request failed: {e}")
        return False


async def test_trace_persistence(db_path: str, session_uid: str):
    """Test that traces were persisted to SQLite."""
    print("\n" + "=" * 60)
    print("TEST 5: Trace Persistence")
    print("=" * 60)

    try:
        store = SqliteTraceStore(db_path=db_path)
        print(f"Checking database: {db_path}")
        print(f"Looking for session: {session_uid}")

        # Wait a bit for async storage to complete
        await asyncio.sleep(1.0)

        traces = await store.get_by_session_uid(session_uid)

        if traces:
            print(f"✓ Found {len(traces)} trace(s) in database")
            for i, trace in enumerate(traces, 1):
                print(f"  - Trace {i}:")
                print(f"    - ID: {trace.id}")
                print(f"    - Model: {trace.data.get('model', 'N/A')}")
                print(f"    - Session: {trace.data.get('session_name', 'N/A')}")
                print(f"    - Tokens: {trace.data.get('tokens', {}).get('total', 'N/A')}")
            return True
        else:
            print("✗ No traces found in database")
            print("  - Check that proxy is configured with correct db-path")
            print("  - Check that flush completed successfully")
            return False

    except Exception as e:
        print(f"✗ Failed to check persistence: {e}")
        return False


async def test_concurrent_requests(proxy_url: str):
    """Test concurrent requests from multiple sessions."""
    print("\n" + "=" * 60)
    print("TEST 6: Concurrent Requests")
    print("=" * 60)

    client = get_chat_client_async(base_url=f"{proxy_url}/v1", api_key="EMPTY")

    async def make_request_in_session(session_name: str, num_requests: int):
        """Make requests within a session."""
        with session(test=session_name) as sess:
            for i in range(num_requests):
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"{session_name} request {i}"}],
                    max_tokens=5,
                )
            return sess._uid

    try:
        num_sessions = 3
        requests_per_session = 2

        print(f"Launching {num_sessions} concurrent sessions...")
        print(f"Each session will make {requests_per_session} requests...")

        start_time = time.time()
        tasks = [make_request_in_session(f"concurrent-{i}", requests_per_session) for i in range(num_sessions)]
        session_uids = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        total_requests = num_sessions * requests_per_session
        print(f"✓ Completed {total_requests} requests across {num_sessions} sessions in {elapsed:.2f}s")
        print(f"  - Rate: {total_requests/elapsed:.1f} req/s")

        return session_uids

    except Exception as e:
        print(f"✗ Concurrent requests failed: {e}")
        return None


async def main():
    """Run all proxy tracer integration tests."""
    parser = argparse.ArgumentParser(description="Test LiteLLM proxy tracer integration")
    parser.add_argument("--proxy-url", type=str, default="http://localhost:4000", help="LiteLLM proxy URL")
    parser.add_argument("--db-path", type=str, default="/tmp/rllm_test.db", help="SQLite database path")
    parser.add_argument("--admin-token", type=str, default="my-shared-secret", help="Proxy admin token")
    args = parser.parse_args()

    print("=" * 60)
    print("LiteLLM Proxy Tracer Integration Test Suite")
    print("=" * 60)
    print(f"Proxy URL: {args.proxy_url}")
    print(f"Database: {args.db_path}")
    print()

    results = []

    # Test 1: Proxy health
    health_ok = await test_proxy_health(args.proxy_url)
    results.append(("Proxy Health", health_ok))

    if not health_ok:
        print("\n✗ Proxy is not running. Please start it first:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  cd examples/omni_trainer")
        print("  ./start_proxy_openai.sh")
        sys.exit(1)

    # Test 2: Simple request
    response_id, session_uid = await test_simple_request(args.proxy_url)
    results.append(("Simple Request", response_id is not None))

    # Test 3: Multiple requests
    multi_session_uid = await test_multiple_requests(args.proxy_url)
    results.append(("Multiple Requests", multi_session_uid is not None))

    # Test 4: Flush endpoint
    flush_ok = await test_flush_endpoint(args.proxy_url, args.admin_token)
    results.append(("Flush Endpoint", flush_ok))

    # Test 5: Trace persistence (use session from test 2 if available)
    if session_uid:
        persist_ok = await test_trace_persistence(args.db_path, session_uid)
        results.append(("Trace Persistence", persist_ok))

    # Test 6: Concurrent requests
    concurrent_uids = await test_concurrent_requests(args.proxy_url)
    results.append(("Concurrent Requests", concurrent_uids is not None))

    # Final flush
    print("\n" + "=" * 60)
    print("Final flush to ensure all traces are persisted...")
    print("=" * 60)
    await test_flush_endpoint(args.proxy_url, args.admin_token)

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

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
