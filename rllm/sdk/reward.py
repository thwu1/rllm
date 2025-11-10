"""Simple reward assignment for LLM responses."""

from __future__ import annotations

import os
from typing import Any

import httpx


def set_reward(
    response_or_id: str | Any,
    reward: float,
    metadata: dict[str, Any] | None = None,
    *,
    proxy_url: str | None = None,
    admin_token: str | None = None,
) -> dict[str, Any]:
    """Set reward for an LLM response (synchronous).

    The reward is stored with a context_id of "reward_<response_id>" to avoid
    overwriting the original LLM trace in the database. The original response_id
    is automatically included in metadata as "response_context_id" for easy linking.

    Args:
        response_or_id: Either a string context_id or a response object with .id attribute
        reward: Reward score to assign
        metadata: Optional metadata associated with the reward (will be merged with
                  response_context_id)
        proxy_url: Proxy URL (defaults to LITELLM_PROXY_URL env var or http://localhost:4000)
        admin_token: Admin token (defaults to LITELLM_PROXY_ADMIN_TOKEN env var)

    Returns:
        Response from the server

    Example:
        ```python
        from rllm.sdk import get_chat_client, session, set_reward

        chat = get_chat_client(api_key="token-123", base_url="http://localhost:4000/v1")

        with session(job="nightly"):
            response = chat.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )

            # Later, after evaluation
            set_reward(response, reward=1.0)
            # or
            set_reward(response.id, reward=1.0, metadata={"quality": "good"})
        ```
    """
    # Extract context_id
    if isinstance(response_or_id, str):
        base_id = response_or_id
    else:
        base_id = getattr(response_or_id, "id", None)
        if not base_id:
            raise ValueError(f"Cannot extract id from response object: {type(response_or_id)}")

    # Add prefix to avoid overwriting the original trace
    context_id = f"reward_{base_id}"

    # Get proxy URL and admin token
    proxy_url = proxy_url or os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    admin_token = admin_token or os.getenv("LITELLM_PROXY_ADMIN_TOKEN", "my-shared-secret")

    # Build request
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    # Build payload with reward and original response context_id in metadata
    payload_metadata = {"response_context_id": base_id}
    if metadata:
        payload_metadata.update(metadata)

    payload = {"context_id": context_id, "reward": reward, "metadata": payload_metadata}

    # Send request (synchronous)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(f"{proxy_url.rstrip('/')}/admin/publish-reward", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


async def set_reward_async(
    response_or_id: str | Any,
    reward: float,
    metadata: dict[str, Any] | None = None,
    *,
    proxy_url: str | None = None,
    admin_token: str | None = None,
) -> dict[str, Any]:
    """Set reward for an LLM response (asynchronous).

    The reward is stored with a context_id of "reward_<response_id>" to avoid
    overwriting the original LLM trace in the database. The original response_id
    is automatically included in metadata as "response_context_id" for easy linking.

    Args:
        response_or_id: Either a string context_id or a response object with .id attribute
        reward: Reward score to assign
        metadata: Optional metadata associated with the reward (will be merged with
                  response_context_id)
        proxy_url: Proxy URL (defaults to LITELLM_PROXY_URL env var or http://localhost:4000)
        admin_token: Admin token (defaults to LITELLM_PROXY_ADMIN_TOKEN env var)

    Returns:
        Response from the server

    Example:
        ```python
        from rllm.sdk import get_chat_client, session, set_reward_async

        chat = get_chat_client(api_key="token-123", base_url="http://localhost:4000/v1")

        with session(job="nightly"):
            response = chat.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
            )

            # Later, after evaluation
            await set_reward_async(response, reward=1.0)
            # or
            await set_reward_async(response.id, reward=1.0, metadata={"quality": "good"})
        ```
    """
    # Extract context_id
    if isinstance(response_or_id, str):
        base_id = response_or_id
    else:
        base_id = getattr(response_or_id, "id", None)
        if not base_id:
            raise ValueError(f"Cannot extract id from response object: {type(response_or_id)}")

    # Add prefix to avoid overwriting the original trace
    context_id = f"reward_{base_id}"

    # Get proxy URL and admin token
    proxy_url = proxy_url or os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    admin_token = admin_token or os.getenv("LITELLM_PROXY_ADMIN_TOKEN", "my-shared-secret")

    # Build request
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    # Build payload with reward and original response context_id in metadata
    payload_metadata = {"response_context_id": base_id}
    if metadata:
        payload_metadata.update(metadata)

    payload = {"context_id": context_id, "reward": reward, "metadata": payload_metadata}

    # Send request (asynchronous)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{proxy_url.rstrip('/')}/admin/publish-reward", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
