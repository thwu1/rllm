"""RLLM SDK for automatic LLM trace collection and RL training."""

from rllm.sdk.client import RLLMClient
from rllm.sdk.context import get_current_metadata, get_current_session
from rllm.sdk.reward import set_reward, set_reward_async
from rllm.sdk.session import SessionContext
from rllm.sdk.shortcuts import get_chat_client, get_chat_client_async, session
from rllm.sdk.tracing import LLMTracer, get_tracer

__all__ = [
    "RLLMClient",
    "get_current_session",
    "get_current_metadata",
    "SessionContext",
    "session",
    "get_chat_client",
    "get_chat_client_async",
    "LLMTracer",
    "get_tracer",
    "set_reward",
    "set_reward_async",
]
