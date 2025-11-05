"""LiteLLM callbacks for parameter injection and tracing."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from rllm.sdk.tracing import LLMTracer


class SamplingParametersCallback(CustomLogger):
    """Inject sampling parameters and metadata before LiteLLM sends requests.

    Adds logprobs and top_logprobs to all requests.
    Only adds return_token_ids for vLLM-compatible backends (not OpenAI/Anthropic).
    Injects metadata from request state if available.
    """

    def __init__(self, add_return_token_ids: bool = False):
        super().__init__()
        self.add_return_token_ids = add_return_token_ids

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or (args[2] if len(args) > 2 else {})
        model = data.get("model", "")

        # Request token-level logprobs; do not force top_logprobs list
        result = {**data, "logprobs": True}

        # Only add return_token_ids if explicitly enabled AND model supports it
        if self.add_return_token_ids and self._supports_token_ids(model):
            result["return_token_ids"] = True

        # Inject metadata from request state if available
        litellm_params = kwargs.get("litellm_params", {})
        proxy_server_request = litellm_params.get("proxy_server_request")
        if proxy_server_request:
            request_state = getattr(proxy_server_request, "state", None)
            if request_state:
                rllm_metadata = getattr(request_state, "rllm_metadata", None)
                if rllm_metadata:
                    result["metadata"] = {**result.get("metadata", {}), **rllm_metadata}

        return result

    @staticmethod
    def _supports_token_ids(model: str) -> bool:
        """Check if model supports return_token_ids parameter.

        OpenAI, Anthropic, and most cloud providers don't support this.
        vLLM and local/self-hosted models typically do.
        """
        model_lower = model.lower()
        # vLLM or self-hosted indicators
        if any(x in model_lower for x in ["vllm", "localhost", "127.0.0.1", "http://"]):
            return True
        # OpenAI models - don't support
        if any(x in model_lower for x in ["gpt-", "openai/", "o1-"]):
            return False
        # Anthropic models - don't support
        if "claude" in model_lower or "anthropic/" in model_lower:
            return False
        # Default: assume cloud provider doesn't support
        return False


class TracingCallback(CustomLogger):
    """Log LLM calls to episodic tracer using LiteLLM success/failure hooks."""

    def __init__(self, tracer: LLMTracer):
        super().__init__()
        self.tracer = tracer

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: float, end_time: float) -> None:
        """Called after successful LLM completion."""
        litellm_params = kwargs.get("litellm_params", {})
        # Whitelist metadata keys to avoid noisy provider internals
        raw_meta = litellm_params.get("metadata", {})
        allowed = {"session_id", "job", "user_api_key_request_route"}
        metadata = {k: v for k, v in raw_meta.items() if k in allowed}

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Compute latency in milliseconds supporting datetime or float inputs
        _delta = end_time - start_time
        if isinstance(_delta, timedelta):
            latency_ms = _delta.total_seconds() * 1000.0
        else:
            latency_ms = float(_delta) * 1000.0

        usage = getattr(response_obj, "usage", None)
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total": getattr(usage, "total_tokens", 0) if usage else 0,
        }

        response_dict = response_obj.model_dump() if hasattr(response_obj, "model_dump") else {}

        self.tracer.log_llm_call(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=response_dict,
            metadata=metadata,
            session_id=metadata.get("session_id"),
            latency_ms=latency_ms,
            tokens=tokens,
        )

    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: float, end_time: float) -> None:
        """Called after failed LLM completion."""
        litellm_params = kwargs.get("litellm_params", {})
        # Whitelist metadata keys to avoid noisy provider internals
        raw_meta = litellm_params.get("metadata", {})
        allowed = {"session_id", "job", "user_api_key_request_route"}
        metadata = {k: v for k, v in raw_meta.items() if k in allowed}

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        # Compute latency in milliseconds supporting datetime or float inputs
        _delta = end_time - start_time
        if isinstance(_delta, timedelta):
            latency_ms = _delta.total_seconds() * 1000.0
        else:
            latency_ms = float(_delta) * 1000.0

        error_info = {
            "error": str(response_obj) if response_obj else "Unknown error",
            "type": type(response_obj).__name__ if response_obj else "UnknownError",
        }

        self.tracer.log_llm_call(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=error_info,
            metadata={**metadata, "error": True},
            session_id=metadata.get("session_id"),
            latency_ms=latency_ms,
            tokens={"prompt": 0, "completion": 0, "total": 0},
        )
