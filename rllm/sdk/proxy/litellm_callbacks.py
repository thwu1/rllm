"""LiteLLM callbacks for parameter injection and tracing."""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream

from rllm.sdk.tracers import EpisodicTracer, SqliteTracer


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
        # result = {**data, "logprobs": True}
        result = {**data}

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
    """Log LLM calls to episodic tracer right after provider response.

    Uses LiteLLM's async_post_call_success_hook which fires at the proxy level,
    once per HTTP request, immediately before the response is sent to the user.
    This guarantees we log with the actual response object, while still being
    pre-send, and avoids duplicate logging from nested deployment calls.
    """

    def __init__(self, tracer: EpisodicTracer | SqliteTracer):
        super().__init__()
        self.tracer = tracer
        # Track logged call IDs to prevent duplicates

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: ModelResponse | ModelResponseStream,
    ) -> Any:
        """Called once per HTTP request at proxy level, before response is sent to user.

        This hook is called only once per HTTP request
        It has access to the actual response object
        and runs synchronously before the HTTP response is returned.

        Uses litellm_call_id for deduplication to ensure we only log once per request.
        """
        import logging

        logger = logging.getLogger(__name__)

        # One-time debug print of tracer identity
        # Check both data["metadata"] (injected by middleware) and litellm_params["metadata"]
        # litellm_params = data.get("litellm_params", {}) if isinstance(data, dict) else {}
        # raw_meta_from_params = litellm_params.get("metadata", {}) if isinstance(litellm_params, dict) else {}
        raw_meta_from_data = data.get("metadata", {}) if isinstance(data, dict) else {}
        # Merge both sources (data metadata takes precedence as it comes from middleware)
        # allowed = {"session_id", "job", "user_api_key_request_route"}
        metadata = raw_meta_from_data.get("requester_metadata", {})

        model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
        messages = data.get("messages", []) if isinstance(data, dict) else []

        # Latency best-effort: prefer provider response_ms if available
        latency_ms: float = 0.0
        latency_ms = float(getattr(response, "response_ms", 0.0) or 0.0)

        usage = getattr(response, "usage", None)
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total": getattr(usage, "total_tokens", 0) if usage else 0,
        }

        if hasattr(response, "model_dump"):
            response_payload: Any = response.model_dump()
        elif isinstance(response, dict):
            response_payload = response
        else:
            response_payload = {"text": str(response)}

        # Extract the response ID from the LLM provider to use as trace_id/context_id
        # This ensures the context_id matches the actual completion ID from the provider
        response_id = response_payload.get("id", None)

        # Extract session_uids from metadata (sent from client via metadata routing)
        session_uids = metadata.get("session_uids")

        logger.info(f"[TracingCallback] Logging LLM call: trace_id={response_id}, model={model}, session_id={metadata.get('session_id')}, metadata_keys={list(metadata.keys())}")
        logger.info(f"[TracingCallback] session_uids from metadata: {session_uids}")
        logger.info(f"[TracingCallback] raw_meta_from_data={raw_meta_from_data}")
        logger.info(f"[TracingCallback] Tracer type: {type(self.tracer).__name__}, tracer_id={hex(id(self.tracer))}")

        self.tracer.log_llm_call(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=response_payload,
            metadata=metadata,
            session_id=metadata.get("session_id"),
            latency_ms=latency_ms,
            tokens=tokens,
            trace_id=response_id,  # Use the provider's response ID as the trace_id
            session_uids=session_uids,  # Pass session UIDs from client
        )

        logger.info(f"[TracingCallback] log_llm_call completed for trace_id={response_id}")

        # Return response unchanged
        return response

    # def log_failure_event(self, kwargs: dict[str, Any], response_obj: ModelResponse | ModelResponseStream, start_time: float, end_time: float) -> None:
    #     """Called after failed LLM completion (sync path).

    #     This complements the success hook to capture error outcomes.
    #     """
    #     litellm_params = kwargs.get("litellm_params", {})
    #     # Whitelist metadata keys to avoid noisy provider internals
    #     raw_meta = litellm_params.get("metadata", {})
    #     allowed = {"session_id", "job", "user_api_key_request_route"}
    #     metadata = {k: v for k, v in raw_meta.items() if k in allowed}

    #     model = kwargs.get("model", "unknown")
    #     messages = kwargs.get("messages", [])

    #     # Compute latency in milliseconds supporting datetime or float inputs
    #     _delta = end_time - start_time
    #     if isinstance(_delta, timedelta):
    #         latency_ms = _delta.total_seconds() * 1000.0
    #     else:
    #         latency_ms = float(_delta) * 1000.0

    #     error_info = {
    #         "error": str(response_obj) if response_obj else "Unknown error",
    #         "type": type(response_obj).__name__ if response_obj else "UnknownError",
    #     }

    #     self.tracer.log_llm_call(
    #         name=f"proxy/{model}",
    #         model=model,
    #         input={"messages": messages},
    #         output=error_info,
    #         metadata={**metadata, "error": True},
    #         session_id=metadata.get("session_id"),
    #         latency_ms=latency_ms,
    #         tokens={"prompt": 0, "completion": 0, "total": 0},
    #     )
