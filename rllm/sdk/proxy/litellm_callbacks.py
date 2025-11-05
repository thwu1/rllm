"""LiteLLM callbacks for parameter injection."""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger


class SamplingParametersCallback(CustomLogger):
    """Inject sampling parameters before LiteLLM sends requests.

    Adds logprobs and top_logprobs to all requests.
    Only adds return_token_ids for vLLM-compatible backends (not OpenAI/Anthropic).
    """

    def __init__(self, add_return_token_ids: bool = False):
        super().__init__()
        self.add_return_token_ids = add_return_token_ids

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or (args[2] if len(args) > 2 else {})
        model = data.get("model", "")

        result = {
            **data,
            "logprobs": True,
            "top_logprobs": data.get("top_logprobs", 1),
        }

        # Only add return_token_ids if explicitly enabled AND model supports it
        if self.add_return_token_ids and self._supports_token_ids(model):
            result["return_token_ids"] = True

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
