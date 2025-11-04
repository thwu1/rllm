"""Extremely small wrapper that logs chat completions via the episodic tracer."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from rllm.sdk.context import get_current_metadata, get_current_session


@dataclass
class _ChatCompletions:
    parent: SimpleTrackedChatClient

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self.parent._merge_args(args, kwargs)
        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self.parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        metadata = call_kwargs.pop("metadata", None) or {}

        prompt_text = self.parent._render_prompt(messages)
        prompt_token_ids = self.parent._encode_prompt(prompt_text)

        start = time.perf_counter()
        response = self.parent._client.chat.completions.create(**call_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        response_dict = response.model_dump()

        completion_token_ids = self.parent._extract_completion_tokens(response_dict)

        self.parent._log_trace(
            model=model,
            messages=messages,
            response_payload=response_dict,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            metadata_overrides=metadata,
            latency_ms=latency_ms,
        )

        return response


@dataclass
class _ChatNamespace:
    parent: SimpleTrackedChatClient

    @property
    def completions(self) -> _ChatCompletions:
        return _ChatCompletions(self.parent)


class SimpleTrackedChatClient:
    """Lean wrapper around `OpenAI` that records chat completions only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        tracer: Any = None,
        tokenizer: Any = None,
        default_model: str | None = None,
        client: OpenAI | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            init_kwargs = dict(client_kwargs)
            if api_key is not None:
                init_kwargs["api_key"] = api_key
            if base_url is not None:
                init_kwargs["base_url"] = base_url
            self._client = OpenAI(**init_kwargs)

        self.tracer = tracer
        self.tokenizer = tokenizer
        self.default_model = default_model

        self.chat = _ChatNamespace(self)

    # --- helpers -----------------------------------------------------------------

    @staticmethod
    def _merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                merged = dict(args[0])
                merged.update(kwargs)
                return merged
            raise TypeError("Positional arguments are not supported; pass keyword arguments.")
        return dict(kwargs)

    def _render_prompt(self, messages: Iterable[Mapping[str, Any]]) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "")
            parts.append(f"<{role}>{message.get('content', '')}")
        parts.append("<assistant>")
        return "".join(parts)

    def _encode_prompt(self, prompt: str) -> list[int]:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return []
        if hasattr(tokenizer, "encode"):
            try:
                encoded = tokenizer.encode(prompt, add_special_tokens=True)
            except TypeError:
                encoded = tokenizer.encode(prompt)
            if isinstance(encoded, list):
                return [int(tok) for tok in encoded]
            if isinstance(encoded, Mapping) and isinstance(encoded.get("input_ids"), list):
                return [int(tok) for tok in encoded["input_ids"]]
        if callable(tokenizer):
            encoded = tokenizer(prompt)
            if isinstance(encoded, Mapping) and isinstance(encoded.get("input_ids"), list):
                return [int(tok) for tok in encoded["input_ids"]]
            if isinstance(encoded, list):
                return [int(tok) for tok in encoded]
        return []

    @staticmethod
    def _extract_completion_tokens(response_payload: Mapping[str, Any]) -> list[int] | None:
        choices = response_payload.get("choices") or []
        if not choices:
            return None
        choice0 = choices[0]
        output_ids = choice0.get("output_token_ids")
        if isinstance(output_ids, list):
            return [int(tok) for tok in output_ids]
        logprobs = choice0.get("logprobs")
        if isinstance(logprobs, Mapping):
            token_ids = logprobs.get("token_ids")
            if isinstance(token_ids, list):
                return [int(tok) for tok in token_ids]
        return None

    def _log_trace(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_payload: Mapping[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int] | None,
        metadata_overrides: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        if not self.tracer:
            return

        session_id = get_current_session()
        context_metadata = dict(get_current_metadata() or {})

        metadata: dict[str, Any] = dict(context_metadata)
        metadata.update(dict(metadata_overrides or {}))
        metadata["token_ids"] = {"prompt": prompt_token_ids}
        if completion_token_ids:
            metadata["token_ids"]["completion"] = completion_token_ids

        usage = response_payload.get("usage") or {}
        tokens_summary = {
            "prompt": len(prompt_token_ids),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }

        try:
            self.tracer.log_llm_call(
                name="simple.chat.completions.create",
                model=model,
                input={"messages": messages},
                output=response_payload,
                session_id=session_id,
                metadata=metadata,
                latency_ms=latency_ms,
                tokens=tokens_summary,
            )
        except Exception:
            # Keep the client usable even if tracing fails.
            pass
