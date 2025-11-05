"""Minimal OpenAI client wrapper that records chat/completion calls via the tracer."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any

from openai import OpenAI
from openai.resources.chat import Chat as OpenAIChat
from openai.resources.chat.completions import Completions as OpenAIChatCompletions
from openai.resources.completions import Completions as OpenAICompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from rllm.sdk.context import get_current_metadata, get_current_session

logger = logging.getLogger(__name__)

_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "logprobs",
    "top_logprobs",
    "n",
    "stop",
)


class OpenAIChatClient(OpenAI):
    """Subclasses the official OpenAI client to capture call metadata for tracing."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        tracer: Any = None,
        production_config: dict[str, Any] | None = None,
        default_model: str | None = None,
        default_sampling: Mapping[str, Any] | None = None,
        default_metadata: Mapping[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        cfg = dict(production_config or {})
        init_kwargs = {k: cfg[k] for k in ("api_key", "organization", "project", "base_url", "timeout", "max_retries") if cfg.get(k) is not None}
        if cfg.get("headers"):
            init_kwargs["default_headers"] = cfg["headers"]
        init_kwargs.update(client_kwargs)

        super().__init__(**init_kwargs)

        self._tokenizer = tokenizer
        self.tracer = tracer
        self.default_model = default_model
        self.default_sampling = dict(default_sampling or {})
        self.default_metadata = dict(default_metadata or {})
        self._requester = cfg.get("requester")

    @cached_property
    def chat(self) -> InstrumentedChat:
        return InstrumentedChat(self)

    @cached_property
    def completions(self) -> InstrumentedCompletions:
        return InstrumentedCompletions(self)

    # --- helpers -----------------------------------------------------------------

    @staticmethod
    def _merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        if args:
            if len(args) == 1 and isinstance(args[0], Mapping):
                merged = dict(args[0])
                merged.update(kwargs)
                return merged
            raise TypeError("Positional arguments are not supported. Use keyword arguments only.")
        return dict(kwargs)

    def _render_prompt(self, messages: Iterable[Mapping[str, Any]]) -> str:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                return tokenizer.apply_chat_template(messages, tokenize=False)
        chunks: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            chunks.append(f"<{role}>{msg.get('content', '')}")
        chunks.append("<assistant>")
        return "".join(chunks)

    def _encode_text(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "encode"):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            except TypeError:
                encoded = tokenizer.encode(text)
            if isinstance(encoded, list):
                return [int(t) for t in encoded]
            if isinstance(encoded, Mapping) and isinstance(encoded.get("input_ids"), list):
                return [int(t) for t in encoded["input_ids"]]
        if callable(tokenizer):
            encoded = tokenizer(text)
            if isinstance(encoded, Mapping) and isinstance(encoded.get("input_ids"), list):
                return [int(t) for t in encoded["input_ids"]]
            if isinstance(encoded, list):
                return [int(t) for t in encoded]
        return []

    def _collect_sampling(self, params: Mapping[str, Any]) -> dict[str, Any]:
        sampling = dict(self.default_sampling)
        for key in _SAMPLING_FIELDS:
            if key in params and params[key] is not None:
                sampling[key] = params[key]
        return sampling

    @staticmethod
    def _extract_tokens(choice: Mapping[str, Any]) -> tuple[list[int] | None, Any | None]:
        if isinstance(choice.get("output_token_ids"), list):
            return [int(t) for t in choice["output_token_ids"]], choice.get("logprobs")

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, Mapping) and isinstance(logprobs.get("token_ids"), list):
            return [int(t) for t in logprobs["token_ids"]], logprobs

        return None, logprobs

    def _record_call(
        self,
        *,
        name: str,
        model: str,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any],
        sampling: Mapping[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int] | None,
        logprobs_payload: Any | None,
        metadata_overrides: Mapping[str, Any],
        latency_ms: float,
    ) -> None:
        if not self.tracer:
            return

        session_id = get_current_session()
        context_metadata = dict(get_current_metadata() or {})

        metadata: dict[str, Any] = dict(self.default_metadata)
        metadata.update(context_metadata)
        metadata.update(dict(metadata_overrides or {}))
        metadata["sampling"] = dict(sampling)
        metadata["token_ids"] = {"prompt": prompt_token_ids}
        if completion_token_ids:
            metadata["token_ids"]["completion"] = completion_token_ids
        if logprobs_payload is not None:
            metadata["logprobs"] = logprobs_payload

        usage = response_payload.get("usage") or {}
        tokens_summary = {
            "prompt": len(prompt_token_ids),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        }

        try:
            self.tracer.log_llm_call(
                name=name,
                model=model,
                input=request_payload,
                output=response_payload,
                session_id=session_id,
                metadata=metadata,
                latency_ms=latency_ms,
                tokens=tokens_summary,
            )
        except Exception:
            logger.exception("Failed to record LLM call via tracer")


class InstrumentedChat(OpenAIChat):
    def __init__(self, parent: OpenAIChatClient) -> None:
        super().__init__(parent)
        self._parent = parent

    @cached_property
    def completions(self) -> InstrumentedChatCompletions:
        return InstrumentedChatCompletions(self._parent)


class InstrumentedChatCompletions(OpenAIChatCompletions):
    def __init__(self, parent: OpenAIChatClient) -> None:
        super().__init__(parent)
        self._parent = parent

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        call_kwargs = self._parent._merge_args(args, kwargs)

        metadata_overrides = call_kwargs.pop("metadata", None) or {}
        if call_kwargs.get("stream"):
            raise NotImplementedError("stream=True is not supported by this client.")

        messages = call_kwargs.get("messages")
        if not messages:
            raise ValueError("messages must be provided for chat.completions.create.")

        model = call_kwargs.setdefault("model", self._parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        for key, value in self._parent.default_sampling.items():
            call_kwargs.setdefault(key, value)
        sampling = self._parent._collect_sampling(call_kwargs)

        prompt_text = self._parent._render_prompt(messages)
        prompt_token_ids = self._parent._encode_text(prompt_text)

        requester_payload = dict(call_kwargs)
        requester_payload["prompt_token_ids"] = [prompt_token_ids]

        start = time.perf_counter()
        if self._parent._requester:
            response_dict = self._parent._requester(requester_payload, {"endpoint": "chat"})
            response = ChatCompletion.model_validate(response_dict)
        else:
            response = super().create(**call_kwargs)
            response_dict = response.model_dump()
        latency_ms = (time.perf_counter() - start) * 1000

        completion_token_ids = None
        logprobs_payload = None
        choices = response_dict.get("choices") or []
        if choices:
            completion_token_ids, logprobs_payload = self._parent._extract_tokens(choices[0])

        self._parent._record_call(
            name="openai.chat.completions.create",
            model=model,
            request_payload={"messages": messages},
            response_payload=response_dict,
            sampling=sampling,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs_payload=logprobs_payload,
            metadata_overrides=metadata_overrides,
            latency_ms=latency_ms,
        )

        return response


class InstrumentedCompletions(OpenAICompletions):
    def __init__(self, parent: OpenAIChatClient) -> None:
        super().__init__(parent)
        self._parent = parent

    def create(self, *args: Any, **kwargs: Any) -> Completion:
        call_kwargs = self._parent._merge_args(args, kwargs)

        metadata_overrides = call_kwargs.pop("metadata", None) or {}
        if call_kwargs.get("stream"):
            raise NotImplementedError("stream=True is not supported by this client.")

        if "prompt" not in call_kwargs:
            raise ValueError("prompt must be provided for completions.create.")

        prompt = call_kwargs["prompt"]
        model = call_kwargs.setdefault("model", self._parent.default_model)
        if not model:
            raise ValueError("model must be supplied either in the call or via default_model.")

        for key, value in self._parent.default_sampling.items():
            call_kwargs.setdefault(key, value)
        sampling = self._parent._collect_sampling(call_kwargs)

        prompt_token_ids = self._parent._encode_text(str(prompt))

        requester_payload = dict(call_kwargs)
        requester_payload["prompt_token_ids"] = [prompt_token_ids]

        start = time.perf_counter()
        if self._parent._requester:
            response_dict = self._parent._requester(requester_payload, {"endpoint": "completions"})
            response = Completion.model_validate(response_dict)
        else:
            response = super().create(**call_kwargs)
            response_dict = response.model_dump()
        latency_ms = (time.perf_counter() - start) * 1000

        completion_token_ids = None
        logprobs_payload = None
        choices = response_dict.get("choices") or []
        if choices:
            completion_token_ids, logprobs_payload = self._parent._extract_tokens(choices[0])

        self._parent._record_call(
            name="openai.completions.create",
            model=model,
            request_payload={"prompt": prompt},
            response_payload=response_dict,
            sampling=sampling,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            logprobs_payload=logprobs_payload,
            metadata_overrides=metadata_overrides,
            latency_ms=latency_ms,
        )

        return response
