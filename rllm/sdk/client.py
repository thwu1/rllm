"""RLLM Client for orchestrating trace collection and training."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI, OpenAI

from .chat import OpenAIChatClient, SimpleTrackedAsyncChatClient, SimpleTrackedChatClient
from .session import SessionContext
from .tracing import get_tracer


class RLLMClient:
    """Client for managing tracing, chat access, and training workflows."""

    def __init__(
        self,
        api_key: str | None = None,
        project: str | None = None,
        cs_endpoint: str | None = None,
        cs_api_key: str | None = None,
        tracer: Any = None,
        chat_providers: dict[str, dict[str, Any]] | None = None,
        **config: Any,
    ) -> None:
        """Initialize RLLM client.

        Args:
            api_key: Primary API key for the default chat provider (OpenAI).
            project: Project/namespace used by the episodic tracer.
            cs_endpoint: Episodic context store endpoint for automatic tracer setup.
            cs_api_key: Episodic context store API key for automatic tracer setup.
            tracer: Optional pre-configured LLMTracer instance.
            chat_providers: Optional provider configuration overrides.
            **config: Additional chat/tracer configuration (base_url, organization, etc.).
        """
        self.config = config
        self.project = project
        self.chat_provider_configs = chat_providers or config.get("chat_providers", {})

        # Persist default OpenAI settings so callers can override per-request.
        self._openai_settings: dict[str, Any] = {}
        for key in ("api_key", "base_url", "organization", "timeout", "max_retries", "default_model"):
            if key in config and config[key] is not None:
                self._openai_settings[key] = config[key]
        if api_key is not None:
            self._openai_settings["api_key"] = api_key

        # Auto-configure tracer when episodic credentials are supplied.
        self._owns_tracer = False
        if tracer is None and cs_endpoint and cs_api_key:
            tracer = get_tracer(project=project, endpoint=cs_endpoint, api_key=cs_api_key)
            if tracer is not None:
                self._owns_tracer = True

        self.tracer = tracer

    # --------------------------------------------------------------------- Context
    def session(self, session_id: str | None = None, **metadata: Any) -> SessionContext:
        """Create a session context manager for automatic trace tracking."""
        return SessionContext(session_id, **metadata)

    # --------------------------------------------------------------------- Traces
    def _parse_time_filter(
        self,
        since: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> str | None:
        if since:
            return since
        if start_time or end_time:
            import warnings

            warnings.warn(
                "Absolute time ranges (start_time/end_time) not yet supported. Use 'since' instead.",
                stacklevel=2,
            )
        return None

    async def get_traces_async(
        self,
        session_id: str | None = None,
        since: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        **metadata_filters: Any,
    ) -> list[dict[str, Any]]:
        if not self.tracer:
            raise ValueError("No tracer configured. Initialize RLLMClient with tracer or episodic credentials.")

        time_filter = self._parse_time_filter(since, start_time, end_time)

        query_kwargs: dict[str, Any] = {"tags": None, "limit": limit}
        if time_filter:
            query_kwargs["since"] = time_filter

        contexts = await self.tracer.query_traces(**query_kwargs)

        traces: list[dict[str, Any]] = []
        for ctx in contexts:
            trace_data = ctx.data
            if session_id and trace_data.get("session_id") != session_id:
                continue

            trace_metadata = trace_data.get("metadata", {})
            if metadata_filters:
                match = all(trace_metadata.get(key) == value for key, value in metadata_filters.items())
                if not match:
                    continue

            traces.append(trace_data)

        return traces

    def get_traces(
        self,
        session_id: str | None = None,
        since: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        **metadata_filters: Any,
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            self.get_traces_async(
                session_id=session_id,
                since=since,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **metadata_filters,
            )
        )

    async def get_session_traces_async(self, session_id: str) -> list[dict[str, Any]]:
        return await self.get_traces_async(session_id=session_id, limit=10000)

    def get_session_traces(self, session_id: str) -> list[dict[str, Any]]:
        return self.get_traces(session_id=session_id, limit=10000)

    def get_recent_traces(self, hours: int = 1, limit: int = 100, **metadata_filters: Any) -> list[dict[str, Any]]:
        return self.get_traces(since=f"{hours}h", limit=limit, **metadata_filters)

    def export_traces(self, traces: list[dict[str, Any]], path: str, format: str = "jsonl") -> str:
        if format == "jsonl":
            with open(path, "w") as fh:
                for trace in traces:
                    fh.write(json.dumps(trace) + "\n")
        elif format == "json":
            with open(path, "w") as fh:
                json.dump(traces, fh, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'jsonl'.")
        return path

    # ---------------------------------------------------------------- Chat Clients
    def get_chat_client(
        self,
        provider: str = "openai",
        *,
        model: str | None = None,
        tokenizer: Any = None,
        adapter: str | None = None,
        **overrides: Any,
    ):
        """Return a chat client for the requested provider."""
        provider_key = provider.lower()
        config = dict(self.chat_provider_configs.get(provider_key, {}))
        config.update(overrides)

        if provider_key != "openai":
            raise ValueError(f"Unsupported chat provider '{provider}'.")

        # Support legacy adapter path when a tokenizer is provided explicitly.
        if adapter == "legacy" or tokenizer is not None:
            if tokenizer is None:
                raise ValueError("Tokenizer is required when using the legacy OpenAIChatClient adapter.")
            return self._build_legacy_openai_client(tokenizer=tokenizer, default_target=config.pop("default_target", "production"), config=config)

        return self._build_simple_openai_client(model=model, config=config)

    def _build_simple_openai_client(self, model: str | None, config: dict[str, Any]):
        merged = dict(self._openai_settings)
        merged.update(config)

        api_key = merged.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide api_key=... or configure it during RLLMClient initialization.")

        openai_kwargs: dict[str, Any] = {"api_key": api_key}
        for key in ("base_url", "organization", "timeout", "max_retries"):
            if merged.get(key) is not None:
                openai_kwargs[key] = merged[key]

        client = merged.get("client")
        if client is None:
            client = OpenAI(**openai_kwargs)

        resolved_model = model or merged.get("default_model")

        wrapper = SimpleTrackedChatClient(
            tracer=self.tracer,
            default_model=resolved_model,
            client=client,
        )
        self._attach_owner(wrapper)
        return wrapper

    def get_chat_client_async(
        self,
        provider: str = "openai",
        *,
        model: str | None = None,
        tokenizer: Any = None,
        adapter: str | None = None,
        **overrides: Any,
    ):
        """Return an async chat client for the requested provider."""
        provider_key = provider.lower()
        config = dict(self.chat_provider_configs.get(provider_key, {}))
        config.update(overrides)

        if provider_key != "openai":
            raise ValueError(f"Unsupported chat provider '{provider}'.")

        if adapter is not None or tokenizer is not None:
            raise ValueError("Async chat client does not support the legacy adapter or tokenizer overrides.")

        return self._build_simple_openai_client_async(model=model, config=config)

    def _build_legacy_openai_client(self, *, tokenizer: Any, default_target: str, config: dict[str, Any]):
        production_cfg = {
            "base_url": config.get("base_url", "http://localhost:8000/v1"),
            "api_key": config.get("api_key") or self._openai_settings.get("api_key"),
            "organization": config.get("organization") or self._openai_settings.get("organization"),
            "timeout": config.get("timeout", 60.0),
            "headers": config.get("headers", {}),
            "requester": config.get("requester"),
        }
        training_cfg = config.get("training")
        default_model = config.get("default_model") or self._openai_settings.get("default_model")
        default_sampling = config.get("default_sampling")
        default_metadata = config.get("default_metadata")

        wrapper = OpenAIChatClient(
            tokenizer=tokenizer,
            tracer=self.tracer,
            production_config=production_cfg,
            training_config=training_cfg,
            default_model=default_model,
            default_sampling=default_sampling,
            default_metadata=default_metadata,
            default_target=default_target,
        )
        self._attach_owner(wrapper)
        return wrapper

    def _build_simple_openai_client_async(self, model: str | None, config: dict[str, Any]):
        merged = dict(self._openai_settings)
        merged.update(config)

        api_key = merged.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide api_key=... or configure it during RLLMClient initialization.")

        openai_kwargs: dict[str, Any] = {"api_key": api_key}
        for key in ("base_url", "organization", "timeout", "max_retries"):
            if merged.get(key) is not None:
                openai_kwargs[key] = merged[key]

        client = merged.get("client")
        if client is None:
            client = AsyncOpenAI(**openai_kwargs)

        resolved_model = model or merged.get("default_model")

        wrapper = SimpleTrackedAsyncChatClient(
            tracer=self.tracer,
            default_model=resolved_model,
            client=client,
        )
        self._attach_owner(wrapper)
        return wrapper

    def _attach_owner(self, client_obj: Any) -> None:
        """Keep the SDK client alive while downstream wrappers are in use."""
        try:
            client_obj._sdk_owner = self
        except Exception:
            pass

    # --------------------------------------------------------------------- Cleanup
    def close(self, timeout: float = 30.0) -> None:
        """Flush the tracer if this client created it."""
        if self.tracer is None:
            return
        if hasattr(self.tracer, "close_sync"):
            try:
                self.tracer.close_sync(timeout=timeout)
            except Exception:  # pragma: no cover - tracer errors shouldn't crash shutdown
                pass

    def __del__(self):  # pragma: no cover - best effort cleanup
        if self._owns_tracer and self.tracer is not None and hasattr(self.tracer, "close_sync"):
            try:
                self.tracer.close_sync(timeout=5.0)
            except Exception:
                pass
