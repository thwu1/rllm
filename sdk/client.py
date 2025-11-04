"""RLLM Client for orchestrating trace collection and training."""

import asyncio
import json
from typing import Any

from .chat import OpenAIChatClient
from .session import SessionContext


class RLLMClient:
    """Client for managing LLM tracing and RL training workflows.

    Example:
        >>> client = RLLMClient()
        >>> with client.session("my-session", experiment="v1"):
        ...     # All traces automatically get session_id and metadata
        ...     tracer.log_llm_call(...)
    """

    def __init__(self, tracer=None, chat_providers: dict[str, dict[str, Any]] | None = None, **config):
        """Initialize RLLM client.

        Args:
            tracer: Optional LLMTracer instance for trace retrieval
            chat_providers: Optional chat provider configuration mapping
            **config: Configuration options (reserved for future use)
        """
        self.config = config
        self.tracer = tracer
        self.chat_provider_configs = chat_providers or config.get("chat_providers", {})

    def session(self, session_id: str | None = None, **metadata) -> SessionContext:
        """Create a session context manager for automatic trace tracking.

        Args:
            session_id: Session ID (auto-generated if None)
            **metadata: Arbitrary metadata to attach to all traces in this session

        Returns:
            SessionContext instance

        Examples:
            Simple session:
            >>> with client.session("task-123"):
            ...     tracer.log_llm_call(...)

            Session with custom metadata:
            >>> with client.session("task-123", experiment="v1", user="alice"):
            ...     tracer.log_llm_call(...)

            Auto-generated session ID:
            >>> with client.session(experiment="v1"):
            ...     tracer.log_llm_call(...)
        """
        return SessionContext(session_id, **metadata)

    def _parse_time_filter(self, since: str | None = None, start_time: float | None = None, end_time: float | None = None) -> str | None:
        """Parse time filters into episodic 'since' format.

        Args:
            since: Relative time like "1h", "30m", "1d"
            start_time: Absolute start timestamp (not yet supported by episodic)
            end_time: Absolute end timestamp (not yet supported by episodic)

        Returns:
            Time filter string or None
        """
        if since:
            return since
        elif start_time or end_time:
            # For now, episodic only supports 'since' relative time
            # TODO: Add absolute time range support when available
            import warnings

            warnings.warn(
                "Absolute time ranges (start_time/end_time) not yet supported. Use 'since' instead.",
                stacklevel=2,
            )
            return None
        return None

    async def get_traces_async(self, session_id: str | None = None, since: str | None = None, start_time: float | None = None, end_time: float | None = None, limit: int = 100, **metadata_filters) -> list[dict[str, Any]]:
        """Get traces asynchronously with flexible filtering.

        Args:
            session_id: Filter by specific session
            since: Relative time ("1h", "30m", "1d", "7d")
            start_time: Absolute start (Unix timestamp) - not yet supported
            end_time: Absolute end (Unix timestamp) - not yet supported
            limit: Max traces to return
            **metadata_filters: Filter by metadata (task_id=5, experiment="v1", etc.)

        Returns:
            List of trace dictionaries

        Examples:
            # Last hour
            traces = await client.get_traces_async(since="1h")

            # Specific session
            traces = await client.get_traces_async(session_id="task_0_abc")

            # Time range + metadata
            traces = await client.get_traces_async(
                since="24h",
                task_id=5,
                experiment="v1"
            )
        """
        if not self.tracer:
            raise ValueError("No tracer configured. Initialize RLLMClient with tracer argument.")

        # Parse time filter
        time_filter = self._parse_time_filter(since, start_time, end_time)

        # Query traces from episodic (only pass since if it's not None)
        query_kwargs = {"tags": None, "limit": limit}
        if time_filter:
            query_kwargs["since"] = time_filter

        contexts = await self.tracer.query_traces(**query_kwargs)

        # Convert Context objects to dicts and filter by metadata
        traces = []
        for ctx in contexts:
            trace_data = ctx.data

            # Filter by session_id if specified
            if session_id and trace_data.get("session_id") != session_id:
                continue

            # Filter by metadata if specified
            trace_metadata = trace_data.get("metadata", {})
            if metadata_filters:
                match = all(trace_metadata.get(key) == value for key, value in metadata_filters.items())
                if not match:
                    continue

            traces.append(trace_data)

        return traces

    def get_traces(self, session_id: str | None = None, since: str | None = None, start_time: float | None = None, end_time: float | None = None, limit: int = 100, **metadata_filters) -> list[dict[str, Any]]:
        """Get traces synchronously with flexible filtering.

        Wrapper around get_traces_async() for synchronous usage.

        Args:
            session_id: Filter by specific session
            since: Relative time ("1h", "30m", "1d", "7d")
            start_time: Absolute start (Unix timestamp) - not yet supported
            end_time: Absolute end (Unix timestamp) - not yet supported
            limit: Max traces to return
            **metadata_filters: Filter by metadata (task_id=5, experiment="v1", etc.)

        Returns:
            List of trace dictionaries

        Examples:
            # Last hour
            traces = client.get_traces(since="1h")

            # Specific session
            traces = client.get_traces(session_id="task_0_abc")

            # Multiple filters
            traces = client.get_traces(
                since="24h",
                task_id=5,
                experiment="v1"
            )
        """
        return asyncio.run(self.get_traces_async(session_id=session_id, since=since, start_time=start_time, end_time=end_time, limit=limit, **metadata_filters))

    async def get_session_traces_async(self, session_id: str) -> list[dict[str, Any]]:
        """Convenience: Get all traces for a specific session (async).

        Args:
            session_id: Session ID to retrieve

        Returns:
            List of trace dictionaries

        Example:
            traces = await client.get_session_traces_async("task_0_abc12345")
        """
        return await self.get_traces_async(session_id=session_id, limit=10000)

    def get_session_traces(self, session_id: str) -> list[dict[str, Any]]:
        """Convenience: Get all traces for a specific session.

        Args:
            session_id: Session ID to retrieve

        Returns:
            List of trace dictionaries

        Example:
            traces = client.get_session_traces("task_0_abc12345")
        """
        return self.get_traces(session_id=session_id, limit=10000)

    def get_recent_traces(self, hours: int = 1, limit: int = 100, **metadata_filters) -> list[dict[str, Any]]:
        """Convenience: Get traces from the last N hours.

        Args:
            hours: Number of hours to look back
            limit: Max traces to return
            **metadata_filters: Additional metadata filters

        Returns:
            List of trace dictionaries

        Example:
            # Last 24 hours with experiment filter
            traces = client.get_recent_traces(hours=24, experiment="v1")
        """
        return self.get_traces(since=f"{hours}h", limit=limit, **metadata_filters)

    def export_traces(self, traces: list[dict[str, Any]], path: str, format: str = "jsonl") -> str:
        """Export traces to file.

        Args:
            traces: List of trace dictionaries
            path: Output file path
            format: "json" or "jsonl"

        Returns:
            Path to exported file

        Example:
            traces = client.get_traces(since="1h")
            client.export_traces(traces, "traces.jsonl")
        """
        if format == "jsonl":
            with open(path, "w") as f:
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
        elif format == "json":
            with open(path, "w") as f:
                json.dump(traces, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'jsonl'.")

        return path

    def get_chat_client(
        self,
        provider: str,
        *,
        tokenizer,
        default_target: str = "production",
        **overrides: Any,
    ):
        """Get a chat client for the specified provider.

        Args:
            provider: Provider name (currently only "openai" is supported).
            tokenizer: Tokenizer instance used to convert chat messages into token IDs.
            default_target: Default routing target ("production" or "training").
            **overrides: Provider-specific configuration overrides.

        Returns:
            Provider-specific chat client instance.
        """
        provider_key = provider.lower()
        config = dict(self.chat_provider_configs.get(provider_key, {}))
        config.update(overrides)

        if provider_key == "openai":
            production_cfg = {
                "base_url": config.get("base_url", "http://localhost:8000/v1"),
                "api_key": config.get("api_key"),
                "organization": config.get("organization"),
                "timeout": config.get("timeout", 60.0),
                "headers": config.get("headers", {}),
                "requester": config.get("requester"),
            }
            training_cfg = config.get("training")
            default_model = config.get("default_model")
            default_sampling = config.get("default_sampling")
            default_metadata = config.get("default_metadata")

            return OpenAIChatClient(
                tokenizer=tokenizer,
                tracer=self.tracer,
                production_config=production_cfg,
                training_config=training_cfg,
                default_model=default_model,
                default_sampling=default_sampling,
                default_metadata=default_metadata,
                default_target=default_target,
            )

        raise ValueError(f"Unsupported chat provider '{provider}'.")
