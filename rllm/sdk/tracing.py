"""LLM Tracing utilities for the rLLM SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .context import get_current_metadata, get_current_session

if TYPE_CHECKING:
    from episodic.core import Context
else:
    Context = Any


@runtime_checkable
class ContextStoreProtocol(Protocol):
    """Protocol describing the context store surface LLMTracer consumes."""

    async def store(
        self,
        context_id: str,
        data: dict[str, Any],
        text: str,
        namespace: str,
        context_type: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> None: ...

    async def get(self, context_id: str) -> Context | None: ...

    async def query(self, context_filter: Any) -> list[Context]: ...


logger = logging.getLogger(__name__)


class LLMTracer:
    """
    Non-blocking tracer for logging LLM calls to the Episodic Context Store.

    This tracer provides manual instrumentation for logging LLM interactions,
    including inputs, outputs, token usage, latency, and the context elements
    that were used to construct the prompt. Traces are queued in memory and
    stored asynchronously by a background worker with concurrent processing
    (up to 100 concurrent store operations by default), making the logging
    operation completely non-blocking.

    Example:
        ```python
        from episodic import ContextStore
        from rllm.sdk import LLMTracer

        store = ContextStore(endpoint="http://localhost:8000", api_key="your-key")
        tracer = LLMTracer(store, max_concurrent_stores=100)

        # Log call is now synchronous and non-blocking
        tracer.log_llm_call(
            name="scheduler_step",
            input="What tasks should I do today?",
            output="Based on your calendar...",
            model="gpt-4",
            latency_ms=1234.5,
            tokens={"prompt": 100, "completion": 200, "total": 300},
            contexts=["user_calendar", "task_history"]
        )

        # Use as async context manager for automatic cleanup
        async with LLMTracer(store) as tracer:
            tracer.log_llm_call(...)
        # All pending traces are flushed on exit
        ```
    """

    def __init__(self, context_store: ContextStoreProtocol, project: str | None = None, namespace: str | None = None, default_tags: list[str] | None = None, max_queue_size: int = 10000, max_concurrent_stores: int = 100):
        """
        Initialize the LLM tracer.

        Args:
            context_store: The context store backend to use for storing traces
            project: Project name to organize traces (recommended). Maps to namespace automatically.
                    Example: project="my-app" → namespace="my-app"
            namespace: Direct namespace specification (legacy, for backward compatibility).
                      If both project and namespace are provided, namespace takes precedence.
            default_tags: Default tags to apply to all traces
            max_queue_size: Maximum number of traces to buffer in memory (default: 10,000)
            max_concurrent_stores: Maximum number of concurrent store operations (default: 100)
        """
        self.context_store = context_store

        # Project-based approach: project name maps to namespace
        # Backward compatibility: if namespace is provided, use it
        if namespace is not None:
            self.namespace = namespace
        elif project is not None:
            self.namespace = project
        else:
            self.namespace = "default"

        self.default_tags = default_tags or []

        # Queue for non-blocking trace storage (thread-safe queue)
        self._trace_queue = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._shutdown = False
        self._max_concurrent_stores = max_concurrent_stores

        # Thread and event loop for background worker
        self._worker_thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_started = threading.Event()

        # Start background worker
        self._start_background_worker()

    def _start_background_worker(self):
        """Start the background worker thread with its own event loop."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(target=self._run_worker_loop, daemon=True, name="LLMTracer-Worker")
        self._worker_thread.start()

        # Wait for the worker to start
        self._worker_started.wait(timeout=5.0)

    def _run_worker_loop(self):
        """Run the worker event loop in a separate thread."""
        # Create a new event loop for this thread
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)

        # Signal that the worker has started
        self._worker_started.set()

        try:
            # Run the async worker
            self._worker_loop.run_until_complete(self._async_worker())
        finally:
            self._worker_loop.close()
            self._worker_loop = None

    async def _async_worker(self):
        """
        Async worker that processes traces from the queue with parallel execution.
        Single worker spawns tasks directly with controlled concurrency.
        """
        # Create semaphore for limiting concurrent operations
        semaphore = asyncio.Semaphore(self._max_concurrent_stores)
        active_tasks = set()

        try:
            while not self._shutdown or not self._trace_queue.empty() or active_tasks:
                # Get item from queue (non-blocking)
                try:
                    trace_data = self._trace_queue.get_nowait()
                    # Spawn task immediately
                    task = asyncio.create_task(self._store_with_semaphore(semaphore, trace_data))
                    active_tasks.add(task)
                except queue.Empty:
                    # No items, wait briefly for new items or tasks to complete
                    if active_tasks:
                        done, pending = await asyncio.wait(active_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                        # Update active set with only pending tasks
                        active_tasks = pending
                    else:
                        await asyncio.sleep(0.1)

            # Wait for remaining tasks
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        except Exception as e:
            logger.exception("Error in async worker: %s", e)
            # Wait for any remaining tasks
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _store_with_semaphore(self, semaphore: asyncio.Semaphore, trace_data: dict[str, Any]):
        """
        Store trace with semaphore-controlled concurrency.

        Args:
            semaphore: Semaphore to limit concurrent store operations
            trace_data: Trace data to store
        """
        async with semaphore:
            await self._store_trace_with_retry(trace_data)

    async def _store_trace_with_retry(self, trace_data: dict[str, Any]):
        """
        Store a trace with retry logic.

        Args:
            trace_data: Dict containing all trace information prepared by log_llm_call
        """
        max_retries = 3
        retry_delays = [1, 2, 4]  # seconds

        for attempt in range(max_retries):
            try:
                await self._store_trace(trace_data)
                return  # Success, exit
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to store trace (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delays[attempt]}s...")
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    logger.exception("Failed to store trace after %d attempts: %s", max_retries, e)
                    raise

    async def _store_trace(self, trace_data: dict[str, Any]):
        """
        Store a trace to the context store.

        Args:
            trace_data: Dict containing all trace information prepared by log_llm_call
        """
        await self.context_store.store(context_id=trace_data["context_id"], data=trace_data["data"], text=trace_data["text"], namespace=trace_data["namespace"], context_type=trace_data["context_type"], tags=trace_data["tags"], metadata=trace_data["metadata"])

    def log_llm_call(self, name: str, input: str | list | dict, output: str | dict, model: str, latency_ms: float, tokens: dict[str, int], contexts: list[str | dict[str, Any]] | None = None, metadata: dict[str, Any] | None = None, tags: list[str] | None = None, trace_id: str | None = None, parent_trace_id: str | None = None, cost: float | None = None, environment: str | None = None, tools: list[dict[str, Any]] | None = None, session_id: str | None = None) -> None:
        """
        Log an LLM call to the context store (non-blocking).

        This method is synchronous and returns immediately after queuing the trace.
        The trace is stored asynchronously by a background worker.

        Args:
            name: Name/identifier for this LLM call (e.g., "scheduler_step", "email_agent")
            input: The input/prompt sent to the LLM (can be string, messages array, or dict)
            output: The output/response from the LLM (can be string or structured dict)
            model: The model used (e.g., "gpt-4", "claude-3-opus")
            latency_ms: Latency in milliseconds
            tokens: Token usage dict with keys: prompt, completion, total
            contexts: List of context elements used to construct the prompt.
                     Can be context IDs (strings) or full context objects (dicts)
            metadata: Additional metadata to attach to the trace
            tags: Additional tags for this trace (merged with default_tags)
            trace_id: Optional custom trace ID (auto-generated if not provided)
            parent_trace_id: ID of parent trace for hierarchical tracing
            cost: Optional cost in USD for this LLM call
            environment: Optional environment identifier (e.g., "production", "staging")
            tools: Optional list of tools/functions available to the LLM during this call
            session_id: Optional session ID to group related traces (e.g., conversation session)

        Returns:
            None (trace is queued for asynchronous storage)
        """
        # Generate trace ID if not provided
        if trace_id is None:
            trace_id = f"tr_{uuid.uuid4().hex[:16]}"

        # Get session_id from context if not provided
        if session_id is None:
            session_id = get_current_session()

        # Merge context metadata with call-specific metadata
        context_meta = get_current_metadata()
        final_metadata = {**context_meta, **(metadata or {})}

        # Prepare trace data
        trace_data_content = {"name": name, "input": input, "output": output, "model": model, "latency_ms": latency_ms, "tokens": tokens, "contexts": contexts or [], "trace_id": trace_id, "timestamp": time.time()}

        # Add optional fields
        if parent_trace_id:
            trace_data_content["parent_trace_id"] = parent_trace_id

        if cost is not None:
            trace_data_content["cost"] = cost

        if environment:
            trace_data_content["environment"] = environment

        if tools is not None:
            trace_data_content["tools"] = tools

        if session_id:
            trace_data_content["session_id"] = session_id

        if final_metadata:
            trace_data_content["metadata"] = final_metadata

        # Generate searchable text from the trace
        text = self._generate_trace_text(name, input, output, model)

        # Merge tags
        all_tags = list(set(self.default_tags + (tags or [])))
        all_tags.append("llm_trace")

        # Prepare metadata for storage
        trace_metadata = {"model": model, "latency_ms": latency_ms, "total_tokens": tokens.get("total", 0)}

        # Add tools count to metadata if tools are provided
        if tools is not None:
            trace_metadata["tools_count"] = len(tools)

        # Add session_id to metadata if provided for easier querying
        if session_id:
            trace_metadata["session_id"] = session_id

        # Package everything for the background worker
        queue_item = {"context_id": trace_id, "data": trace_data_content, "text": text, "namespace": self.namespace, "context_type": "llm_trace", "tags": all_tags, "metadata": trace_metadata}

        # Ensure background worker is running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_background_worker()

        # Queue the trace (non-blocking)
        try:
            self._trace_queue.put_nowait(queue_item)
        except queue.Full:
            logger.warning(f"Trace queue is full (max size: {self._max_queue_size}). Dropping trace: {trace_id}")

    def _generate_trace_text(self, name: str, input: str | list | dict, output: str | dict, model: str) -> str:
        """
        Generate searchable text representation of the trace.

        Args:
            name: Trace name
            input: LLM input (string, messages array, or dict)
            output: LLM output (string or dict)
            model: Model name

        Returns:
            Searchable text string
        """

        # Convert input to text for searching
        if isinstance(input, str):
            input_text = input
        elif isinstance(input, list):
            # Messages array - extract content (handle None values from tool calls)
            input_text = " ".join([(msg.get("content") or "") if isinstance(msg, dict) else (str(msg) if msg else "") for msg in input])
        elif isinstance(input, dict):
            input_text = json.dumps(input)
        else:
            input_text = str(input) if input else ""

        # Convert output to text for searching
        if isinstance(output, str):
            output_text = output or ""
        elif isinstance(output, dict):
            # Extract assistant content if available
            assistant_content = output.get("assistant") or ""
            output_text = assistant_content if assistant_content else json.dumps(output)
        else:
            output_text = str(output) if output else ""

        # Truncate for text field if too long
        max_len = 500
        input_preview = input_text[:max_len] + "..." if len(input_text) > max_len else input_text
        output_preview = output_text[:max_len] + "..." if len(output_text) > max_len else output_text

        return f"[{name}] Model: {model}\nInput: {input_preview}\nOutput: {output_preview}"

    async def close(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the tracer and flush all pending traces.

        Args:
            timeout: Maximum time in seconds to wait for pending traces to be stored
        """
        logger.info("Shutting down LLMTracer, flushing pending traces...")
        self._shutdown = True

        # Wait for queue to empty
        start_time = time.time()
        while not self._trace_queue.empty() and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if self._trace_queue.empty():
            logger.info("All pending traces flushed successfully")
        else:
            remaining = self._trace_queue.qsize()
            logger.warning(f"Timeout reached during shutdown. {remaining} traces may not be stored.")

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not terminate gracefully")

    def close_sync(self, timeout: float = 30.0) -> None:
        """
        Synchronous version of close() for non-async contexts.

        Args:
            timeout: Maximum time in seconds to wait for pending traces to be stored
        """
        logger.info("Shutting down LLMTracer, flushing pending traces...")
        self._shutdown = True

        # Wait for queue to empty
        start_time = time.time()
        while not self._trace_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self._trace_queue.empty():
            logger.info("All pending traces flushed successfully")
        else:
            remaining = self._trace_queue.qsize()
            logger.warning(f"Timeout reached during shutdown. {remaining} traces may not be stored.")

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("Worker thread did not terminate gracefully")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    async def get_trace(self, trace_id: str) -> Context | None:
        """
        Retrieve a trace by its ID.

        Args:
            trace_id: The trace ID to retrieve

        Returns:
            The Context object representing the trace, or None if not found
        """
        try:
            return await self.context_store.get(trace_id)
        except Exception:
            return None

    async def query_traces(self, tags: list[str] | None = None, since: str | None = None, limit: int = 100) -> list[Context]:
        """
        Query traces with filters.

        Args:
            tags: Filter by tags
            since: Time filter (e.g., "1h", "30m", "1d")
            limit: Maximum number of traces to return

        Returns:
            List of Context objects representing traces
        """
        from episodic.core import ContextFilter  # Imported lazily to avoid hard dependency

        # Build filter - only include since if provided
        filter_kwargs = {"namespaces": [self.namespace], "tags": tags if tags else None, "context_types": ["llm_trace"], "limit": limit}

        if since:
            filter_kwargs["since"] = since

        filter_obj = ContextFilter(**filter_kwargs)

        return await self.context_store.query(filter_obj)


_TRACER: LLMTracer | None = None


def get_tracer(
    project: str | None,
    endpoint: str | None,
    api_key: str | None,
) -> LLMTracer | None:
    """Lazily create or return a global LLMTracer instance."""
    from episodic import ContextStore

    global _TRACER
    if _TRACER is not None:
        return _TRACER

    if not endpoint:
        print("⚠️ Episodic: EPISODIC_ENDPOINT not configured")
        return None

    try:
        print(f"🔄 Episodic: Initializing tracer with endpoint {endpoint}")
        context_store: ContextStoreProtocol = ContextStore(
            endpoint=endpoint,
            api_key=api_key,
        )
        _TRACER = LLMTracer(
            context_store=context_store,
            project=project,
        )
        print(f"✅ Episodic: Tracer initialized successfully for project '{project or 'default'}'")
        return _TRACER
    except Exception as exc:  # pragma: no cover - diagnostic print
        import traceback

        print("❌ Episodic: Failed to initialize tracer:")
        print(f"   Error: {exc}")
        print(f"   Traceback: {traceback.format_exc()}")
        return None
