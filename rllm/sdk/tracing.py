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

from rllm.sdk.context import get_current_metadata, get_current_session

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

    async def store_batch(
        self,
        contexts: list[dict[str, Any]],
        preserve_order: bool = True,
    ) -> list[Any]: ...

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
        tracer = LLMTracer(store, project="my-app", max_concurrent_stores=100)

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

    # Sentinel used to unblock the worker on shutdown
    _STOP = object()

    def __init__(self, context_store: ContextStoreProtocol, project: str, default_tags: list[str] | None = None, max_queue_size: int = 10000, max_concurrent_stores: int = 100, max_batch_size: int = 8):
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

        self.namespace = project

        self.default_tags = default_tags or []

        # Queue for non-blocking trace storage (thread-safe queue)
        self._trace_queue = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._shutdown = False
        self._max_concurrent_stores = max_concurrent_stores
        self._max_batch_size = max(1, max_batch_size)

        # Thread and event loop for background worker
        self._worker_thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_started = threading.Event()
        # Track number of active store tasks (for simple idle detection)
        self._inflight_tasks: int = 0

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
        """Run the worker event loop forever in a separate thread."""
        # Create a new event loop for this thread
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)

        # Signal that the worker has started
        self._worker_started.set()

        # Schedule the async worker as a task using the loop's create_task method
        self._worker_loop.create_task(self._async_worker())

        # Run the loop forever until explicitly stopped
        try:
            self._worker_loop.run_forever()
        finally:
            # Close the loop only after it has been stopped
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
            self._worker_loop = None

    async def _async_worker(self):
        """Process trace items in batches sequentially (strict FIFO persistence with batching)."""
        batch_count = 0
        while True:
            # Collect a batch of items from the queue
            batch = []

            # Blocking get for the first item
            item = self._trace_queue.get()
            if item is self._STOP:
                break
            batch.append(item)

            # Try to get more items up to max_batch_size (non-blocking)
            while len(batch) < self._max_batch_size:
                try:
                    item = self._trace_queue.get_nowait()
                    if item is self._STOP:
                        # Put STOP back and process current batch first
                        self._trace_queue.put(self._STOP)
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            # Store the batch and wait for completion before processing next batch
            batch_count += 1
            logger.debug(f"[LLMTracer] Storing batch {batch_count} with {len(batch)} items (queue size: {self._trace_queue.qsize()})")

            if len(batch) == 1:
                await self._store_trace_with_retry(batch[0])
                self._trace_queue.task_done()
            else:
                await self._store_batch_with_retry(batch)
                for _ in batch:
                    self._trace_queue.task_done()

            logger.debug(f"[LLMTracer] Batch {batch_count} stored successfully")

        # Stop the loop when worker exits (schedule stop callback)
        if self._worker_loop and not self._worker_loop.is_closed():
            self._worker_loop.call_soon(self._worker_loop.stop)

    def _stop_worker_loop(self):
        """Stop the worker event loop gracefully."""
        if self._worker_loop is None:
            return

        try:

            def _stop():
                if self._worker_loop and not self._worker_loop.is_closed():
                    self._worker_loop.stop()

            self._worker_loop.call_soon_threadsafe(_stop)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            if self._worker_loop and not self._worker_loop.is_closed():
                self._worker_loop.close()
        except Exception as e:
            logger.exception("Error stopping worker loop: %s", e)
        finally:
            self._worker_loop = None

    async def _store_with_semaphore(self, semaphore: asyncio.Semaphore, trace_data: dict[str, Any]):
        async with semaphore:
            await self._store_trace_with_retry(trace_data)

    async def _store_batch_with_semaphore(self, semaphore: asyncio.Semaphore, batch: list[dict[str, Any]]):
        async with semaphore:
            await self._store_batch_with_retry(batch)

    async def _store_batch_and_mark_done(self, semaphore: asyncio.Semaphore, batch: list[dict[str, Any]]):
        try:
            if len(batch) == 1:
                await self._store_with_semaphore(semaphore, batch[0])
            else:
                await self._store_batch_with_semaphore(semaphore, batch)
        finally:
            for _ in batch:
                self._trace_queue.task_done()

    async def _store_batch_with_retry(self, batch: list[dict[str, Any]]):
        max_retries = 3
        retry_delays = [1, 2, 4]
        for attempt in range(max_retries):
            try:
                await self._store_batch(batch)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to store trace batch (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delays[attempt]}s...")
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    logger.exception("Failed to store trace batch after %d attempts: %s", max_retries, e)
                    raise

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
                    logger.exception("Dropping trace due to failed to store trace after %d attempts: %s", max_retries, e)

    async def _store_trace(self, trace_data: dict[str, Any]):
        """
        Store a trace to the context store.

        Args:
            trace_data: Dict containing all trace information prepared by log_llm_call
        """
        await self.context_store.store(context_id=trace_data["context_id"], data=trace_data["data"], text=trace_data["text"], namespace=trace_data["namespace"], context_type=trace_data["context_type"], tags=trace_data["tags"], metadata=trace_data["metadata"])

    async def _store_batch(self, batch: list[dict[str, Any]]):
        contexts_payload = [
            {
                "id": trace["context_id"],
                "data": trace["data"],
                "text": trace["text"],
                "namespace": trace["namespace"],
                "type": trace["context_type"],
                "tags": trace["tags"],
                "metadata": trace["metadata"],
            }
            for trace in batch
        ]
        await self.context_store.store_batch(contexts_payload, preserve_order=True)

    async def store_signal(self, context_id: str, context_type: str = "trace_batch_end", data: dict[str, Any] | None = None, tags: list[str] | None = None):
        """Queue a lightweight tracer signal so it is stored in-order with traces."""
        ordered_tags = list(dict.fromkeys([*(self.default_tags or []), "tracer_signal", *(tags or [])]))
        queue_item = {
            "context_id": context_id,
            "data": data or {},
            "text": None,
            "namespace": self.namespace,
            "context_type": context_type,
            "tags": ordered_tags,
            "metadata": {},
        }

        self._trace_queue.put_nowait(queue_item)

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
        """Stop the background worker without waiting for queue drain."""
        self._shutdown = True
        # Unblock the worker if it's waiting for new items
        try:
            self._trace_queue.put_nowait(self._STOP)
        except Exception:
            pass
        self._stop_worker_loop()

    def close_sync(self, timeout: float = 30.0) -> None:
        """Synchronous close without waiting."""
        self._shutdown = True
        try:
            self._trace_queue.put_nowait(self._STOP)
        except Exception:
            pass
        self._stop_worker_loop()

    async def flush(self, timeout: float = 30.0) -> None:
        """Block until all queued traces are persisted (best-effort).

        Uses the underlying queue.join() to wait for all items currently
        enqueued to be marked done by the worker. New items enqueued after
        this call begins are not guaranteed to be included.
        """
        try:
            await asyncio.wait_for(asyncio.to_thread(self._trace_queue.join), timeout=timeout)
        except Exception:
            # Best-effort: do not raise to callers; ordering guarantee is best-effort
            pass

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

        # if since:
        #     filter_kwargs["since"] = since

        filter_obj = ContextFilter(**filter_kwargs)

        return await self.context_store.query(filter_obj)


_TRACER: LLMTracer | None = None
_CONTEXT_STORE: ContextStoreProtocol | None = None


def get_context_store(endpoint: str | None, api_key: str | None) -> ContextStoreProtocol | None:
    from episodic import ContextStore

    global _CONTEXT_STORE
    if _CONTEXT_STORE is not None:
        return _CONTEXT_STORE

    if not endpoint:
        raise ValueError("Endpoint is required")

    _ContextStore = ContextStore(endpoint=endpoint, api_key=api_key)
    _CONTEXT_STORE = _ContextStore
    return _CONTEXT_STORE


def get_tracer(
    project: str | None,
    endpoint: str | None,
    api_key: str | None,
) -> LLMTracer | None:
    """Lazily create or return a global LLMTracer instance."""

    global _TRACER
    if _TRACER is not None:
        return _TRACER

    if not endpoint:
        raise ValueError("Endpoint is required")

    context_store: ContextStoreProtocol = get_context_store(endpoint, api_key)
    _TRACER = LLMTracer(
        context_store=context_store,
        project=project,
    )
    return _TRACER
