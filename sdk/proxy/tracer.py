"""
LLM call tracing for the RLLM proxy.

Captures prompts, completions, token IDs, logprobs, timing, and metadata.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any, List
import json
import logging
from pathlib import Path

from .context import SessionContext


logger = logging.getLogger(__name__)


@dataclass
class LLMCallTrace:
    """
    Comprehensive trace of a single LLM API call.

    Attributes:
        request_id: Unique identifier for this request
        session_id: Episode/trajectory identifier
        timestamp: ISO timestamp when request started
        model: Model name/identifier
        messages: Input messages (chat format)
        prompt: Raw prompt string (for completion API)
        response_text: Generated text response
        prompt_tokens: Token IDs for the prompt
        completion_tokens: Token IDs for the completion
        logprobs: Token-level log probabilities
        prompt_length: Number of tokens in prompt
        completion_length: Number of tokens in completion
        latency_ms: Request latency in milliseconds
        finish_reason: Completion finish reason
        metadata: Additional context from session
        provider: Upstream LLM provider (openai, anthropic, etc.)
        error: Error message if request failed
    """
    request_id: str
    session_id: str
    timestamp: str
    model: str
    messages: Optional[List[dict]] = None
    prompt: Optional[str] = None
    response_text: Optional[str] = None
    prompt_tokens: Optional[List[int]] = None
    completion_tokens: Optional[List[int]] = None
    logprobs: Optional[List[dict]] = None
    prompt_length: Optional[int] = None
    completion_length: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None
    metadata: dict = None
    provider: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LLMTracer:
    """
    Tracer for logging LLM calls in episodic context.

    This class manages trace storage and integrates with RLLM's
    EpisodeLogger for persistent storage.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the LLM tracer.

        Args:
            log_dir: Directory for storing trace logs. If None, uses default.
        """
        self.log_dir = log_dir or Path("./logs/llm_traces")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_traces: dict[str, List[LLMCallTrace]] = {}

    def start_trace(
        self,
        context: SessionContext,
        model: str,
        messages: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> LLMCallTrace:
        """
        Start tracing an LLM call.

        Args:
            context: Session context for this request
            model: Model identifier
            messages: Chat messages (for chat completion)
            prompt: Raw prompt (for text completion)
            provider: Upstream provider name

        Returns:
            Started trace object
        """
        trace = LLMCallTrace(
            request_id=context.request_id,
            session_id=context.session_id,
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            messages=messages,
            prompt=prompt,
            metadata=context.metadata.copy(),
            provider=provider,
        )
        return trace

    def complete_trace(
        self,
        trace: LLMCallTrace,
        response_text: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        completion_tokens: Optional[List[int]] = None,
        logprobs: Optional[List[dict]] = None,
        prompt_length: Optional[int] = None,
        completion_length: Optional[int] = None,
        latency_ms: Optional[float] = None,
        finish_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Complete a trace with response data.

        Args:
            trace: The trace object to complete
            response_text: Generated response
            prompt_tokens: Prompt token IDs
            completion_tokens: Completion token IDs
            logprobs: Token log probabilities
            prompt_length: Number of prompt tokens
            completion_length: Number of completion tokens
            latency_ms: Request latency
            finish_reason: Reason for completion
            error: Error message if failed
        """
        trace.response_text = response_text
        trace.prompt_tokens = prompt_tokens
        trace.completion_tokens = completion_tokens
        trace.logprobs = logprobs
        trace.prompt_length = prompt_length
        trace.completion_length = completion_length
        trace.latency_ms = latency_ms
        trace.finish_reason = finish_reason
        trace.error = error

        # Store in memory indexed by session_id
        if trace.session_id not in self._in_memory_traces:
            self._in_memory_traces[trace.session_id] = []
        self._in_memory_traces[trace.session_id].append(trace)

        # Also write to disk immediately for durability
        self._write_trace(trace)

    def _write_trace(self, trace: LLMCallTrace) -> None:
        """
        Write trace to disk.

        Args:
            trace: Trace to persist
        """
        session_dir = self.log_dir / trace.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        trace_file = session_dir / f"{trace.request_id}.json"
        try:
            with open(trace_file, "w") as f:
                json.dump(trace.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write trace {trace.request_id}: {e}")

    def get_traces(self, session_id: str) -> List[LLMCallTrace]:
        """
        Retrieve all traces for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of traces for the session
        """
        return self._in_memory_traces.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """
        Clear traces for a completed session.

        Args:
            session_id: Session identifier to clear
        """
        if session_id in self._in_memory_traces:
            del self._in_memory_traces[session_id]


# Global tracer instance
_tracer: Optional[LLMTracer] = None


def get_tracer() -> LLMTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer()
    return _tracer


def set_tracer(tracer: LLMTracer) -> None:
    """Set the global tracer instance."""
    global _tracer
    _tracer = tracer
