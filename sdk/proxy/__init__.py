"""
RLLM LiteLLM Proxy - OpenAI-compatible proxy with episodic tracing.
"""

from .server import create_app, main
from .context import SessionContext, get_session_context, set_session_context
from .tracer import LLMTracer, LLMCallTrace, get_tracer, set_tracer
from .middleware import (
    SessionContextMiddleware,
    TelemetryAugmentationMiddleware,
    LatencyTrackingMiddleware,
    ErrorHandlingMiddleware,
    augment_request_params,
    extract_telemetry_from_response,
)

__version__ = "0.1.0"

__all__ = [
    # Server
    "create_app",
    "main",
    # Context
    "SessionContext",
    "get_session_context",
    "set_session_context",
    # Tracer
    "LLMTracer",
    "LLMCallTrace",
    "get_tracer",
    "set_tracer",
    # Middleware
    "SessionContextMiddleware",
    "TelemetryAugmentationMiddleware",
    "LatencyTrackingMiddleware",
    "ErrorHandlingMiddleware",
    "augment_request_params",
    "extract_telemetry_from_response",
]
