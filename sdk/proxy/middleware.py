"""
FastAPI middleware for RLLM proxy tracing.

Implements request interception, context extraction, and telemetry augmentation.
"""

from typing import Callable, Optional, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

from .context import SessionContext, set_session_context, get_session_context
from .tracer import get_tracer


logger = logging.getLogger(__name__)


class SessionContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and propagate session context from headers.

    This middleware:
    1. Extracts X-RLLM-Session and X-RLLM-Metadata headers
    2. Sets the session context for the request
    3. Clears context after request completes
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request and extract session context.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response from downstream handlers
        """
        # Extract session context from headers
        context = SessionContext.from_headers(dict(request.headers))

        # Set context for this request
        set_session_context(context)

        if context:
            logger.debug(
                f"Request with session_id={context.session_id}, "
                f"metadata={context.metadata}"
            )

        try:
            # Process request
            response = await call_next(request)
            return response
        finally:
            # Clear context after request
            set_session_context(None)


class TelemetryAugmentationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to augment LLM requests with telemetry flags.

    Automatically adds flags like:
    - logprobs: true (for token-level probabilities)
    - echo: true (for returning prompt tokens)
    - Additional sampling parameters for tracing
    """

    def __init__(
        self,
        app,
        enable_logprobs: bool = True,
        enable_prompt_tokens: bool = True,
        logprobs_count: int = 5,
    ):
        """
        Initialize telemetry augmentation middleware.

        Args:
            app: FastAPI application
            enable_logprobs: Whether to request logprobs by default
            enable_prompt_tokens: Whether to request prompt token IDs
            logprobs_count: Number of top logprobs to return
        """
        super().__init__(app)
        self.enable_logprobs = enable_logprobs
        self.enable_prompt_tokens = enable_prompt_tokens
        self.logprobs_count = logprobs_count

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Augment request with telemetry parameters.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response from downstream
        """
        # Only augment chat/completion endpoints
        if not (
            "/chat/completions" in request.url.path
            or "/completions" in request.url.path
        ):
            return await call_next(request)

        # Augment request body with telemetry flags
        # Note: This is handled in the proxy handler directly
        # since middleware can't easily modify request body
        return await call_next(request)


class LatencyTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request latency.

    Measures end-to-end latency and adds timing headers to response.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Track request latency.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response with latency headers
        """
        start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Add latency header
        response.headers["X-RLLM-Latency-Ms"] = f"{latency_ms:.2f}"

        # Log latency for traced requests
        context = get_session_context()
        if context:
            logger.info(
                f"Request {context.request_id} completed in {latency_ms:.2f}ms"
            )

        return response


# Dependency injection for FastAPI routes
async def get_session_context_dependency() -> Optional[SessionContext]:
    """
    FastAPI dependency to inject session context.

    Usage:
        @app.post("/chat/completions")
        async def chat_completions(
            context: Optional[SessionContext] = Depends(get_session_context_dependency)
        ):
            ...
    """
    return get_session_context()


def augment_request_params(
    params: dict,
    enable_logprobs: bool = True,
    enable_prompt_tokens: bool = True,
    logprobs_count: int = 5,
) -> dict:
    """
    Augment LLM request parameters with telemetry flags.

    Args:
        params: Original request parameters
        enable_logprobs: Add logprobs flag
        enable_prompt_tokens: Request prompt token IDs
        logprobs_count: Number of top logprobs

    Returns:
        Augmented parameters
    """
    augmented = params.copy()

    # Add logprobs for token-level probabilities
    if enable_logprobs and "logprobs" not in augmented:
        augmented["logprobs"] = True
        if "top_logprobs" not in augmented:
            augmented["top_logprobs"] = logprobs_count

    # Request prompt tokens (OpenAI-specific)
    if enable_prompt_tokens and "echo" not in augmented:
        # Note: echo is only for /completions, not /chat/completions
        # For chat, we need to use logprobs with prompt_logprobs
        if "stream_options" not in augmented:
            augmented["stream_options"] = {}
        augmented["stream_options"]["include_usage"] = True

    return augmented


def extract_telemetry_from_response(response: dict) -> dict:
    """
    Extract telemetry data from LLM response.

    Normalizes provider-specific response formats into a common structure.

    Args:
        response: Raw LLM API response

    Returns:
        Normalized telemetry data
    """
    telemetry = {
        "prompt_tokens": [],
        "completion_tokens": [],
        "logprobs": [],
        "prompt_length": 0,
        "completion_length": 0,
        "finish_reason": None,
    }

    # Extract usage information
    if "usage" in response:
        usage = response["usage"]
        telemetry["prompt_length"] = usage.get("prompt_tokens", 0)
        telemetry["completion_length"] = usage.get("completion_tokens", 0)

    # Extract choices
    if "choices" in response and len(response["choices"]) > 0:
        choice = response["choices"][0]
        telemetry["finish_reason"] = choice.get("finish_reason")

        # Extract logprobs if available
        if "logprobs" in choice and choice["logprobs"]:
            logprobs_data = choice["logprobs"]

            # OpenAI format: content is a list of token logprobs
            if "content" in logprobs_data:
                telemetry["logprobs"] = logprobs_data["content"]

                # Extract tokens from logprobs
                for token_data in logprobs_data["content"]:
                    if "token" in token_data:
                        # For completion tokens
                        telemetry["completion_tokens"].append(
                            token_data.get("bytes", token_data["token"])
                        )

    return telemetry


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling and logging.

    Catches exceptions and logs them with session context.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Handle errors and log with context.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response or error response
        """
        try:
            return await call_next(request)
        except Exception as e:
            context = get_session_context()
            logger.error(
                f"Error processing request"
                f"{f' for session {context.session_id}' if context else ''}: {e}",
                exc_info=True,
            )

            # Log error to tracer if in session
            if context:
                tracer = get_tracer()
                trace = tracer.start_trace(
                    context=context,
                    model="unknown",
                    provider="unknown",
                )
                tracer.complete_trace(trace, error=str(e))

            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type(e).__name__},
            )
