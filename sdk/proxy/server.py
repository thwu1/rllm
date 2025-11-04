"""
RLLM LiteLLM Proxy Server.

FastAPI-based proxy server that wraps LiteLLM with tracing middleware.
"""

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, Any, Dict
import logging
import time
import litellm
from litellm import acompletion

from .context import SessionContext, get_session_context
from .tracer import get_tracer, LLMCallTrace
from .middleware import (
    SessionContextMiddleware,
    LatencyTrackingMiddleware,
    ErrorHandlingMiddleware,
    get_session_context_dependency,
    augment_request_params,
    extract_telemetry_from_response,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(
    enable_logprobs: bool = True,
    enable_prompt_tokens: bool = True,
    log_dir: Optional[str] = None,
) -> FastAPI:
    """
    Create the RLLM proxy FastAPI application.

    Args:
        enable_logprobs: Whether to enable logprobs by default
        enable_prompt_tokens: Whether to request prompt tokens
        log_dir: Directory for trace logs

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="RLLM LiteLLM Proxy",
        description="OpenAI-compatible proxy with episodic tracing",
        version="0.1.0",
    )

    # Add middleware (order matters - executed bottom to top)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LatencyTrackingMiddleware)
    app.add_middleware(SessionContextMiddleware)

    # Store config in app state
    app.state.enable_logprobs = enable_logprobs
    app.state.enable_prompt_tokens = enable_prompt_tokens

    # Configure LiteLLM
    litellm.set_verbose = False  # Disable verbose logging
    litellm.drop_params = False  # Don't drop unknown params

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "rllm-proxy"}

    @app.get("/v1/models")
    async def list_models():
        """
        List available models.

        This endpoint returns models configured in the proxy.
        """
        # TODO: Load from configuration
        return {
            "object": "list",
            "data": [
                {"id": "gpt-4", "object": "model", "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
                {"id": "claude-3-opus", "object": "model", "owned_by": "anthropic"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        context: Optional[SessionContext] = Depends(get_session_context_dependency),
    ):
        """
        OpenAI-compatible chat completions endpoint.

        Args:
            request: FastAPI request
            context: Session context from headers

        Returns:
            Chat completion response
        """
        # Parse request body
        body = await request.json()

        # Extract parameters
        messages = body.get("messages", [])
        model = body.get("model", "gpt-3.5-turbo")
        stream = body.get("stream", False)

        # Augment with telemetry flags
        augmented_params = augment_request_params(
            body,
            enable_logprobs=request.app.state.enable_logprobs,
            enable_prompt_tokens=request.app.state.enable_prompt_tokens,
        )

        # Start tracing if in session
        tracer = get_tracer()
        trace: Optional[LLMCallTrace] = None
        start_time = time.perf_counter()

        if context:
            trace = tracer.start_trace(
                context=context,
                model=model,
                messages=messages,
                provider=_extract_provider(model),
            )

        try:
            # Call LiteLLM
            response = await acompletion(**augmented_params)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Handle streaming
            if stream:
                # For streaming, we wrap the generator
                return StreamingResponse(
                    _trace_streaming_response(
                        response, trace, tracer, latency_ms, context
                    ),
                    media_type="text/event-stream",
                )

            # Convert response to dict
            response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

            # Complete trace
            if trace:
                telemetry = extract_telemetry_from_response(response_dict)

                # Extract response text
                response_text = None
                if "choices" in response_dict and len(response_dict["choices"]) > 0:
                    choice = response_dict["choices"][0]
                    if "message" in choice:
                        response_text = choice["message"].get("content", "")

                tracer.complete_trace(
                    trace,
                    response_text=response_text,
                    prompt_tokens=telemetry.get("prompt_tokens"),
                    completion_tokens=telemetry.get("completion_tokens"),
                    logprobs=telemetry.get("logprobs"),
                    prompt_length=telemetry.get("prompt_length"),
                    completion_length=telemetry.get("completion_length"),
                    latency_ms=latency_ms,
                    finish_reason=telemetry.get("finish_reason"),
                )

            return JSONResponse(content=response_dict)

        except Exception as e:
            logger.error(f"Error in chat completion: {e}", exc_info=True)

            # Log error in trace
            if trace:
                latency_ms = (time.perf_counter() - start_time) * 1000
                tracer.complete_trace(trace, error=str(e), latency_ms=latency_ms)

            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(
        request: Request,
        context: Optional[SessionContext] = Depends(get_session_context_dependency),
    ):
        """
        OpenAI-compatible text completions endpoint.

        Args:
            request: FastAPI request
            context: Session context from headers

        Returns:
            Completion response
        """
        # Parse request body
        body = await request.json()

        # Extract parameters
        prompt = body.get("prompt", "")
        model = body.get("model", "gpt-3.5-turbo")
        stream = body.get("stream", False)

        # Augment with telemetry flags
        augmented_params = augment_request_params(
            body,
            enable_logprobs=request.app.state.enable_logprobs,
            enable_prompt_tokens=request.app.state.enable_prompt_tokens,
        )

        # Start tracing
        tracer = get_tracer()
        trace: Optional[LLMCallTrace] = None
        start_time = time.perf_counter()

        if context:
            trace = tracer.start_trace(
                context=context,
                model=model,
                prompt=prompt,
                provider=_extract_provider(model),
            )

        try:
            # Call LiteLLM
            response = await acompletion(**augmented_params)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Handle streaming
            if stream:
                return StreamingResponse(
                    _trace_streaming_response(
                        response, trace, tracer, latency_ms, context
                    ),
                    media_type="text/event-stream",
                )

            # Convert response to dict
            response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

            # Complete trace
            if trace:
                telemetry = extract_telemetry_from_response(response_dict)

                # Extract response text
                response_text = None
                if "choices" in response_dict and len(response_dict["choices"]) > 0:
                    choice = response_dict["choices"][0]
                    response_text = choice.get("text", "")

                tracer.complete_trace(
                    trace,
                    response_text=response_text,
                    prompt_tokens=telemetry.get("prompt_tokens"),
                    completion_tokens=telemetry.get("completion_tokens"),
                    logprobs=telemetry.get("logprobs"),
                    prompt_length=telemetry.get("prompt_length"),
                    completion_length=telemetry.get("completion_length"),
                    latency_ms=latency_ms,
                    finish_reason=telemetry.get("finish_reason"),
                )

            return JSONResponse(content=response_dict)

        except Exception as e:
            logger.error(f"Error in completion: {e}", exc_info=True)

            # Log error in trace
            if trace:
                latency_ms = (time.perf_counter() - start_time) * 1000
                tracer.complete_trace(trace, error=str(e), latency_ms=latency_ms)

            raise HTTPException(status_code=500, detail=str(e))

    return app


async def _trace_streaming_response(
    response_generator,
    trace: Optional[LLMCallTrace],
    tracer,
    latency_ms: float,
    context: Optional[SessionContext],
):
    """
    Wrap streaming response to collect telemetry.

    Args:
        response_generator: LiteLLM streaming response
        trace: Active trace
        tracer: Tracer instance
        latency_ms: Initial latency
        context: Session context

    Yields:
        SSE chunks
    """
    collected_text = []
    finish_reason = None

    try:
        async for chunk in response_generator:
            # Collect text for tracing
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if hasattr(delta, "content") and delta.content:
                        collected_text.append(delta.content)
                if hasattr(choice, "finish_reason") and choice.finish_reason:
                    finish_reason = choice.finish_reason

            # Yield chunk to client
            if hasattr(chunk, "model_dump_json"):
                yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                import json
                yield f"data: {json.dumps(dict(chunk))}\n\n"

        # Send done signal
        yield "data: [DONE]\n\n"

    finally:
        # Complete trace after streaming finishes
        if trace:
            response_text = "".join(collected_text)
            tracer.complete_trace(
                trace,
                response_text=response_text,
                completion_length=len(collected_text),
                latency_ms=latency_ms,
                finish_reason=finish_reason,
            )


def _extract_provider(model: str) -> str:
    """
    Extract provider from model name.

    Args:
        model: Model identifier

    Returns:
        Provider name
    """
    # LiteLLM model naming: provider/model
    if "/" in model:
        return model.split("/")[0]

    # Common prefixes
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("command"):
        return "cohere"
    elif model.startswith("gemini"):
        return "google"

    return "unknown"


# Factory function for creating app
def main():
    """Main entry point for running the proxy server."""
    import uvicorn

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
