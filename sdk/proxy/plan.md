# LiteLLM Proxy Integration Plan

This document sketches how to move tracing logic out of the Python SDK and into a LiteLLM proxy that fronts the upstream OpenAI-compatible endpoints.

## Goals
- Keep application code unchanged beyond pointing the OpenAI client at the proxy URL.
- Have the proxy mutate requests to request token-level telemetry (`logprobs`, `return_prompt`, etc.) whenever models support it.
- Record every call in the episodic context store via the `LLMTracer`, including prompts, completions, token ids, logprobs, timing, and metadata.

## High-Level Architecture
1. **Proxy Deployment**
   - Run LiteLLM in proxy mode (`litellm --proxy` or equivalent Docker image).
   - Configure upstream providers (OpenAI, Anthropic, Azure, vLLM, etc.) in the LiteLLM config file.
   - Register a custom middleware module that hooks request/response flow.

2. **Tracing Middleware**
   - Implement a LiteLLM proxy middleware (FastAPI dependency) that:
     - Extracts session/metadata headers injected by the SDK (e.g., `X-RLLM-Session`, `X-RLLM-Metadata`).
     - Adds sampling flags to the outgoing request (`logprobs`, `top_logprobs`, `return_prompt`, `return_token_ids`—names vary by provider, so map accordingly).
     - Measures latency across the upstream call.
     - After receiving the response, collects:
       - Prompt text (from request) and completion content.
       - Token IDs and logprobs when provided.
       - Usage statistics.
     - Calls `LLMTracer.log_llm_call(...)` with the assembled payload.

3. **Context Propagation**
   - SDK updates:
     - Inject session metadata into HTTP headers so the proxy can retrieve them. (Re-use the existing episodic context helpers.)
     - Default the OpenAI base URL to `http://<proxy-host>:<port>/v1`.

4. **Testing Strategy**
   - Unit-test middleware logic with mocked LiteLLM request/response objects.
   - Integration test: run LiteLLM proxy locally, point SDK client at it, verify traces land in an in-memory tracer store.
   - Load test: ensure added logging does not introduce unacceptable latency (consider async logging or background tasks).

## Implementation Steps
1. **Prototype Middleware**
   - Create `sdk/proxy/middleware.py` with FastAPI dependencies for before/after hooks.
   - Use LiteLLM’s documented hook interfaces: `proxy.add_middleware()` or the `callbacks` setting in `litellm.conf`.
2. **Session Header Contract**
   - Define headers: `X-RLLM-Session`, `X-RLLM-Metadata` (JSON).
   - Update SDK client to set these headers when a session is active.
3. **Telemetry Normalisation**
   - Map provider-specific telemetry objects (OpenAI: `choices[0].logprobs.content`, vLLM: `output_token_ids`, etc.) into a consistent format for the tracer.
4. **Tracer Integration**
   - Inject `LLMTracer` via dependency (FastAPI `Depends`) or initialize inside middleware with access to the episodic context client.
5. **Configuration & Deployment**
   - Provide Helm/docker-compose snippets showing proxy + tracer store deployment.
   - Document environment variables for credentials and routing.

## Open Questions
- Some providers charge for logprobs token; decide whether to enable by default or gate via config.
- Streaming support: decide whether to buffer full responses before logging or emit incremental trace events.
- Authentication passthrough: ensure user API keys are forwarded securely if the proxy is shared across teams.

