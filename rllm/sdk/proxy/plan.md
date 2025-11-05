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
     - Parses the metadata slug embedded in the routed path (see Metadata Slug Contract) and rewrites the ASGI scope back to the vanilla OpenAI path.
     - Stashes the decoded metadata dict on `request.state.rllm_metadata` and injects a monotonic `x-sequence-id` header so downstream callbacks can attribute every request without assuming fixed metadata keys.
     - Adds sampling flags to the outgoing request (`logprobs`, `top_logprobs`, `return_prompt`, `return_token_ids`—names vary by provider, so map accordingly).
     - Measures latency across the upstream call.
     - After receiving the response, collects:
       - Prompt text (from request) and completion content.
       - Token IDs and logprobs when provided.
       - Usage statistics.
     - Calls `LLMTracer.log_llm_call(...)` directly with the assembled payload plus the metadata dict from `request.state`.

3. **Context Propagation**
   - SDK updates:
     - Resolve the metadata slug lazily on every request: the wrapped OpenAI client reads the current session contextvars, serializes them, and issues the call against `http://<proxy-host>:<port>/meta/{slug}/v1`. A single client instance can therefore serve multiple sessions without re-instantiation.
     - Fall back to the plain proxy URL (`…/v1`) when no session is active so production traffic can route through the proxy without attribution.

4. **Testing Strategy**
   - Unit-test middleware logic with mocked LiteLLM request/response objects.
   - Integration test: run LiteLLM proxy locally, point SDK client at it, verify traces land in an in-memory tracer store.
   - Load test: ensure added logging does not introduce unacceptable latency (consider async logging or background tasks).

## Implementation Steps
1. **Prototype Middleware**
   - Create `sdk/proxy/middleware.py` with FastAPI dependencies for before/after hooks.
   - Use LiteLLM’s documented hook interfaces: `proxy.add_middleware()` or the `callbacks` setting in `litellm.conf`.
2. **Metadata Slug Contract**
   - Define a metadata serializer that accepts the full session context dict (e.g., `session_id`, `split`, `job`, arbitrary key/value pairs) and produces a compact, URL-safe slug (base64url-encoded JSON with a version prefix for forward compatibility).
   - Update SDK helpers (`get_chat_client`, `@entrypoint`) to construct provider clients using the rewritten base URL derived from session contextvars via the serializer.
   - Ensure wrapped OpenAI clients recompute the base URL before each request so a single client instance can serve different sessions/metadata scopes without recreation.
3. **Middleware Logging & Telemetry**
   - Extend the proxy middleware to capture request/response payloads, normalize token usage fields, and submit them to `LLMTracer.log_llm_call(...)` using the decoded metadata dict.
   - Treat custom LiteLLM callbacks as optional extensions (e.g., to push metrics elsewhere) rather than the primary logging path.
4. **Tracer Integration**
   - Wire the middleware to receive an `LLMTracer` instance (FastAPI dependency or app state) and ensure logging happens asynchronously without blocking the proxy event loop.
5. **Configuration & Deployment**
   - Provide Helm/docker-compose snippets showing proxy + tracer store deployment.
   - Document environment variables for credentials and routing.

## Design Notes & Open Questions
- **Metadata slug versioning**: encode slugs as `rllm1:<base64url(json)>` so the proxy can validate and evolve the schema without breaking older clients.
- **Logprob sampling cost**: make `return_logprobs` opt-in via config/metadata so high-cost providers (Anthropic, Gemini) can be excluded without code changes. Consider per-model defaults plus per-request overrides in the metadata slug.
- **Streaming handling**: accumulate streamed chunks in `request.state` (content, logprobs, token ids) and flush a single `LLMTracer.log_llm_call(...)` when LiteLLM yields the final chunk or errors, to keep tracer payloads consistent with non-streaming calls.
- **Authentication passthrough**: ensure user API keys are forwarded securely if the proxy is shared across teams.
