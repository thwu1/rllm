# Design Document: Logprobs Extraction for LiteLLM Proxy

**Status**: Draft
**Author**: Claude
**Date**: 2025-11-05
**Related**: `plan.md`, `litellm_callbacks.py`

## Overview

This document describes the design for extracting and storing token-level logprobs from LLM responses in the rLLM SDK proxy integration. The goal is to capture detailed token probability information for all LLM calls routed through the LiteLLM proxy, enabling downstream analysis, debugging, and training workflows.

## Background

### Current State

The proxy implementation currently:
- ✅ Requests logprobs via `SamplingParametersCallback` (sets `logprobs: true`)
- ✅ Logs LLM calls via `TracingCallback`
- ❌ **Does NOT extract or store logprobs from responses**

The `TracingCallback.async_log_success_event()` receives the full response object but only extracts basic usage statistics. The logprobs data is present in the response but not captured.

### Goals

From `plan.md`:
> "Record every call in the episodic context store via the LLMTracer, including prompts, completions, **token ids, logprobs**, timing, and metadata."

**Primary Objectives**:
1. Extract token-level logprobs from all LLM responses
2. Store logprobs in a structured, queryable format
3. Support all backends uniformly (OpenAI, vLLM, Anthropic, etc.)
4. Handle edge cases gracefully (missing logprobs, provider variations)
5. Support vLLM token IDs when using passthrough mode

**Non-Goals** (for this phase):
- Streaming support (separate design doc)
- Prompt token IDs extraction
- Logprobs visualization or analysis tools

## Design Approach

### Key Insight: LiteLLM Normalization

**Critical Decision**: Use LiteLLM's normalized response format for logprobs, with optional passthrough mode for vLLM token IDs.

LiteLLM operates in **two modes**:

#### Mode 1: Translation Mode (Standard `/v1/` endpoints)

LiteLLM normalizes all provider responses (OpenAI, vLLM, Anthropic, Azure, Together, etc.) into the **OpenAI Chat Completions format**:

```python
# All backends produce this structure after LiteLLM normalization:
response.choices[0].logprobs.content = [
    {
        "token": "Hello",
        "logprob": -0.31725305,
        "bytes": [72, 101, 108, 108, 111],  # UTF-8 byte representation
        "top_logprobs": [  # Optional, if top_logprobs > 0
            {"token": "Hello", "logprob": -0.31725, "bytes": [...]},
            {"token": "Hi", "logprob": -1.31904, "bytes": [...]}
        ]
    },
    # ... one entry per output token
]
```

**Benefits**:
- ✅ Single extraction path for all providers
- ✅ Consistent data format in traces
- ✅ Leverages LiteLLM's normalization layer

**Limitations**:
- ⚠️ Provider-specific fields (e.g., vLLM's `token_ids`) are **dropped**
- ⚠️ Only OpenAI-standard fields are preserved

#### Mode 2: Passthrough Mode (vLLM `/vllm/` endpoints)

For vLLM specifically, LiteLLM offers passthrough endpoints that **preserve all vLLM-specific fields** without normalization:

```python
# vLLM passthrough response (with return_token_ids=true):
response = {
    "prompt_token_ids": [101, 102, 103, ...],  # Root-level field
    "choices": [
        {
            "token_ids": [201, 202, 203, ...],  # Completion token IDs
            "logprobs": {
                "content": [...]  # Standard logprobs
            },
            "message": {"content": "..."}
        }
    ]
}
```

**Usage**:
```bash
POST http://proxy:4000/vllm/chat/completions
{
  "model": "vllm-model",
  "messages": [...],
  "return_token_ids": true
}
```

**Trade-offs**:
- ✅ Preserves vLLM token IDs
- ⚠️ vLLM-specific (not available for OpenAI, Anthropic, etc.)
- ⚠️ Requires separate endpoint configuration

### Recommended Approach

**Use Translation Mode as the primary path**, with optional passthrough for vLLM when token IDs are needed:

1. **Default**: Extract OpenAI-format logprobs (works for all providers)
2. **Optional**: Add vLLM passthrough mode support for token IDs
3. **Graceful degradation**: If token IDs aren't available, store what we have

## Implementation Design

### 1. Response Structure

#### OpenAI Format (Translation Mode)

LiteLLM's ModelResponse follows OpenAI's structure:

```python
class ModelResponse:
    id: str
    choices: List[Choice]
    created: int
    model: str
    usage: Usage

class Choice:
    index: int
    message: Message
    logprobs: Optional[ChoiceLogprobs]
    finish_reason: str

class ChoiceLogprobs:
    content: Optional[List[ChatCompletionTokenLogprob]]

class ChatCompletionTokenLogprob:
    token: str
    logprob: float
    bytes: Optional[List[int]]  # UTF-8 bytes
    top_logprobs: List[TopLogprob]

class TopLogprob:
    token: str
    logprob: float
    bytes: Optional[List[int]]
```

#### vLLM Format (Passthrough Mode)

When using `/vllm/` endpoints with `return_token_ids=true`:

```python
{
    "prompt_token_ids": [int, ...],  # At root level
    "choices": [{
        "token_ids": [int, ...],     # At choice level
        "logprobs": { ... },          # Standard logprobs
        ...
    }]
}
```

### 2. Extraction Function

Add to `litellm_callbacks.py`:

```python
def _extract_logprobs(self, response_obj: Any) -> Optional[Dict[str, Any]]:
    """Extract logprobs from LiteLLM-normalized response.

    Args:
        response_obj: LiteLLM ModelResponse object

    Returns:
        Dict with extracted logprobs, or None if not available:
        {
            "tokens": ["Hello", "world", "!"],
            "logprobs": [-0.317, -0.512, -0.089],
            "bytes": [[72,101,108,108,111], [119,111,114,108,100], [33]],
            "top_logprobs": [
                [{"token": "Hello", "logprob": -0.317, "bytes": [...]}],
                [{"token": "world", "logprob": -0.512, "bytes": [...]}],
                [{"token": "!", "logprob": -0.089, "bytes": [...]}]
            ]
        }
    """
    # Initialize result structure
    result = {
        "tokens": [],
        "logprobs": [],
        "bytes": [],
        "top_logprobs": []
    }

    # Extract from all choices (typically just one for chat completions)
    choices = getattr(response_obj, 'choices', [])
    if not choices:
        return None

    # Iterate through choices and collect logprobs
    for choice in choices:
        choice_logprobs = getattr(choice, 'logprobs', None)
        if not choice_logprobs:
            continue

        content = getattr(choice_logprobs, 'content', None)
        if not content:
            continue

        # Extract each token's logprob information
        for item in content:
            # Main token info
            result["tokens"].append(getattr(item, 'token', None))
            result["logprobs"].append(getattr(item, 'logprob', None))
            result["bytes"].append(getattr(item, 'bytes', None))

            # Top alternative tokens (if requested via top_logprobs)
            top_lps = getattr(item, 'top_logprobs', None) or []
            serialized_top = []
            for top_item in top_lps:
                serialized_top.append({
                    "token": getattr(top_item, 'token', None),
                    "logprob": getattr(top_item, 'logprob', None),
                    "bytes": getattr(top_item, 'bytes', None)
                })
            result["top_logprobs"].append(serialized_top)

    # Return None if no logprobs were found
    if not result["tokens"]:
        return None

    return result


def _extract_vllm_token_ids(self, response_obj: Any) -> Optional[Dict[str, Any]]:
    """Extract vLLM token IDs (passthrough mode only).

    This only works when using LiteLLM's vLLM passthrough endpoint (/vllm/)
    with return_token_ids=true. In translation mode, these fields are dropped.

    Args:
        response_obj: Raw vLLM response (via passthrough)

    Returns:
        Dict with token IDs, or None if not available:
        {
            "prompt": [101, 102, 103, ...],
            "completion": [201, 202, 203, ...]
        }
    """
    result = {}

    # vLLM returns prompt_token_ids at root level
    if hasattr(response_obj, 'prompt_token_ids'):
        prompt_ids = getattr(response_obj, 'prompt_token_ids', None)
        if prompt_ids:
            result['prompt'] = prompt_ids

    # Completion token_ids are in choices
    choices = getattr(response_obj, 'choices', [])
    if choices:
        choice = choices[0]
        if hasattr(choice, 'token_ids'):
            completion_ids = getattr(choice, 'token_ids', None)
            if completion_ids:
                result['completion'] = completion_ids

    return result if result else None
```

### 3. Integration with TracingCallback

Modify `TracingCallback.async_log_success_event()`:

```python
async def async_log_success_event(
    self,
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: float,
    end_time: float
) -> None:
    """Called after successful LLM completion."""
    # ... existing metadata extraction ...

    # Extract logprobs from response (works for all providers)
    logprobs_data = self._extract_logprobs(response_obj)

    # Optionally extract vLLM token IDs (passthrough mode only)
    token_ids_data = self._extract_vllm_token_ids(response_obj)

    # Build output dict
    response_dict = response_obj.model_dump() if hasattr(response_obj, "model_dump") else {}
    output = {
        "response": response_dict,
    }

    # Add extracted data if available
    if logprobs_data:
        output["logprobs"] = logprobs_data

    if token_ids_data:
        output["token_ids"] = token_ids_data

    # ... existing token usage extraction ...

    self.tracer.log_llm_call(
        name=f"proxy/{model}",
        model=model,
        input={"messages": messages},
        output=output,  # Now includes logprobs and token_ids
        metadata=metadata,
        session_id=metadata.get("session_id"),
        latency_ms=latency_ms,
        tokens=tokens,
    )
```

### 4. Trace Schema

#### Translation Mode (Standard)

```json
{
  "name": "proxy/gpt-4o-mini",
  "model": "gpt-4o-mini",
  "input": {
    "messages": [{"role": "user", "content": "Hello"}]
  },
  "output": {
    "response": {
      "id": "chatcmpl-...",
      "choices": [...],
      "usage": {...}
    },
    "logprobs": {
      "tokens": ["Hello", " world", "!"],
      "logprobs": [-0.317, -0.512, -0.089],
      "bytes": [[72,101,108,108,111], [119,111,114,108,100], [33]],
      "top_logprobs": [
        [
          {"token": "Hello", "logprob": -0.317, "bytes": [72,101,108,108,111]},
          {"token": "Hi", "logprob": -1.319, "bytes": [72,105]}
        ]
      ]
    }
  },
  "latency_ms": 234.5,
  "tokens": {"prompt": 5, "completion": 3, "total": 8},
  "session_id": "sess-123"
}
```

#### Passthrough Mode (vLLM with token IDs)

```json
{
  "name": "proxy/vllm-model",
  "model": "vllm-model",
  "output": {
    "response": {...},
    "logprobs": {
      "tokens": ["Hello", " world", "!"],
      "logprobs": [-0.317, -0.512, -0.089],
      "bytes": [...],
      "top_logprobs": [...]
    },
    "token_ids": {
      "prompt": [101, 102, 103, 104, 105],
      "completion": [201, 202, 203]
    }
  }
}
```

## Configuration

### 1. Translation Mode (Default)

```yaml
# litellm_proxy_config.yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      drop_params: true

  - model_name: vllm-model
    litellm_params:
      model: hosted_vllm/my-model
      api_base: https://my-vllm-server.com
```

Call via standard endpoint:
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...]
)
# Logprobs extracted, token_ids not available
```

### 2. Passthrough Mode (vLLM Token IDs)

Same config, but call via passthrough endpoint:
```python
# Use vLLM passthrough endpoint
import openai
client = openai.OpenAI(base_url="http://proxy:4000/vllm")

response = client.chat.completions.create(
    model="vllm-model",
    messages=[...],
    extra_body={"return_token_ids": True}
)
# Both logprobs AND token_ids extracted
```

### 3. Per-Model Logprobs Control

From `plan.md` requirement:
> "Logprob sampling cost: make `return_logprobs` opt-in via config/metadata"

Add configuration to `SamplingParametersCallback`:

```python
class SamplingParametersCallback(CustomLogger):
    def __init__(
        self,
        add_return_token_ids: bool = False,
        logprobs_config: Optional[Dict[str, bool]] = None
    ):
        """
        Args:
            add_return_token_ids: Enable token_ids for vLLM passthrough
            logprobs_config: Per-model logprobs control:
                {
                    "default": True,           # Default behavior
                    "claude-*": False,         # Disable for Anthropic
                    "gemini-*": False,         # Disable for Gemini
                    "gpt-4o-mini": True        # Enable for specific model
                }
        """
        self.logprobs_config = logprobs_config or {"default": True}
        self.add_return_token_ids = add_return_token_ids

    def _should_request_logprobs(self, model: str) -> bool:
        """Check if logprobs should be requested for this model."""
        # Check exact match first
        if model in self.logprobs_config:
            return self.logprobs_config[model]

        # Check wildcard patterns
        for pattern, enabled in self.logprobs_config.items():
            if pattern.endswith("*") and model.startswith(pattern[:-1]):
                return enabled

        # Default
        return self.logprobs_config.get("default", True)

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or {}
        model = data.get("model", "")

        result = {**data}

        # Only add logprobs if enabled for this model
        if self._should_request_logprobs(model):
            result["logprobs"] = True

        # Only add return_token_ids for vLLM in passthrough mode
        if self.add_return_token_ids and self._supports_token_ids(model):
            result["return_token_ids"] = True

        # ... rest of existing logic ...
        return result
```

**Usage in proxy_app.py**:

```python
# Disable logprobs for expensive providers
litellm.callbacks.append(SamplingParametersCallback(
    add_return_token_ids=False,  # Enable for vLLM passthrough
    logprobs_config={
        "default": True,
        "claude-*": False,      # Anthropic doesn't support logprobs well
        "gemini-*": False,      # Google charges extra for logprobs
        "gpt-4o-mini": True,    # Keep for cheap models
    }
))
```

## Edge Cases & Error Handling

### 1. Missing Logprobs

**Scenario**: Provider doesn't support logprobs, or they weren't requested.

**Handling**:
- `_extract_logprobs()` returns `None`
- Trace stored without logprobs field
- No error logged (this is expected behavior)

### 2. Partial Logprobs

**Scenario**: Some tokens have logprobs, others don't (malformed response).

**Handling**:
- Use `getattr(item, 'logprob', None)` to handle missing attributes
- Store `None` for missing values
- Continue processing remaining tokens

### 3. Multiple Choices

**Scenario**: Response has multiple choices (e.g., `n > 1`).

**Current Handling**: Iterate all choices, concatenate logprobs.

**Future Consideration**: Choice-level separation:
```python
{
  "logprobs": [
    {"choice_index": 0, "tokens": [...], "logprobs": [...]},
    {"choice_index": 1, "tokens": [...], "logprobs": [...]}
  ]
}
```

### 4. vLLM Token IDs in Translation Mode

**Scenario**: User expects token IDs but is using translation mode.

**Handling**:
- `_extract_vllm_token_ids()` returns `None`
- Document that token IDs require passthrough mode
- Provide clear migration path in docs

### 5. Mixed Providers

**Scenario**: Some models support logprobs, others don't.

**Handling**:
- Per-model configuration via `logprobs_config`
- Graceful degradation (store what's available)
- No errors for unsupported models

## Performance Considerations

### Memory

Logprobs can be large:
- 1000 token response = ~1000 logprob entries
- With `top_logprobs=5`: ~5000 alternative token entries
- With token IDs: +8 bytes per token
- Estimated size: 50-150 KB per trace (with top_logprobs and token_ids)

**Mitigation**:
- Async background worker already handles buffering (LLMTracer)
- Per-model opt-out for expensive providers

### Latency

Extraction happens in the callback (not on request path):
- ~0.1-1ms for logprobs extraction
- ~0.01ms for token IDs extraction
- Non-blocking storage via LLMTracer worker

**Impact**: Negligible (callback already processes response).

### Storage

- Episodic context store handles compression
- Logprobs add 10-50% to trace size
- Token IDs add 5-10% to trace size
- Acceptable for most use cases

## Testing Strategy

### Unit Tests

Add to `rllm/sdk/test_proxy_callbacks.py`:

```python
def test_extract_logprobs_basic():
    """Test basic logprobs extraction from OpenAI-format response."""
    mock_response = MockModelResponse(
        choices=[
            MockChoice(
                logprobs=MockChoiceLogprobs(
                    content=[
                        MockTokenLogprob(
                            token="Hello",
                            logprob=-0.317,
                            bytes=[72, 101, 108, 108, 111],
                            top_logprobs=[]
                        )
                    ]
                )
            )
        ]
    )

    callback = TracingCallback(mock_tracer)
    result = callback._extract_logprobs(mock_response)

    assert result is not None
    assert result["tokens"] == ["Hello"]
    assert result["logprobs"] == [-0.317]
    assert result["bytes"] == [[72, 101, 108, 108, 111]]


def test_extract_logprobs_missing():
    """Test extraction when logprobs are not present."""
    mock_response = MockModelResponse(choices=[MockChoice(logprobs=None)])

    callback = TracingCallback(mock_tracer)
    result = callback._extract_logprobs(mock_response)

    assert result is None


def test_extract_vllm_token_ids():
    """Test vLLM token IDs extraction (passthrough mode)."""
    mock_response = MockModelResponse(
        prompt_token_ids=[101, 102, 103],
        choices=[MockChoice(token_ids=[201, 202, 203])]
    )

    callback = TracingCallback(mock_tracer)
    result = callback._extract_vllm_token_ids(mock_response)

    assert result is not None
    assert result["prompt"] == [101, 102, 103]
    assert result["completion"] == [201, 202, 203]


def test_extract_vllm_token_ids_missing():
    """Test token IDs extraction when not in passthrough mode."""
    mock_response = MockModelResponse(
        choices=[MockChoice()]  # No token_ids field
    )

    callback = TracingCallback(mock_tracer)
    result = callback._extract_vllm_token_ids(mock_response)

    assert result is None


def test_per_model_logprobs_config():
    """Test per-model logprobs enable/disable."""
    callback = SamplingParametersCallback(
        logprobs_config={
            "default": True,
            "claude-*": False,
            "gpt-4o": True
        }
    )

    assert callback._should_request_logprobs("gpt-4o-mini") == True
    assert callback._should_request_logprobs("claude-3-opus") == False
    assert callback._should_request_logprobs("gpt-4o") == True
    assert callback._should_request_logprobs("unknown-model") == True
```

### Integration Tests

Add to `examples/proxy_demo/test_proxy_integration.py`:

```python
async def test_logprobs_end_to_end_translation_mode():
    """Test that logprobs are captured in translation mode."""
    # Start proxy with callbacks
    # Make OpenAI call through proxy
    # Query traces
    # Assert logprobs are present, token_ids are not
    pass


async def test_logprobs_end_to_end_passthrough_mode():
    """Test that logprobs AND token_ids are captured in passthrough mode."""
    # Start proxy with vLLM passthrough endpoint
    # Make call with return_token_ids=true
    # Query traces
    # Assert both logprobs and token_ids are present
    pass
```

### Manual Testing

```bash
# 1. Start proxy
export OPENAI_API_KEY="sk-..."
python examples/proxy_demo/proxy_app.py

# 2. Test translation mode (standard)
python -c "
from rllm.sdk import RLLMClient

client = RLLMClient(
    api_key='sk-...',
    base_url='http://localhost:4000/v1',
    project='logprobs-test'
)

with client.session('test-session'):
    chat = client.get_chat_client()
    response = chat.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}],
    )

traces = client.get_session_traces('test-session')
print('Logprobs:', traces[0]['output'].get('logprobs'))
print('Token IDs:', traces[0]['output'].get('token_ids'))  # Should be None
"

# 3. Test passthrough mode (vLLM with token IDs)
python -c "
import openai

client = openai.OpenAI(
    api_key='dummy',
    base_url='http://localhost:4000/vllm'
)

response = client.chat.completions.create(
    model='vllm-model',
    messages=[{'role': 'user', 'content': 'Hello'}],
    extra_body={'return_token_ids': True}
)

# Check trace in episodic store - should have both logprobs and token_ids
"
```

## Future Work

### 1. Streaming Support

**Requirement**: Accumulate logprobs from streamed chunks.

**Design Consideration**:
- Store partial logprobs in `request.state`
- Flush on stream completion
- Handle interruptions gracefully
- Accumulate token_ids for vLLM streaming

**Separate Design Doc**: See `streaming_logprobs_design.md` (future)

### 2. Prompt Token IDs Extraction

**Current**: Only completion token IDs are extracted.

**Future**: Extract `prompt_token_ids` from vLLM passthrough.

**Challenge**: Need to associate with correct input message.

### 3. Logprobs Analysis Tools

**Future Features**:
- Entropy calculation
- Confidence scoring
- Token-level debugging UI
- Retokenization drift detection

**Out of Scope**: This design focuses on capture, not analysis.

### 4. Alternative Token ID Sources

**Exploration**: OpenAI's `bytes` field can be used to reconstruct tokens.

**Benefits**: Works across all providers in translation mode.

**Trade-offs**: Requires byte-to-token-id mapping, model-specific.

## Alternatives Considered

### Alternative 1: Provider-Specific Parsing

**Approach**: Detect provider and use custom extraction logic.

```python
if "openai" in model:
    extract_openai_logprobs(response)
elif "vllm" in model:
    extract_vllm_logprobs(response)
```

**Rejected Because**:
- ❌ High maintenance burden
- ❌ Breaks when providers change formats
- ❌ LiteLLM already normalizes responses
- ❌ Duplicates LiteLLM's normalization logic

### Alternative 2: Raw Response Storage

**Approach**: Store entire raw response, parse later.

**Rejected Because**:
- ❌ Wastes storage space
- ❌ Requires client-side parsing
- ❌ Loses normalization benefits
- ❌ Hard to query/analyze

### Alternative 3: Client-Side Extraction

**Approach**: Extract in `ProxyTrackedChatClient` before sending to proxy.

**Rejected Because**:
- ❌ Proxy doesn't see logprobs (can't log them)
- ❌ Duplicates logic across SDK and proxy
- ❌ Doesn't work for direct proxy calls

### Alternative 4: Always Use Passthrough Mode

**Approach**: Only support vLLM passthrough, skip translation mode.

**Rejected Because**:
- ❌ Loses multi-provider support
- ❌ vLLM-only solution
- ❌ Breaks OpenAI, Anthropic, etc.
- ❌ Requires separate endpoints per provider

## Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Use LiteLLM normalized format for logprobs | Single extraction path, provider-agnostic | 2025-11-05 |
| Add optional vLLM passthrough support | Preserves token IDs when needed | 2025-11-05 |
| Store logprobs in `output.logprobs` | Clean separation from raw response | 2025-11-05 |
| Store token_ids in `output.token_ids` | Separate from logprobs structure | 2025-11-05 |
| Extract in `TracingCallback` | Centralized logging point | 2025-11-05 |
| Defensive attribute access | Graceful handling of variations | 2025-11-05 |
| Per-model logprobs configuration | Cost control for expensive providers | 2025-11-05 |
| Defer streaming to future work | Complexity, separate design needed | 2025-11-05 |

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM vLLM Passthrough](https://docs.litellm.ai/docs/pass_through/vllm)
- [OpenAI Logprobs Format](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs)
- [vLLM return_token_ids Feature](https://blog.vllm.ai/2025/10/22/agent-lightning.html)
- [rLLM SDK Proxy Plan](./plan.md)
- [Episodic Tracing Docs](https://github.com/episodic-ai/episodic)

## Appendix A: Example Responses

### OpenAI Response (Translation Mode)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello world!"
      },
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.31725305,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
              {"token": "Hello", "logprob": -0.31725305, "bytes": [72, 101, 108, 108, 111]},
              {"token": "Hi", "logprob": -1.3190403, "bytes": [72, 105]}
            ]
          },
          {
            "token": " world",
            "logprob": -0.0123456,
            "bytes": [32, 119, 111, 114, 108, 100],
            "top_logprobs": []
          },
          {
            "token": "!",
            "logprob": -0.08935,
            "bytes": [33],
            "top_logprobs": []
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 3,
    "total_tokens": 8
  }
}
```

### vLLM Response (Passthrough Mode)

```json
{
  "id": "cmpl-xyz789",
  "model": "vllm-model",
  "prompt_token_ids": [101, 102, 103, 104, 105],
  "choices": [
    {
      "index": 0,
      "token_ids": [201, 202, 203],
      "message": {
        "role": "assistant",
        "content": "Hello world!"
      },
      "logprobs": {
        "content": [
          {
            "token": "Hello",
            "logprob": -0.31725305,
            "bytes": [72, 101, 108, 108, 111]
          },
          {
            "token": " world",
            "logprob": -0.0123456,
            "bytes": [32, 119, 111, 114, 108, 100]
          },
          {
            "token": "!",
            "logprob": -0.08935,
            "bytes": [33]
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 3,
    "total_tokens": 8
  }
}
```

## Appendix B: Extraction Examples

### Extracted Logprobs (All Providers)

```json
{
  "tokens": ["Hello", " world", "!"],
  "logprobs": [-0.31725305, -0.0123456, -0.08935],
  "bytes": [
    [72, 101, 108, 108, 111],
    [32, 119, 111, 114, 108, 100],
    [33]
  ],
  "top_logprobs": [
    [
      {"token": "Hello", "logprob": -0.31725305, "bytes": [72, 101, 108, 108, 111]},
      {"token": "Hi", "logprob": -1.3190403, "bytes": [72, 105]}
    ],
    [],
    []
  ]
}
```

### Extracted Token IDs (vLLM Passthrough Only)

```json
{
  "prompt": [101, 102, 103, 104, 105],
  "completion": [201, 202, 203]
}
```

### Combined Trace (vLLM Passthrough)

```json
{
  "name": "proxy/vllm-model",
  "model": "vllm-model",
  "input": {
    "messages": [{"role": "user", "content": "Hello"}]
  },
  "output": {
    "response": { "id": "...", "choices": [...] },
    "logprobs": {
      "tokens": ["Hello", " world", "!"],
      "logprobs": [-0.317, -0.012, -0.089],
      "bytes": [[72,101,108,108,111], [32,119,111,114,108,100], [33]],
      "top_logprobs": [[...], [], []]
    },
    "token_ids": {
      "prompt": [101, 102, 103, 104, 105],
      "completion": [201, 202, 203]
    }
  },
  "latency_ms": 234.5,
  "tokens": {"prompt": 5, "completion": 3, "total": 8},
  "session_id": "sess-123"
}
```
