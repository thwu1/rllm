"""Data processing utilities for converting trajectories to training data.

This module provides the bridge between trajectory generation and training,
handling filtering, advantage computation, and conversion to training formats.

## Trace Structure

Traces are LLM call records created by the proxy and consumed by agent_omni_engine.
They flow through the following pipeline:

1. **Created by**: TracingCallback in litellm_callbacks.py
2. **Stored in**: Episodic Context Store via LLMTracer.log_llm_call()
3. **Retrieved by**: agent_omni_engine via context subscriber
4. **Converted to**: Step objects via trace_to_step()

### Trace Dictionary Format

A trace is a dictionary with the following structure:

```python
trace = {
    # Core LLM call information
    "name": str,              # e.g., "proxy/gpt-4"
    "model": str,             # e.g., "gpt-4", "claude-3-opus"
    "trace_id": str,          # e.g., "tr_abc123def456"
    "timestamp": float,       # Unix timestamp

    # Input/Output
    "input": {
        "messages": list[dict]  # OpenAI-style messages array
    },
    "output": {
        "choices": [
            {
                "message": {
                    "content": str,      # Response text
                    "reasoning": str,    # Optional reasoning (for o1 models)
                    "role": str,         # Usually "assistant"
                },
                "finish_reason": str,    # e.g., "stop", "length"
                "provider_specific_fields": {
                    "token_ids": list[int]  # Completion token IDs (vLLM only)
                }
            }
        ],
        "prompt_token_ids": list[int],  # Prompt token IDs (vLLM only)
        # ... other OpenAI response fields
    },

    # Metadata
    "metadata": {
        "session_id": str,  # Format: "task_id:rollout_idx:retry_attempt"
        "job": str,         # Optional job identifier
        # ... other custom metadata from middleware
    },

    # Performance metrics
    "latency_ms": float,
    "tokens": {
        "prompt": int,
        "completion": int,
        "total": int
    },

    # Optional fields
    "session_id": str,      # Same as metadata.session_id
    "contexts": list,       # Context elements used
    "tools": list[dict],    # Available tools
    "cost": float,          # USD cost
    "environment": str,     # e.g., "production"
}
```

### Critical Fields for Training

| Field | Purpose | Required? |
|-------|---------|-----------|
| `session_id` | Groups traces into episodes | ✅ Yes |
| `output.prompt_token_ids` | Training data (prompt tokens) | ✅ Yes (vLLM) |
| `output.choices[0].provider_specific_fields.token_ids` | Training data (completion tokens) | ✅ Yes (vLLM) |
| `input.messages` | Conversation context | ✅ Yes |
| `output.choices[0].message` | Response message | ✅ Yes |
| `metadata` | Additional context (stored in Step.info) | Optional |

### Important Notes

1. **Token IDs are vLLM-specific**: The `prompt_token_ids` and `provider_specific_fields.token_ids`
   are only available when using vLLM backend. OpenAI/Anthropic don't provide these.

2. **Session ID is critical**: The `session_id` (format: `"task_id:rollout_idx:retry_attempt"`)
   is used to group traces into episodes and must be injected via metadata.

3. **Event wrapper**: Traces are wrapped in an event object with `{"type": "...", "data": trace}`
   when retrieved from the context subscriber.

4. **Batch end signal**: A special trace with `type="trace_batch_end"` signals completion of a batch.
"""

import logging
import uuid
from collections import defaultdict

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout import ModelOutput

logger = logging.getLogger(__name__)


def trace_to_model_output(trace: dict) -> ModelOutput:
    """Convert a trace dictionary to a ModelOutput object.

    Extracts token IDs, content, and other fields from the trace's output section.

    Args:
        trace: Trace dictionary from the context store (see module docstring for structure)

    Returns:
        ModelOutput object containing:
            - prompt_ids: Token IDs for the prompt (from output.prompt_token_ids)
            - completion_ids: Token IDs for the completion (from provider_specific_fields.token_ids)
            - content: Text content of the response
            - reasoning: Optional reasoning text (for o1 models)
            - finish_reason: Why generation stopped (e.g., "stop", "length")

    Raises:
        AssertionError: If required fields are missing (output, choices, prompt_ids, completion_ids)

    Note:
        - Token IDs are only available with vLLM backend
        - OpenAI/Anthropic providers don't return token_ids, so this will fail for those
    """
    output = trace.get("output", {})

    # Extract prompt token IDs (vLLM-specific field)
    prompt_ids = output.get("prompt_token_ids", [])

    # Extract response choices (OpenAI-compatible format)
    choices = output.get("choices", [])

    # Extract message content and reasoning from first choice
    content = choices[0].get("message", {}).get("content", "")
    reasoning = choices[0].get("message", {}).get("reasoning", "")

    # Extract completion token IDs from provider-specific fields (vLLM only)
    provider_specific_fields = choices[0].get("provider_specific_fields", {})
    completion_ids = provider_specific_fields.get("token_ids", [])

    # Validate required fields
    assert output, trace
    assert len(choices) == 1, "Only one choice is supported for now"
    assert prompt_ids, "Prompt IDs are required"
    assert completion_ids, "Completion IDs are required"

    return ModelOutput(
        text="",  # Not used in current implementation
        content=content,
        reasoning=reasoning,
        tool_calls=[],  # TODO: Extract tool calls from message if present
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=[],  # TODO: Extract logprobs if available
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=choices[0].get("finish_reason", "stop"),
    )


def trace_to_step(trace: dict) -> Step:
    """Convert a trace dictionary to a Step object.

    This is the main conversion function used by agent_omni_engine to transform
    LLM call traces into Step objects that can be used for training.

    Args:
        trace: Trace dictionary from the context store (see module docstring for structure)

    Returns:
        Step object containing:
            - chat_completions: Full conversation history (input messages + response message)
            - model_output: ModelOutput with token IDs and other generation details
            - info: Metadata dictionary from the trace (includes session_id, job, etc.)

    Raises:
        AssertionError: If response_message is missing

    Example:
        ```python
        # Trace from context store
        trace = {
            "input": {"messages": [{"role": "user", "content": "Hello"}]},
            "output": {
                "choices": [{
                    "message": {"role": "assistant", "content": "Hi!"},
                    "provider_specific_fields": {"token_ids": [123, 456]}
                }],
                "prompt_token_ids": [1, 2, 3]
            },
            "metadata": {"session_id": "task_123:0:0"}
        }

        # Convert to Step
        step = trace_to_step(trace)
        # step.chat_completions = [
        #     {"role": "user", "content": "Hello"},
        #     {"role": "assistant", "content": "Hi!"}
        # ]
        # step.model_output.prompt_ids = [1, 2, 3]
        # step.model_output.completion_ids = [123, 456]
        # step.info = {"session_id": "task_123:0:0"}
        ```

    Note:
        - The chat_completions field contains the FULL conversation history including the response
        - This is used by the training pipeline to reconstruct the prompt for each step
        - The info field preserves metadata that can be used for filtering/grouping
    """
    # Extract input messages (conversation history before this LLM call)
    messages = trace.get("input", {}).get("messages", [])

    # Extract response message (the LLM's response)
    response_message = trace.get("output", {}).get("choices", [])[0].get("message", {})

    assert response_message, "Response message is required in trace output"

    return Step(
        # Full conversation: input messages + response message
        chat_completions=messages + [response_message],
        # Structured output with token IDs for training
        model_output=trace_to_model_output(trace),
        # Preserve metadata (session_id, job, etc.) for downstream processing
        info=trace.get("metadata", {}),
    )


def get_trajectory_name(steps: list[Step], name_key: str | None = None) -> str:
    if name_key is None:
        return "agent"
    else:
        return steps[0].info.get(name_key, "agent")


def group_steps(steps: list[Step], by: str | None = None, name_key: str | None = None) -> list[Trajectory]:
    # if some step doesnt have the group key, we assign a random key to avoid grouping them together
    # in this case, the grpo reduce to reinforce
    if by is None:
        return [Trajectory(name="agent", steps=steps)]
    else:
        step_groups = defaultdict(list)
        for step in steps:
            step_groups[step.info.get(by, str(uuid.uuid4()))].append(step)
        return [Trajectory(name=get_trajectory_name(group_steps, name_key), steps=group_steps) for group_key, group_steps in step_groups.items()]


class SequenceAccumulator:
    def __init__(self):
        self.full_sequence = []
        self.logprobs = []
        self.advantages = []
        self.mask = []

    def is_empty(self):
        return len(self.full_sequence) == 0

    def clear(self):
        self.full_sequence = []
        self.logprobs = []
        self.advantages = []
        self.mask = []

    def add_step(self, step: Step, advantage: float, is_extension: bool = False):
        """Add a step to the accumulated sequence."""
        if is_extension:
            # Only add the new tokens (delta)
            prev_len = len(self.full_sequence)
            delta_prompt = step.prompt_ids[prev_len:]
            delta_prompt_len = len(delta_prompt)
        else:
            # Add entire prompt
            delta_prompt = step.prompt_ids
            delta_prompt_len = len(delta_prompt)

        # Add prompt tokens (observation)
        self.full_sequence.extend(delta_prompt)
        self.logprobs.extend([0.0] * delta_prompt_len)
        self.advantages.extend([0.0] * delta_prompt_len)
        self.mask.extend([0.0] * delta_prompt_len)

        # Add response tokens (action)
        self.full_sequence.extend(step.response_ids)
        self.logprobs.extend(step.logprobs)
        self.advantages.extend([advantage] * len(step.response_ids))
        self.mask.extend([1.0] * len(step.response_ids))


# def build_trajectories_from_steps(steps: List[Step]) -> list[Trajectory]:
#     """
#     Build one or more Datums from a trajectory, merging steps when possible.

#     Steps are merged when the next step's prompt is an extension of the
#     previous step's full sequence (prompt + response).

#     Args:
#         trajectory: Trajectory with steps
#         advantage: Advantage value for this trajectory

#     Returns:
#         List of Datum objects (may contain 1+ datums depending on merging)
#     """
#     if not steps:
#         return []

#     assert all(step.model_output is not None for step in steps), "model_output is None for some steps"
#     model_outputs = [step.model_output for step in steps]

#     # Build datums by iterating through steps
#     datums = []
#     accumulator = SequenceAccumulator()

#     for step_idx, step in enumerate(trajectory.steps):
#         if accumulator.is_empty():
#             # First step - start accumulating
#             accumulator.add_step(step, advantage, is_extension=False)
#         else:
#             # Check if current step extends previous sequence
#             prev_full_sequence = accumulator.full_sequence
#             current_prompt = step.prompt_ids

#             if TinkerDatumBuilder._is_prefix(prev_full_sequence, current_prompt):
#                 # Step extends previous - merge
#                 accumulator.add_step(step, advantage, is_extension=True)
#             else:
#                 # Step doesn't extend - create datum and start fresh
#                 datums.append(accumulator.to_datum())
#                 accumulator.clear()
#                 accumulator.add_step(step, advantage, is_extension=False)

#     # Create final datum from accumulated sequence
#     if not accumulator.is_empty():
#         datums.append(accumulator.to_datum())

#     return datums

# def build_trajectory_from_steps(steps: list[Step]) -> Trajectory:
