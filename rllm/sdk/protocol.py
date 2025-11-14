from typing import Any

from pydantic import BaseModel, Field


class Trace(BaseModel):
    trace_id: str
    session_name: str
    name: str
    input: str | list | dict
    output: str | dict
    model: str
    latency_ms: float
    tokens: dict[str, int]
    metadata: dict = Field(default_factory=dict)
    timestamp: float
    parent_trace_id: str | None = None
    cost: float | None = None
    environment: str | None = None
    tools: list[dict] | None = None
    contexts: list[str | dict] | None = None
    tags: list[str] | None = None


class StepView(BaseModel):
    """
    A view of a single step execution.

    Represents a semantic unit of work that may contain multiple LLM calls.
    Provides a high-level view for reward assignment and action extraction.

    Fields:
        - input/output: LLM-level data (input to model, response from model)
          * Filled by tracer when converting Trace → StepView (single trace)
          * OR by @step decorator: populated only if exactly 1 LLM call
          * For 0 or multiple calls: both None (check metadata['llm_traces'])
        - result: User-defined function return value (set by @step decorator)
        - action: Parsed action/answer (set manually after step creation)
        - reward: Step reward (set manually, supports delayed assignment)
        - metadata: Always contains complete LLM trace list in metadata['llm_traces']
    """
    id: str
    action: str | None = None
    output: dict | None = None  # LLM output (from traces)
    input: dict | None = None  # LLM input (from traces)
    result: Any = None  # User's function return value
    reward: float = 0.0
    metadata: dict | None = None


class TrajectoryView(BaseModel):
    """
    A view of a trajectory execution.

    Represents a collection of steps that form a complete workflow or episode.
    Used for RL training and workflow composition.

    Fields:
        - input: Function arguments (dict of args/kwargs from @trajectory decorator)
        - output: Function return value (set by @trajectory decorator)
        - steps: List of collected StepViews
        - reward: Trajectory reward (calculated based on reward_mode)
        - metadata: Additional tracking data (flexible for future use)
    """
    name: str = "agent"
    steps: list[StepView] = Field(default_factory=list)
    reward: float = 0.0
    input: dict | None = None   # Function arguments
    output: Any = None          # Function return value
    metadata: dict | None = None  # Additional tracking data

    @property
    def result(self):
        """Get the result from the last step, or None if no steps."""
        return self.steps[-1].result if self.steps else None


def trace_to_step_view(trace: Trace) -> StepView:
    """Convert a low-level Trace to a high-level StepView."""
    return StepView(
        id=trace.trace_id,
        input=trace.input,
        output=trace.output,
        action=None,
        reward=0.0,
        metadata=trace.metadata,
    )


# Backward compatibility aliases
StepProto = StepView
TrajectoryProto = TrajectoryView
trace_to_step_proto = trace_to_step_view
