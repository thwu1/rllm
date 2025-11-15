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
    A view of a single step (one LLM call).

    Each trace in a trajectory is automatically converted to a StepView.
    StepView is essentially a trace wrapper that adds a reward field.

    Hierarchy:
        TrajectoryView → StepView (1 trace each)

    Fields:
        - id: Trace ID
        - input: LLM input (from trace)
        - output: LLM output (from trace)
        - reward: Step reward (assigned to training Step)
        - metadata: Additional tracking data (can include model, tokens, latency, etc.)
    """

    id: str
    input: str | list | dict | None = None  # LLM input
    output: str | dict | None = None  # LLM output
    action: Any | None = None
    reward: float = 0.0
    metadata: dict | None = None


class TrajectoryView(BaseModel):
    """
    A view of a trajectory execution.

    Represents a collection of steps (each step = 1 trace) that form a workflow.
    Each trace in the trajectory is automatically converted to a StepView.

    Hierarchy:
        TrajectoryView → StepView (1 trace each)

    Fields:
        - name: Trajectory name
        - steps: List of StepViews (auto-generated from traces)
        - reward: Trajectory reward (set manually)
        - input: Function arguments (dict)
        - output: Function return value (Any)
        - metadata: Additional tracking data
    """

    name: str = "agent"
    steps: list[StepView] = Field(default_factory=list)
    reward: float = 0.0
    input: dict | None = None  # Function arguments
    output: Any = None  # Function return value
    metadata: dict | None = None  # Additional tracking data


def trace_to_step_view(trace: Trace) -> StepView:
    """Convert a trace to a StepView (trace wrapper with reward field)."""
    return StepView(
        id=trace.trace_id,
        input=trace.input,
        output=trace.output,
        reward=0.0,
        metadata=trace.metadata,
    )


# Backward compatibility aliases
StepProto = StepView
TrajectoryProto = TrajectoryView
trace_to_step_proto = trace_to_step_view
