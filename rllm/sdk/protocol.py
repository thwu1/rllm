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
    A concise view of a single LLM call (trace) with reward.

    StepView is essentially a trace wrapper that adds a reward field.

    Fields:
        - id: Trace ID, unique per trace, can be used to retrieve the full trace from the store
        - input: LLM input (from trace)
        - output: LLM response (from trace)
        - action: Parsed action (set manually by user)
        - reward: Step reward
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
    A view of a trajectory.

    Represents a collection of steps (each step = 1 trace)
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

    @property
    def result(self):
        """Get the output from the trajectory (backward compatibility)."""
        return self.output


def trace_to_step_view(trace: Trace) -> StepView:
    """Convert a trace to a StepView (trace wrapper with reward field)."""
    return StepView(
        id=trace.trace_id,
        input=trace.input,
        output=trace.output,
        reward=0.0,
        metadata=trace.metadata,
    )
