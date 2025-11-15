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


class StepGroupView(BaseModel):
    """
    A view of a semantic step group.

    Represents a semantic unit of work that may contain multiple LLM calls (traces).
    Each trace becomes a training Step with tokens/logprobs.
    Provides a high-level view for reward assignment across all traces in the group.

    Hierarchy:
        TrajectoryView → StepGroupView → Trace/Step (training)

    Fields:
        - id: Unique identifier for this step group
        - input: Function arguments (dict)
        - output: Function return value (Any)
        - traces: All LLM calls made during this step group (list[Trace])
        - reward: Step group reward (assigned to all training Steps in this group)
        - metadata: Additional tracking data
    """
    id: str
    input: dict | None = None   # Function arguments
    output: Any = None          # Function return value
    traces: list[Trace] = Field(default_factory=list)  # All LLM calls
    reward: float = 0.0
    metadata: dict | None = None

    @property
    def result(self) -> Any:
        """Alias for output (backward compatibility)."""
        return self.output


class TrajectoryView(BaseModel):
    """
    A view of a trajectory execution.

    Represents a collection of step groups that form a complete workflow or episode.
    Used for RL training and workflow composition.

    Hierarchy:
        TrajectoryView → StepGroupView → Trace/Step (training)

    Fields:
        - name: Trajectory name
        - steps: List of collected StepGroupViews (semantic units)
        - reward: Trajectory reward (set manually)
        - input: Function arguments (dict)
        - output: Function return value (Any)
        - metadata: Additional tracking data
    """
    name: str = "agent"
    steps: list[StepGroupView] = Field(default_factory=list)
    reward: float = 0.0
    input: dict | None = None   # Function arguments
    output: Any = None          # Function return value
    metadata: dict | None = None  # Additional tracking data

    @property
    def result(self):
        """Get the result from the last step group, or None if no steps."""
        return self.steps[-1].result if self.steps else None


def trace_to_step_group_view(trace: Trace) -> StepGroupView:
    """Convert a low-level Trace to a high-level StepGroupView (single-trace group)."""
    return StepGroupView(
        id=trace.trace_id,
        input=trace.input,
        output=trace.output,
        traces=[trace],  # Include the trace itself
        reward=0.0,
        metadata=trace.metadata,
    )


# Backward compatibility aliases
StepView = StepGroupView  # Old name
StepProto = StepGroupView  # Even older name
TrajectoryProto = TrajectoryView
trace_to_step_view = trace_to_step_group_view  # Old function name
trace_to_step_proto = trace_to_step_group_view  # Even older function name
