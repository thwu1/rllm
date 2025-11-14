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


class StepProto(BaseModel):
    id: str
    action: str | None = None
    output: dict | None = None
    input: dict | None = None
    reward: float = 0.0
    metadata: dict | None = None


class TrajectoryProto(BaseModel):
    name: str = "agent"
    steps: list[StepProto] = Field(default_factory=list)
    reward: float = 0.0


def trace_to_step_proto(trace: Trace) -> StepProto:
    return StepProto(
        id=trace.trace_id,
        input=trace.input,
        output=trace.output,
        action=None,
        reward=0.0,
        metadata=trace.metadata,
    )
