"""Decorators for step and trajectory tracking using session primitives."""

import asyncio
import inspect
import time
import uuid
from functools import wraps
from typing import Any, Callable

from rllm.sdk.protocol import StepView, TrajectoryView
from rllm.sdk.session import get_active_sessions, get_current_session
from rllm.sdk.shortcuts import session


def step(name: str | None = None, **step_metadata):
    """
    Decorator to mark a function as a step.

    Creates a session internally for this step and returns a StepView.
    The decorator **changes the return value** - it returns StepView instead
    of the original return value.

    The StepView captures:
    - result: User's function return value (accessible via .result)
    - input/output: LLM-level data (formatted from sess.llm_calls)
      * A step should have at most one LLM call (0 or 1)
      * input/output = trace input/output if present, None otherwise
    - action: Can be set later for parsed results
    - reward: Can be set later (supports delayed reward assignment)
    - metadata: Execution info + function args + all LLM traces

    Steps automatically register with parent @trajectory if one exists.

    Args:
        name: Name of the step (defaults to function name)
        **step_metadata: Additional metadata to attach to the step

    Returns:
        Decorator that wraps the function to return StepView

    Example:
        >>> @step(name="solve")
        >>> async def solve_problem(query: str):
        ...     response = await llm.chat.completions.create(...)
        ...     return response.choices[0].message.content

        >>> step_view = await solve_problem("What is 2+2?")
        >>> print(step_view.result)  # "4" - function return value
        >>> step_view.action = parse_answer(step_view.result)
        >>> step_view.reward = 1.0  # Delayed reward assignment
    """
    def decorator(func: Callable) -> Callable:
        step_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> StepView:
                # Create a session for this step
                step_id = f"step_{uuid.uuid4().hex[:16]}"
                start_time = time.time()

                with session(
                    _is_step=True,  # Mark as step session
                    step_name=step_name,
                    step_id=step_id,
                    **step_metadata
                ) as sess:
                    # Execute the function
                    result = await func(*args, **kwargs)

                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # A step should have at most one LLM call
                    assert len(sess.llm_calls) in (0, 1), f"Step should have 0 or 1 LLM call, got {len(sess.llm_calls)}"

                    # Format LLM call into input/output (0 or 1 call)
                    step_input = sess.llm_calls[0].input if sess.llm_calls else None
                    step_output = sess.llm_calls[0].output if sess.llm_calls else None

                    # Create StepView
                    step_view = StepView(
                        id=step_id,
                        input=step_input,  # LLM input
                        output=step_output,  # LLM output
                        result=result,  # User's function return value
                        action=None,  # Can be set later
                        reward=0.0,  # Can be set later (delayed)
                        metadata={
                            "step_name": step_name,
                            "function_name": func.__name__,
                            "function_args": args,
                            "function_kwargs": kwargs,
                            "execution_time_ms": execution_time_ms,
                            "llm_calls_count": len(sess.llm_calls),
                            "llm_traces": [trace.model_dump() for trace in sess.llm_calls],  # Store all traces
                            "session_name": sess.name,
                            **step_metadata
                        }
                    )

                    # Register with parent trajectory (if exists)
                    _register_step_with_trajectory(step_view)

                    return step_view

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> StepView:
                # Create a session for this step
                step_id = f"step_{uuid.uuid4().hex[:16]}"
                start_time = time.time()

                with session(
                    _is_step=True,  # Mark as step session
                    step_name=step_name,
                    step_id=step_id,
                    **step_metadata
                ) as sess:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Calculate execution time
                    execution_time_ms = (time.time() - start_time) * 1000

                    # A step should have at most one LLM call
                    assert len(sess.llm_calls) in (0, 1), f"Step should have 0 or 1 LLM call, got {len(sess.llm_calls)}"

                    # Format LLM call into input/output (0 or 1 call)
                    step_input = sess.llm_calls[0].input if sess.llm_calls else None
                    step_output = sess.llm_calls[0].output if sess.llm_calls else None

                    # Create StepView
                    step_view = StepView(
                        id=step_id,
                        input=step_input,  # LLM input
                        output=step_output,  # LLM output
                        result=result,  # User's function return value
                        action=None,  # Can be set later
                        reward=0.0,  # Can be set later (delayed)
                        metadata={
                            "step_name": step_name,
                            "function_name": func.__name__,
                            "function_args": args,
                            "function_kwargs": kwargs,
                            "execution_time_ms": execution_time_ms,
                            "llm_calls_count": len(sess.llm_calls),
                            "llm_traces": [trace.model_dump() for trace in sess.llm_calls],  # Store all traces
                            "session_name": sess.name,
                            **step_metadata
                        }
                    )

                    # Register with parent trajectory (if exists)
                    _register_step_with_trajectory(step_view)

                    return step_view

            return sync_wrapper

    return decorator


def trajectory(name: str = "agent", reward_mode: str = "return", **traj_metadata):
    """
    Decorator to mark a function as a trajectory.

    Creates a parent session that collects all @step calls made within it.
    The decorator **changes the return value** - it returns TrajectoryView
    instead of the original return value.

    All @step decorated functions called within this trajectory are
    automatically collected into trajectory.steps.

    Args:
        name: Name of the trajectory
        reward_mode: How to calculate trajectory reward:
            - "return": Use function return value as reward (default)
            - "sum": Sum all step rewards
            - "last": Use last step's reward
            - "manual": Reward must be set manually on returned TrajectoryView
        **traj_metadata: Additional metadata for the trajectory

    Returns:
        Decorator that wraps the function to return TrajectoryView

    Example:
        >>> @trajectory(name="solver", reward_mode="sum")
        >>> async def solve_workflow(task: dict):
        ...     # All @step calls are auto-collected
        ...     step1 = await solve(task["question"])
        ...     step1.reward = calc_reward(step1.result)
        ...
        ...     step2 = await verify(step1.result)
        ...     step2.reward = calc_reward(step2.result)
        ...
        ...     return 0.0  # Not used when reward_mode="sum"

        >>> traj = await solve_workflow(task)
        >>> print(traj.steps)  # [step1, step2]
        >>> print(traj.reward)  # sum of step rewards
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> TrajectoryView:
                # Create a parent session for trajectory
                with session(
                    _is_trajectory=True,  # Mark as trajectory session
                    trajectory_name=name,
                    **traj_metadata
                ) as traj_sess:
                    # Initialize step collection in session metadata
                    traj_sess.metadata['_collected_steps'] = []

                    # Run the function - @step calls will register themselves
                    result = await func(*args, **kwargs)

                    # Get collected steps
                    steps = traj_sess.metadata['_collected_steps']

                    # Calculate reward based on mode
                    reward = _calculate_trajectory_reward(reward_mode, result, steps)

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=reward
                    )

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> TrajectoryView:
                # Create a parent session for trajectory
                with session(
                    _is_trajectory=True,  # Mark as trajectory session
                    trajectory_name=name,
                    **traj_metadata
                ) as traj_sess:
                    # Initialize step collection in session metadata
                    traj_sess.metadata['_collected_steps'] = []

                    # Run the function - @step calls will register themselves
                    result = func(*args, **kwargs)

                    # Get collected steps
                    steps = traj_sess.metadata['_collected_steps']

                    # Calculate reward based on mode
                    reward = _calculate_trajectory_reward(reward_mode, result, steps)

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=reward
                    )

            return sync_wrapper

    return decorator


class StepContext:
    """
    Context manager for step execution.

    Unlike the @step decorator, the context manager does NOT change the return value.
    Use this when you need explicit control over step boundaries.

    Example:
        >>> with step_context(name="solve") as step_ctx:
        ...     response = await llm.chat.completions.create(...)
        ...     result = response.choices[0].message.content

        >>> # Access step after context
        >>> print(step_ctx.step_view.result)  # Access captured step
        >>> step_ctx.step_view.reward = 1.0
    """

    def __init__(self, name: str | None = None, **metadata):
        self.name = name or "step"
        self.metadata = metadata
        self.step_view: StepView | None = None
        self._session = None
        self._step_id = f"step_{uuid.uuid4().hex[:16]}"
        self._start_time = None
        self._result = None

    def __enter__(self):
        self._start_time = time.time()
        self._session = session(
            _is_step=True,
            step_name=self.name,
            step_id=self._step_id,
            **self.metadata
        ).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time_ms = (time.time() - self._start_time) * 1000

        # A step should have at most one LLM call
        assert len(self._session.llm_calls) in (0, 1), f"Step should have 0 or 1 LLM call, got {len(self._session.llm_calls)}"

        # Format LLM call into input/output (0 or 1 call)
        step_input = self._session.llm_calls[0].input if self._session.llm_calls else None
        step_output = self._session.llm_calls[0].output if self._session.llm_calls else None

        # Create StepView
        self.step_view = StepView(
            id=self._step_id,
            input=step_input,  # LLM input
            output=step_output,  # LLM output
            result=self._result,  # User-set result via set_result()
            action=None,
            reward=0.0,
            metadata={
                "step_name": self.name,
                "execution_time_ms": execution_time_ms,
                "llm_calls_count": len(self._session.llm_calls),
                "llm_traces": [trace.model_dump() for trace in self._session.llm_calls],
                "session_name": self._session.name,
                **self.metadata
            }
        )

        # Register with parent trajectory
        _register_step_with_trajectory(self.step_view)

        # Exit session
        self._session.__exit__(exc_type, exc_val, exc_tb)
        return False

    def set_result(self, result: Any):
        """Set the result/output for this step."""
        self._result = result


class TrajectoryContext:
    """
    Context manager for trajectory execution.

    Unlike the @trajectory decorator, the context manager does NOT change the return value.
    Use this when you need explicit control over trajectory boundaries.

    Example:
        >>> with trajectory_context(name="solver") as traj_ctx:
        ...     step1 = await solve(problem)
        ...     step2 = await verify(step1.result)
        ...     result = step2.result

        >>> # Access trajectory after context
        >>> print(traj_ctx.trajectory_view.steps)  # [step1, step2]
        >>> print(traj_ctx.trajectory_view.reward)
    """

    def __init__(self, name: str = "agent", reward_mode: str = "sum", **metadata):
        self.name = name
        self.reward_mode = reward_mode
        self.metadata = metadata
        self.trajectory_view: TrajectoryView | None = None
        self._session = None

    def __enter__(self):
        self._session = session(
            _is_trajectory=True,
            trajectory_name=self.name,
            **self.metadata
        ).__enter__()

        # Initialize step collection
        self._session.metadata['_collected_steps'] = []

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get collected steps
        steps = self._session.metadata['_collected_steps']

        # Calculate reward
        reward = _calculate_trajectory_reward(self.reward_mode, None, steps)

        # Create TrajectoryView
        self.trajectory_view = TrajectoryView(
            name=self.name,
            steps=steps,
            reward=reward
        )

        # Exit session
        self._session.__exit__(exc_type, exc_val, exc_tb)
        return False


# Convenience functions for context managers
def step_context(name: str | None = None, **metadata) -> StepContext:
    """Create a step context manager."""
    return StepContext(name=name, **metadata)


def trajectory_context(name: str = "agent", reward_mode: str = "sum", **metadata) -> TrajectoryContext:
    """Create a trajectory context manager."""
    return TrajectoryContext(name=name, reward_mode=reward_mode, **metadata)


# Helper functions

def _register_step_with_trajectory(step_view: StepView):
    """
    Register a step with its parent trajectory (if exists).

    Uses get_active_sessions() to walk up the session stack and find
    the parent trajectory session. This is the ONLY way to communicate
    between steps and trajectories - via session metadata.
    """
    sessions_stack = get_active_sessions()

    # Walk backwards through session stack (inner to outer)
    # Skip the current step session (last in stack)
    for sess in reversed(sessions_stack[:-1]):
        # Check if this is a trajectory session
        if sess.metadata.get('_is_trajectory'):
            # Found parent trajectory - add step to its collection
            collected_steps = sess.metadata.get('_collected_steps')
            if collected_steps is not None:
                collected_steps.append(step_view)
            break


def _calculate_trajectory_reward(reward_mode: str, return_value: Any, steps: list[StepView]) -> float:
    """
    Calculate trajectory reward based on mode.

    Args:
        reward_mode: "return", "sum", "last", or "manual"
        return_value: The function's return value
        steps: List of collected steps

    Returns:
        Calculated reward as float
    """
    if reward_mode == "return":
        # Use function return value
        if isinstance(return_value, (int, float)):
            return float(return_value)
        else:
            return 0.0
    elif reward_mode == "sum":
        # Sum all step rewards
        return sum(s.reward for s in steps)
    elif reward_mode == "last":
        # Use last step's reward
        return steps[-1].reward if steps else 0.0
    elif reward_mode == "manual":
        # User will set reward manually
        return 0.0
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}. Use 'return', 'sum', 'last', or 'manual'")
