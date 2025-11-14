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
    - input: Function arguments (dict)
    - output: Function return value (accessible via .result property)
    - traces: All LLM calls made during step execution
    - reward: Can be set later (supports delayed reward assignment)
    - metadata: Contains execution info and additional tracking data

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
        >>> print(step_view.input)   # {"query": "What is 2+2?"}
        >>> print(step_view.traces)  # [Trace(...)] - all LLM calls
        >>> step_view.reward = 1.0   # Delayed reward assignment
    """
    def decorator(func: Callable) -> Callable:
        step_name = name or func.__name__
        # Get function signature for capturing args/kwargs
        sig = inspect.signature(func)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> StepView:
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

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

                    # Create StepView
                    step_view = StepView(
                        id=step_id,
                        input=func_input,      # Function arguments
                        output=result,         # Function return value
                        traces=sess.llm_calls, # All LLM calls
                        reward=0.0,
                        metadata={
                            "step_name": step_name,
                            "function_name": func.__name__,
                            "execution_time_ms": execution_time_ms,
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
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

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

                    # Create StepView
                    step_view = StepView(
                        id=step_id,
                        input=func_input,      # Function arguments
                        output=result,         # Function return value
                        traces=sess.llm_calls, # All LLM calls
                        reward=0.0,
                        metadata={
                            "step_name": step_name,
                            "function_name": func.__name__,
                            "execution_time_ms": execution_time_ms,
                            "session_name": sess.name,
                            **step_metadata
                        }
                    )

                    # Register with parent trajectory (if exists)
                    _register_step_with_trajectory(step_view)

                    return step_view

            return sync_wrapper

    return decorator


def trajectory(name: str = "agent", **traj_metadata):
    """
    Decorator to mark a function as a trajectory.

    Creates a parent session that collects all @step calls made within it.
    The decorator **changes the return value** - it returns TrajectoryView
    instead of the original return value.

    All @step decorated functions called within this trajectory are
    automatically collected into trajectory.steps.

    Reward must be set manually on the returned TrajectoryView.

    Args:
        name: Name of the trajectory
        **traj_metadata: Additional metadata for the trajectory

    Returns:
        Decorator that wraps the function to return TrajectoryView

    Example:
        >>> @trajectory(name="solver")
        >>> async def solve_workflow(task: dict, n: int):
        ...     # All @step calls are auto-collected
        ...     step1 = await solve(task["question"])
        ...     step1.reward = calc_reward(step1.result)
        ...
        ...     step2 = await verify(step1.result)
        ...     step2.reward = calc_reward(step2.result)
        ...
        ...     return "final_answer"

        >>> traj = await solve_workflow(task, n=3)
        >>> # Set reward manually
        >>> traj.reward = sum(s.reward for s in traj.steps)
        >>> print(traj.input)   # {"task": {...}, "n": 3}
        >>> print(traj.output)  # "final_answer"
        >>> print(traj.steps)   # [step1, step2]
        >>> print(traj.reward)  # Manually set reward
    """
    def decorator(func: Callable) -> Callable:
        # Get function signature for capturing args/kwargs
        sig = inspect.signature(func)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> TrajectoryView:
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

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

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=0.0,  # Must be set manually by user
                        input=func_input,  # Function arguments
                        output=result,     # Function return value
                        metadata=traj_metadata if traj_metadata else None
                    )

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> TrajectoryView:
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

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

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=0.0,  # Must be set manually by user
                        input=func_input,  # Function arguments
                        output=result,     # Function return value
                        metadata=traj_metadata if traj_metadata else None
                    )

            return sync_wrapper

    return decorator


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
