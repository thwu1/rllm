#!/usr/bin/env python3
"""Simple test script for @step and @trajectory decorators."""

import asyncio

from rllm.sdk import step, trajectory, StepView, TrajectoryView


# Test 1: Simple step decorator
@step(name="add")
def add_numbers(a: int, b: int) -> int:
    """Simple addition step."""
    return a + b


# Test 2: Async step decorator
@step(name="multiply")
async def multiply_numbers(a: int, b: int) -> int:
    """Async multiplication step."""
    await asyncio.sleep(0.01)  # Simulate async work
    return a * b


# Test 3: Trajectory decorator with steps
@trajectory(name="math_workflow", reward_mode="sum")
async def math_workflow(x: int, y: int) -> float:
    """A simple math workflow with multiple steps."""
    # Step 1: Add
    step1 = add_numbers(x, y)
    print(f"Step 1 result: {step1.result}")
    step1.reward = 1.0  # Delayed reward assignment

    # Step 2: Multiply
    step2 = await multiply_numbers(step1.result, 2)
    print(f"Step 2 result: {step2.result}")
    step2.reward = 1.0  # Delayed reward assignment

    # Return value not used when reward_mode="sum"
    return 0.0


# Test 4: Trajectory with delayed rewards (like solver-judge pattern)
@step(name="solve")
async def solve_problem(problem: str) -> str:
    """Simulate solving a problem."""
    await asyncio.sleep(0.01)
    return f"Solution to: {problem}"


@step(name="verify")
async def verify_solution(solution: str) -> bool:
    """Simulate verifying a solution."""
    await asyncio.sleep(0.01)
    return "Solution" in solution


@trajectory(name="solver_verifier", reward_mode="sum")
async def solver_verifier_workflow(problem: str):
    """Workflow similar to solver-judge pattern."""
    # Generate solution
    solve_step = await solve_problem(problem)
    print(f"Solve step: {solve_step.result}")

    # Verify solution
    verify_step = await verify_solution(solve_step.result)
    print(f"Verify step: {verify_step.result}")

    # Delayed reward assignment (like in solver-judge example)
    solve_step.reward = 1.0 if verify_step.result else 0.0
    verify_step.reward = 1.0 if verify_step.result else 0.0

    return 0.0  # Not used


async def main():
    print("=" * 60)
    print("Test 1: Simple step decorator (no LLM calls)")
    print("=" * 60)
    step_view = add_numbers(3, 4)
    assert isinstance(step_view, StepView)
    assert step_view.result == 7
    assert step_view.input is None  # No LLM calls
    assert step_view.output is None  # No LLM calls
    print(f"✓ Step result: {step_view.result}")
    print(f"✓ Step ID: {step_view.id}")
    print(f"✓ Step metadata: {step_view.metadata['step_name']}")
    print(f"✓ LLM calls: {step_view.metadata['llm_calls_count']}")
    print()

    print("=" * 60)
    print("Test 2: Async step decorator")
    print("=" * 60)
    step_view = await multiply_numbers(5, 3)
    assert isinstance(step_view, StepView)
    assert step_view.result == 15
    print(f"✓ Async step result: {step_view.result}")
    print()

    print("=" * 60)
    print("Test 3: Trajectory with steps")
    print("=" * 60)
    traj = await math_workflow(10, 5)
    assert isinstance(traj, TrajectoryView)
    assert len(traj.steps) == 2
    assert traj.steps[0].result == 15  # 10 + 5
    assert traj.steps[1].result == 30  # 15 * 2
    assert traj.reward == 2.0  # sum of step rewards (1.0 + 1.0)
    assert traj.input == {"x": 10, "y": 5}  # Function arguments captured
    assert traj.output == 0.0  # Function return value captured
    print(f"✓ Trajectory has {len(traj.steps)} steps")
    print(f"✓ Trajectory reward: {traj.reward}")
    print(f"✓ Trajectory input: {traj.input}")
    print(f"✓ Trajectory output: {traj.output}")
    print(f"✓ Final result: {traj.result}")  # Last step's result
    print()

    print("=" * 60)
    print("Test 4: Solver-Verifier workflow (delayed rewards)")
    print("=" * 60)
    traj = await solver_verifier_workflow("What is 2+2?")
    assert isinstance(traj, TrajectoryView)
    assert len(traj.steps) == 2
    assert traj.reward == 2.0  # Both steps got 1.0 reward
    print(f"✓ Workflow has {len(traj.steps)} steps")
    print(f"✓ Workflow reward: {traj.reward}")
    for i, step in enumerate(traj.steps):
        print(f"  Step {i+1}: {step.metadata['step_name']}, reward={step.reward}")
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
