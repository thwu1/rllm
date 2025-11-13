"""
Example demonstrating the dual-mode support in AgentOmniEngine.

This example shows both modes:
1. MANUAL MODE: User manually assembles trajectories and returns list[TrajectoryProto]
2. AUTOMATED MODE: User returns just a float (reward), engine auto-assembles trajectories

To run this example:
    python examples/omni_trainer/dual_mode_example.py
"""

import asyncio

from rllm.sdk import get_chat_client_async, session
from rllm.sdk.protocol import StepProto, TrajectoryProto


# ============================================================================
# AUTOMATED MODE: Return just a reward (float)
# ============================================================================


async def agent_function_automated_mode(question: str, answer: str) -> float:
    """
    Agent function that uses AUTOMATED trajectory assembly mode.

    Simply returns a float reward - the engine will automatically group
    steps into trajectories using config.rllm.processing.groupby_key and
    config.rllm.processing.traj_name_key.

    Returns:
        float: The final reward for this episode
    """
    client = get_chat_client_async(
        base_url="http://localhost:4000/v1",
        api_key="EMPTY",
        model="vllm/Qwen/Qwen3-4B-Instruct-2507"
    )

    # Use session context with metadata for grouping
    # The 'agent' metadata field will be used as traj_name_key
    # The 'groupby_key' metadata will be used for grouping steps
    with session(agent="solver", groupby_key="reasoning_attempt_1"):
        messages = [{"role": "user", "content": f"Solve this: {question}"}]
        response = await client.chat.completions.create(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
        solution = response.choices[0].message.content

    # Optionally do more steps with different groupby_key
    with session(agent="verifier", groupby_key="reasoning_attempt_2"):
        messages = [{"role": "user", "content": f"Verify this solution: {solution}"}]
        response = await client.chat.completions.create(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
        verification = response.choices[0].message.content

    # Simple reward: 1.0 if solution contains answer, 0.0 otherwise
    reward = 1.0 if answer.lower() in solution.lower() else 0.0

    # AUTOMATED MODE: Just return the reward as a float
    # The engine will automatically:
    # 1. Group steps by 'groupby_key' into separate trajectories
    # 2. Name trajectories using 'agent' field (solver, verifier)
    # 3. Assign the reward to all trajectories
    return reward


# ============================================================================
# MANUAL MODE: Return list[TrajectoryProto]
# ============================================================================


async def agent_function_manual_mode(question: str, answer: str) -> list[TrajectoryProto]:
    """
    Agent function that uses MANUAL trajectory assembly mode.

    Manually assembles trajectories by creating TrajectoryProto objects
    with specific steps and rewards.

    Returns:
        list[TrajectoryProto]: Manually assembled trajectories
    """
    client = get_chat_client_async(
        base_url="http://localhost:4000/v1",
        api_key="EMPTY",
        model="vllm/Qwen/Qwen3-4B-Instruct-2507"
    )

    # Step 1: Generate solution
    with session(agent="solver") as solver_session:
        messages = [{"role": "user", "content": f"Solve this: {question}"}]
        response = await client.chat.completions.create(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
        solution = response.choices[0].message.content

    # Step 2: Verify solution
    with session(agent="verifier") as verifier_session:
        messages = [{"role": "user", "content": f"Verify this solution: {solution}"}]
        response = await client.chat.completions.create(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )
        verification = response.choices[0].message.content

    # Calculate rewards
    solver_reward = 1.0 if answer.lower() in solution.lower() else 0.0
    verifier_reward = 1.0 if "correct" in verification.lower() else 0.0

    # MANUAL MODE: Manually assemble trajectories
    # Create TrajectoryProto objects with specific steps and rewards
    trajectories = [
        TrajectoryProto(
            name="solver",
            steps=[
                StepProto(
                    id=solver_session.steps[0].id,
                    reward=solver_reward,
                )
            ],
            reward=solver_reward,
        ),
        TrajectoryProto(
            name="verifier",
            steps=[
                StepProto(
                    id=verifier_session.steps[0].id,
                    reward=verifier_reward,
                )
            ],
            reward=verifier_reward,
        ),
    ]

    return trajectories


# ============================================================================
# Example usage
# ============================================================================


async def main():
    """Demonstrate both modes."""

    print("=" * 80)
    print("AUTOMATED MODE Example")
    print("=" * 80)
    print("In automated mode, the agent function returns just a float (reward).")
    print("The engine automatically groups steps into trajectories using config.\n")

    # Example task
    task = {"question": "What is 2 + 2?", "answer": "4"}

    # Run automated mode
    reward = await agent_function_automated_mode(**task)
    print(f"✓ Agent returned reward: {reward}")
    print("  → Engine will auto-group steps by 'groupby_key' metadata")
    print("  → Engine will name trajectories using 'agent' metadata")
    print("  → All trajectories will receive the same reward\n")

    print("=" * 80)
    print("MANUAL MODE Example")
    print("=" * 80)
    print("In manual mode, the agent function returns list[TrajectoryProto].")
    print("The user has full control over trajectory assembly and rewards.\n")

    # Run manual mode
    trajectories = await agent_function_manual_mode(**task)
    print(f"✓ Agent returned {len(trajectories)} trajectories:")
    for traj in trajectories:
        print(f"  - {traj.name}: {len(traj.steps)} steps, reward={traj.reward}")
    print("  → Engine will use these exact trajectories")
    print("  → Each trajectory can have different rewards\n")

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("AUTOMATED MODE:")
    print("  ✓ Simpler: just return a float")
    print("  ✓ Auto-grouping via metadata")
    print("  ✓ Single reward for all trajectories")
    print("  ✓ Good for simple workflows\n")

    print("MANUAL MODE:")
    print("  ✓ Full control over trajectories")
    print("  ✓ Per-trajectory rewards")
    print("  ✓ Per-step rewards")
    print("  ✓ Good for complex multi-agent workflows\n")


if __name__ == "__main__":
    asyncio.run(main())
