# AgentOmniEngine Dual-Mode Support

The `AgentOmniEngine` now supports two modes for trajectory assembly:

## Mode 1: Automated Mode (Simpler)

**When to use**: Simple workflows where all trajectories should receive the same reward.

**How it works**: Your agent function returns just a `float` (the reward). The engine automatically groups steps into trajectories using metadata from your `session()` calls.

**Example**:

```python
async def my_agent(question: str, answer: str) -> float:
    client = get_chat_client_async(...)

    # Step 1: Solver
    with session(agent="solver", groupby_key="attempt_1"):
        response = await client.chat.completions.create(...)
        solution = response.choices[0].message.content

    # Step 2: Verifier
    with session(agent="verifier", groupby_key="attempt_2"):
        response = await client.chat.completions.create(...)

    # Just return the reward - engine handles the rest!
    reward = 1.0 if answer in solution else 0.0
    return reward
```

**Configuration**:

You need to specify how to group steps in your config:

```yaml
rllm:
  processing:
    groupby_key: "groupby_key"  # Metadata field to group steps by
    traj_name_key: "agent"      # Metadata field for trajectory names
```

**How grouping works**:
- Steps with the same `groupby_key` value are grouped into one trajectory
- Trajectory names come from the `traj_name_key` metadata field
- All trajectories receive the same reward (the float you returned)

## Mode 2: Manual Mode (Full Control)

**When to use**: Complex workflows where you need fine-grained control over trajectory assembly and per-trajectory/per-step rewards.

**How it works**: Your agent function returns a `list[TrajectoryProto]`. You manually specify which steps belong to which trajectory and what reward each step/trajectory should receive.

**Example**:

```python
from rllm.sdk.protocol import TrajectoryProto, StepProto

async def my_agent(question: str, answer: str) -> list[TrajectoryProto]:
    client = get_chat_client_async(...)

    # Step 1: Generate solutions
    with session(agent="solver") as solver_session:
        response = await client.chat.completions.create(...)
        solution = response.choices[0].message.content

    # Step 2: Verify
    with session(agent="judge") as judge_session:
        response = await client.chat.completions.create(...)

    # Manually assemble trajectories with specific rewards
    trajectories = [
        TrajectoryProto(
            name="solver",
            steps=[
                StepProto(
                    id=solver_session.steps[0].id,
                    reward=1.0 if answer in solution else 0.0,
                )
            ],
        ),
        TrajectoryProto(
            name="judge",
            steps=[
                StepProto(
                    id=judge_session.steps[0].id,
                    reward=0.5,  # Different reward!
                )
            ],
        ),
    ]

    return trajectories
```

**Advantages**:
- Full control over which steps belong to which trajectory
- Different rewards per trajectory
- Different rewards per step within a trajectory
- Can create multiple trajectories with the same name
- Perfect for solver-judge workflows

## Comparison

| Feature | Automated Mode | Manual Mode |
|---------|---------------|-------------|
| Return type | `float` | `list[TrajectoryProto]` |
| Complexity | Simpler | More verbose |
| Reward control | Same for all | Per-trajectory/step |
| Grouping | Automatic via metadata | Manual control |
| Best for | Simple workflows | Complex multi-agent workflows |
| Config required | Yes (groupby_key) | No |

## Migration Guide

### From Old Code (Pre-Dual-Mode)

Before, the engine only supported manual mode. If you were already returning `list[TrajectoryProto]`, no changes needed!

### Switching to Automated Mode

**Before** (manual):
```python
async def my_agent(task):
    with session() as sess:
        # ... do work ...

    return [
        TrajectoryProto(
            name="agent",
            steps=[StepProto(id=step.id, reward=1.0) for step in sess.steps]
        )
    ]
```

**After** (automated):
```python
async def my_agent(task):
    with session(agent="agent", groupby_key="main"):
        # ... do work ...

    return 1.0  # Just the reward!
```

Don't forget to add config:
```yaml
rllm:
  processing:
    groupby_key: "groupby_key"
    traj_name_key: "agent"
```

## Internal Implementation

The engine detects the mode by checking the return type:

```python
if isinstance(user_return_value, list):
    # MANUAL MODE: User manually assembled trajectories
    colorful_print(f"[{session_id}] Using MANUAL trajectory assembly mode", fg="blue")
    # ... convert TrajectoryProto to Trajectory ...

elif isinstance(user_return_value, (float, int)):
    # AUTOMATED MODE: Auto-assemble trajectories
    colorful_print(f"[{session_id}] Using AUTOMATED trajectory assembly mode", fg="cyan")
    trajectories = group_steps(steps, by=groupby, name_key=traj_name_key)
    for trajectory in trajectories:
        trajectory.reward = final_reward
```

## Example Files

- `examples/omni_trainer/dual_mode_example.py` - Demonstrates both modes
- `examples/omni_trainer/solver_judge_workflow/simple_solver_judge_flow.py` - Manual mode example
- `examples/omni_trainer/simple_math/train_hendrycks_math.py` - Can be adapted for automated mode

## Troubleshooting

**Error: "Invalid return type from agent function"**
- Make sure you're returning either a `float` or `list[TrajectoryProto]`
- Check that you're not returning `None` or other types

**Automated mode not grouping correctly**
- Check that your config has `groupby_key` and `traj_name_key` set
- Verify that your `session()` calls include the metadata fields
- Example: `session(agent="solver", groupby_key="attempt1")`

**Steps not appearing in trajectories**
- Make sure your LLM calls are inside a `session()` context manager
- Check that the session metadata is being set correctly
- Verify that step IDs match between sessions and TrajectoryProto (manual mode)

## Best Practices

1. **Use automated mode for simple workflows** where all trajectories should receive the same reward
2. **Use manual mode for complex workflows** like solver-judge where different agents need different rewards
3. **Set clear metadata** in your `session()` calls to make grouping predictable
4. **Test both modes** with a simple example before deploying to production
5. **Log which mode is being used** - the engine will print colored output indicating the mode
