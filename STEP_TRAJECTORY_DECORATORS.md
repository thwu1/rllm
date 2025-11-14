# Step and Trajectory Decorators

## Overview

The SDK now provides `@step` and `@trajectory` decorators for building RL workflows. These decorators are built **entirely on top of the session primitive** - no new context variables are introduced.

## Key Design Principles

1. **Session-based**: Everything uses `session()` under the hood
2. **Simple formatting**: Just wraps session and formats `sess.llm_calls` into StepView fields
3. **Decorator changes return**: `@step` returns `StepView`, `@trajectory` returns `TrajectoryView`
4. **Context manager preserves return**: Use `step_context()` or `trajectory_context()` when you need the original return value
5. **Delayed rewards**: Rewards can be assigned after step/trajectory creation
6. **Automatic collection**: Steps auto-register with parent trajectories via session metadata

## Architecture

```
Low-level:    session() → Trace (individual LLM call)
              ↓
Mid-level:    @step → StepView (semantic unit, 0 or 1 LLM call)
              ↓
High-level:   @trajectory → TrajectoryView (collection of steps)
```

## Data Model: Symmetric Design

Both `StepView` and `TrajectoryView` follow the same pattern:

| Field | StepView (LLM-level) | TrajectoryView (Function-level) |
|---|---|---|
| **input** | LLM request (dict/list) | Function arguments (dict) |
| **output** | LLM response (dict) | Function return value (Any) |
| **result** | Function return value (Any) | Last step's result (property) |
| **action** | Parsed action (str, optional) | N/A |
| **reward** | Step reward (float) | Trajectory reward (float) |
| **metadata** | Execution info + traces (dict) | User-defined tracking (dict) |
| **Other** | `id: str` | `name: str`, `steps: list[StepView]` |

### StepView Fields

- **`input` / `output`**: LLM-level data (input to model, response from model)
  - Automatically formatted from `sess.llm_calls` by `@step` decorator
  - A step should have **at most one LLM call** (0 or 1)
  - `input`/`output` = trace input/output if present, `None` otherwise

- **`result`**: User's function return value
  - Set by `@step` decorator
  - This is what the decorated function returned

- **`action`**: Parsed action/answer
  - Set manually after step creation
  - Example: extract `<answer>42</answer>` from LLM output

- **`reward`**: Step reward
  - Set manually (supports delayed assignment)
  - Used for RL training

- **`metadata`**: Execution info, function args, LLM traces
  - Automatically populated by `@step` decorator

### TrajectoryView Fields

- **`input`**: Function arguments
  - Automatically captured by `@trajectory` decorator
  - Dict of parameter names to values: `{"problem": "2+2", "n": 3}`

- **`output`**: Function return value
  - Automatically captured by `@trajectory` decorator
  - Can be any type

- **`steps`**: List of collected StepViews
  - Steps auto-register with parent trajectory

- **`reward`**: Trajectory reward
  - Calculated based on `reward_mode` ("return", "sum", "last", "manual")

- **`metadata`**: User-defined tracking data
  - Passed via `**traj_metadata` in decorator
  - Flexible for future extensions

- **`result`** (property): Last step's result
  - Convenience property, not stored

**Two ways to create StepView:**

1. **From Trace** (via `trace_to_step_view`): `input`/`output` have single trace LLM data, `result` is None
2. **From Decorator** (via `@step`): `result` has function return, `input`/`output` formatted from `sess.llm_calls`

## Usage

### Basic Step

```python
from rllm.sdk import step, StepView

@step(name="solve")
async def solve_problem(query: str) -> str:
    # Session auto-created internally
    # All LLM calls tracked automatically
    response = await llm.chat.completions.create(...)
    return response.choices[0].message.content

# Returns StepView (not string!)
step_view: StepView = await solve_problem("What is 2+2?")

# Access result (user's function return value)
print(step_view.result)  # "4"

# input/output are formatted from sess.llm_calls
print(step_view.input)   # {"messages": [...]} - LLM request
print(step_view.output)  # {"choices": [...]} - LLM response

# Access all traces in metadata
print(step_view.metadata['llm_traces'])  # All LLM calls

# Delayed reward assignment
step_view.reward = 1.0

# Parse and set action
step_view.action = parse_answer(step_view.result)

# Access metadata
print(step_view.metadata['llm_calls_count'])  # How many LLM calls
print(step_view.metadata['function_args'])    # Function arguments
```

### Basic Trajectory

```python
from rllm.sdk import trajectory, TrajectoryView

@trajectory(name="math_solver", reward_mode="sum")
async def solve_workflow(problem: str) -> float:
    # All @step calls auto-collected
    step1 = await solve_problem(problem)
    step1.reward = calculate_reward(step1.result)

    step2 = await verify_solution(step1.result)
    step2.reward = calculate_reward(step2.result)

    # Return value not used when reward_mode="sum"
    return 0.0

# Returns TrajectoryView (not float!)
traj: TrajectoryView = await solve_workflow("2+2")

# Access function input/output
print(traj.input)   # {"problem": "2+2"}
print(traj.output)  # 0.0 (function return value)

# Access collected steps
print(traj.steps)  # [step1, step2]

# Access reward (sum of step rewards)
print(traj.reward)  # 2.0

# Access final result
print(traj.result)  # Last step's result
```

### Solver-Judge Pattern (from examples)

```python
from rllm.sdk import step, trajectory

class Solver:
    @step(name="solve")
    async def generate_solution(self, problem: str):
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content

class Judge:
    @step(name="judge")
    async def judge_solutions(self, problem: str, solutions: list[str]):
        response = await self.client.chat.completions.create(...)
        return self._parse_judge_response(response)

@trajectory(name="solver_judge", reward_mode="sum")
async def solver_judge_workflow(task: dict):
    problem = task["question"]

    # Generate multiple solutions
    solver_steps = [
        await solver.generate_solution(problem)
        for _ in range(n_solutions)
    ]

    # Delayed reward assignment (after all solutions generated)
    for step in solver_steps:
        step.action = parse_solution(step.result)
        step.reward = reward_function(task, step.action).reward

    # Judge step
    judge_step = await judge.judge_solutions(
        problem,
        [s.action for s in solver_steps]
    )
    judge_step.reward = reward_function(task, judge_step.action).reward

    return 0.0  # Not used with reward_mode="sum"

# Usage
traj = await solver_judge_workflow(task)
# traj.steps = [solver_step1, solver_step2, judge_step]
# traj.reward = sum of all step rewards
```

## Reward Modes

For `@trajectory`, you can specify how rewards are calculated:

```python
# Option 1: Use return value
@trajectory(reward_mode="return")
async def workflow():
    step = await some_step()
    return 1.0  # This becomes traj.reward

# Option 2: Sum all step rewards (default for multi-step)
@trajectory(reward_mode="sum")
async def workflow():
    step1 = await step_a()
    step1.reward = 1.0
    step2 = await step_b()
    step2.reward = 0.5
    return 0.0  # Ignored, traj.reward = 1.5

# Option 3: Use last step's reward
@trajectory(reward_mode="last")
async def workflow():
    step1 = await step_a()
    step2 = await step_b()
    step2.reward = 1.0
    return 0.0  # Ignored, traj.reward = step2.reward

# Option 4: Manual (set reward yourself)
@trajectory(reward_mode="manual")
async def workflow():
    steps = [await step_a(), await step_b()]
    return 0.0  # Ignored, traj.reward = 0.0 (set manually after)

traj = await workflow()
traj.reward = custom_calculation()
```

## Context Managers (when you need original return)

```python
from rllm.sdk import step_context, trajectory_context

# Step context manager - preserves return value
with step_context(name="solve") as ctx:
    result = await some_function()  # Returns original value!
    ctx.set_result(result)

# Access step after
print(ctx.step_view.result)
ctx.step_view.reward = 1.0

# Trajectory context manager - preserves return value
with trajectory_context(name="workflow") as traj_ctx:
    step1 = await solve()  # Returns StepView (still decorated)
    step2 = await verify()
    result = some_calculation()  # Can return anything

# Access trajectory after
print(traj_ctx.trajectory_view.steps)
print(traj_ctx.trajectory_view.reward)
```

## Implementation Details

### How Steps Register with Trajectories

1. `@step` creates a session with `_is_step=True` marker in metadata
2. When step completes, it walks the session stack via `get_active_sessions()`
3. Finds parent session with `_is_trajectory=True` marker
4. Appends itself to parent's `metadata['_collected_steps']` list
5. **No new context variables** - everything via session metadata!

### Session Hierarchy

```python
@trajectory(name="workflow")
async def workflow():
    # Creates parent session with _is_trajectory=True
    # metadata['_collected_steps'] = []

    step1 = await solve()
    # Creates child session with _is_step=True
    # Walks stack, finds parent, appends to _collected_steps

    step2 = await verify()
    # Same process

    # When trajectory exits, metadata['_collected_steps'] = [step1, step2]
```

### Why This Design?

- ✅ Uses **only** session primitive (no new context vars)
- ✅ Session already handles async, threading, multiprocessing
- ✅ Nested trajectories work automatically (session nesting)
- ✅ Backward compatible (sessions work the same way)
- ✅ Clear separation: Trace → StepView → TrajectoryView

## Comparison with Direct Session Usage

**Before (manual):**
```python
with session(agent="solver") as sess:
    response = await client.chat.completions.create(...)
    result = response.choices[0].message.content

step = sess.steps[0]
step.action = parse(result)
return step
```

**After (with decorator):**
```python
@step(name="solve")
async def solve(problem: str):
    response = await client.chat.completions.create(...)
    return response.choices[0].message.content

step = await solve(problem)
step.action = parse(step.result)
return step
```

Benefits:
- Less boilerplate
- Clearer intent
- Automatic step creation
- Type hints work better
- Composable (can call decorated functions from anywhere)

## API Reference

### `@step(name=None, **metadata)`

Decorator that creates a step from a function.

**Parameters:**
- `name`: Step name (default: function name)
- `**metadata`: Additional metadata

**Returns:** `StepView` with:
- `.id`: Unique step ID
- `.input`: LLM input (dict/list, from trace)
- `.output`: LLM output (dict, from trace)
- `.result`: Function return value (Any)
- `.action`: Parsed action (set manually)
- `.reward`: Step reward (set manually)
- `.metadata`: Execution info + function args + LLM traces

### `@trajectory(name="agent", reward_mode="return", **metadata)`

Decorator that creates a trajectory from a function.

**Parameters:**
- `name`: Trajectory name
- `reward_mode`: "return", "sum", "last", or "manual"
- `**metadata`: Additional metadata (stored in `.metadata`)

**Returns:** `TrajectoryView` with:
- `.name`: Trajectory name
- `.input`: Function arguments (dict)
- `.output`: Function return value (Any)
- `.steps`: List of collected StepViews
- `.reward`: Calculated reward
- `.metadata`: User-defined tracking data
- `.result`: Last step's result (property)

### `step_context(name=None, **metadata)`

Context manager for step (preserves return value).

### `trajectory_context(name="agent", reward_mode="sum", **metadata)`

Context manager for trajectory (preserves return value).

## Testing

See `test_decorators.py` for examples and tests.
