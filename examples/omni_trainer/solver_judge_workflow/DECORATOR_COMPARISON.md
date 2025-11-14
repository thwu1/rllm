# Solver-Judge Workflow: Before & After Decorators

This directory contains two implementations of the same workflow to demonstrate the benefits of the new `@step` and `@trajectory` decorators.

## Files

- **`simple_solver_judge_flow.py`**: Original implementation (manual session management)
- **`solver_judge_flow_with_decorators.py`**: New implementation using decorators

## Key Improvements with Decorators

### 1. **Automatic Session Management**

**Before (manual):**
```python
async def generate_solution(self, problem: str) -> Trajectory:
    with session(agent="solver", groupby_key=str(uuid.uuid4())) as sess:
        messages = [{"role": "user", "content": f"{problem}..."}]
        response = await self.client.chat.completions.create(
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content

    # Manual step extraction
    step = sess.steps[0]
    step.action = self._parse_solver_response(response_text)
    return step
```

**After (with @step):**
```python
@step(name="solve")
async def generate_solution(self, problem: str):
    messages = [{"role": "user", "content": f"{problem}..."}]
    response = await self.client.chat.completions.create(
        messages=messages,
        temperature=1,
        max_tokens=1000,
    )
    return response.choices[0].message.content
    # Decorator handles everything! Returns StepView automatically
```

**Benefits:**
- ✅ No manual `with session()` context manager
- ✅ No manual step extraction from `sess.steps[0]`
- ✅ Cleaner, more readable code
- ✅ Less boilerplate

### 2. **Direct Access to Results**

**Before:**
```python
step = sess.steps[0]  # Get step from session
step.action = self._parse_solver_response(response_text)
```

**After:**
```python
step = await self.generate_solution(problem)  # StepView returned directly
step.action = self._parse_solver_response(step.result)  # Access via .result
```

**Benefits:**
- ✅ Result available immediately via `.result` field
- ✅ No need to track response_text separately
- ✅ Type hints work better (IDE knows it returns StepView)

### 3. **Delayed Reward Assignment Still Works**

Both versions support delayed reward assignment:

```python
# Generate all solutions first
solver_steps = await self.solver.generate_solutions(problem, n_solutions)

# THEN assign rewards (delayed)
for solver_step in solver_steps:
    solver_step.action = parse_solution(solver_step.result)
    solver_step.reward = reward_function(task, solver_step.action).reward
```

This pattern is important for RL workflows where you want to:
1. Generate multiple candidates
2. Evaluate them together
3. Assign rewards based on comparison

### 4. **Less Code, Same Functionality**

**Line Count Comparison:**

| File | Lines | Comments |
|------|-------|----------|
| `simple_solver_judge_flow.py` | ~123 | Original with manual sessions |
| `solver_judge_flow_with_decorators.py` (decorated) | ~178 | Includes 3 variants + docs |
| `solver_judge_flow_with_decorators.py` (core only) | ~95 | Just the decorated version |

The decorated version has **~23% less code** in the core workflow implementation.

## Three Approaches Shown

### Approach 1: Drop-in Replacement (Recommended)

`SolverJudgeWorkflowDecorated` - Uses `@step` decorators on methods:

```python
class Solver:
    @step(name="solve")
    async def generate_solution(self, problem: str):
        response = await self.client.chat.completions.create(...)
        return response.choices[0].message.content

# Usage is the same!
solver_steps = await self.solver.generate_solutions(problem, n_solutions)
```

**When to use:** Drop-in replacement for existing workflows

### Approach 2: Fully Decorated

`SolverJudgeWorkflowFullyDecorated` - Same as above, just cleaner:

```python
# Exact same code, just demonstrates the pattern
```

**When to use:** New workflows or refactoring existing ones

### Approach 3: Standalone with @trajectory

`run_solver_judge_pipeline` - Uses `@trajectory` decorator:

```python
@trajectory(name="solver_judge_pipeline", reward_mode="sum")
async def run_solver_judge_pipeline(problem: str, reward_fn, n_solutions: int = 2):
    solver_steps = await solver.generate_solutions(problem, n_solutions)
    judge_step = await judge.judge_solutions(problem, solutions)
    return 0.0  # Reward from reward_mode="sum"

# Returns TrajectoryView with all steps auto-collected
traj = await run_solver_judge_pipeline(problem, reward_fn)
```

**When to use:** Ad-hoc workflows, scripts, notebooks

## Field Mapping: Before vs After

### Before (manual session):
```python
with session(agent="solver") as sess:
    response = await client.chat.completions.create(...)
    response_text = response.choices[0].message.content

step = sess.steps[0]  # StepView from Trace
# step.input = LLM input
# step.output = LLM output
# Need to track response_text separately
```

### After (@step decorator):
```python
@step(name="solve")
async def solve(problem: str):
    response = await client.chat.completions.create(...)
    return response.choices[0].message.content

step = await solve(problem)  # StepView from decorator
# step.input = None (not set by decorator)
# step.output = None (not set by decorator)
# step.result = response.choices[0].message.content ← User's return value
```

**Key Difference:**
- **Manual session**: `sess.steps[0]` comes from `trace_to_step_view()`, has LLM data in `input`/`output`
- **@step decorator**: Returns StepView with user's return value in `result` field

## Running the Examples

Both implementations work with the same trainer:

```python
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn

# Option 1: Original
from examples.omni_trainer.solver_judge_workflow.simple_solver_judge_flow import SolverJudgeWorkflow

# Option 2: Decorated
from examples.omni_trainer.solver_judge_workflow.solver_judge_flow_with_decorators import SolverJudgeWorkflowDecorated

# Both work the same way
trainer = AgentTrainer(
    workflow_class=SolverJudgeWorkflowDecorated,  # or SolverJudgeWorkflow
    workflow_args={
        "n_solutions": 2,
        "reward_function": countdown_reward_fn,
    },
    # ... other config
)
trainer.train()
```

## Migration Guide

To migrate an existing workflow:

1. **Add `@step` decorator to methods**:
   ```python
   # Before
   async def generate_solution(self, problem: str) -> Trajectory:
       with session(agent="solver") as sess:
           # ...
       return sess.steps[0]

   # After
   @step(name="solve")
   async def generate_solution(self, problem: str):
       # ... (no with session needed)
       return result  # Just return the value
   ```

2. **Update result access**:
   ```python
   # Before
   step = await self.generate_solution(problem)
   response_text = step.output  # or tracked separately

   # After
   step = await self.generate_solution(problem)
   response_text = step.result  # User's return value
   ```

3. **Delayed rewards still work**:
   ```python
   # Same in both versions
   step.action = parse(step.result)
   step.reward = reward_fn(task, step.action).reward
   ```

## Summary

The new decorators provide:
- ✅ **23% less code** for the same functionality
- ✅ **Better ergonomics** - no manual session management
- ✅ **Clearer intent** - `@step` explicitly marks semantic units
- ✅ **Same flexibility** - delayed rewards, parallel execution, etc.
- ✅ **Backward compatible** - old code still works

Choose decorators for new workflows, or migrate existing ones incrementally.
