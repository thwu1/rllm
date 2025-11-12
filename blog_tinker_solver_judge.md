# rLLM: Training Agentic Programs with Tinker as the Backend

In this post, we'll explore how rLLM integrates with **Tinker**, a distributed training backend, to enable efficient training of multi-agent programs. The key insight is a **complete separation of concerns**: inference and training are handled by completely separate components, connected only by a simple data transformation layer. We'll walk through how this works using a solver-judge workflow as a concrete example.

## Complete Separation of Concerns

The architecture of rLLM's Tinker integration is built on a **complete separation of concerns** between inference and training:

### Inference: AgentWorkflowEngine

The inference part is handled by `AgentWorkflowEngine` (or `AsyncAgentExecutionEngine` for single-agent programs). This component:

- Generates trajectories through execution of workflow or an agent
- Coordinates multi-agent interactions
- Manages parallel task exsimply:ecution

**Crucially, `AgentWorkflowEngine` knows nothing about training.** It's purely focused on inference: taking tasks, running your program logic, and producing episodes.

### Training: TinkerPolicyTrainer

The training part is handled by `TinkerPolicyTrainer`. This component:

- Takes `Episode` objects as input
- Processes episodes (filtering, advantage computation, datum conversion)
- Performs distributed forward-backward passes via Tinker's training client
- Applies optimizer updates
- Manages checkpoints and weight synchronization

**Crucially, `TinkerPolicyTrainer` knows nothing about inference.** It's purely focused on training: taking episodes and updating model weights.

### The Glue: TinkerWorkflowTrainer / TinkerAgentTrainer

The trainers (`TinkerWorkflowTrainer` and `TinkerAgentTrainer`) are **thin glue layers** that connect inference and training. They:

1. **Use the inference engine** to generate episodes
2. **Transform the data** from inference format to `Episode` format
3. **Feed episodes** to `TinkerPolicyTrainer` for training

Here's how the flow works:

```
AgentWorkflowEngine (Inference)
    ↓ generates episodes
    ↓
TinkerWorkflowTrainer (Glue)
    ↓ transforms to Episode format
    ↓ (make_sure_contain_token_and_logprob, maybe_broadcast_reward, regroup)
    ↓
TinkerPolicyTrainer (Training)
    ↓ processes episodes and updates weights
```

### The Core Interface of `TinkerPolicyTrainer`: `step()`

`TinkerPolicyTrainer` provides a `step()` method that takes a list of `Episode` objects:

```python
async def step(
    self,
    episodes: list[Episode],
    learning_rate: float = None,
    optimizer_step: bool = True,
) -> tuple[list[torch.Tensor], list[tinker.Datum]]:
    """
    Complete training step: process episodes and update policy.

    This method:
    1. Filters episodes (if configured)
    2. Computes advantages
    3. Converts to datums
    4. Performs forward-backward pass
    5. Applies optimizer step

    Args:
        episodes: List of Episode objects
        learning_rate: Learning rate (uses config value if None)
        optimizer_step: Whether to apply optimizer step after forward-backward

    Returns:
        Tuple of (training_logprobs, training_datums)
    """
```

### The `Episode` Format: The Interface Contract

The `Episode` format is the interface contract between inference and training. An `Episode` contains:

```python
class Episode:
    id: str  # Unique identifier for the episode
    task: Any  # Task/problem being solved
    trajectories: list[Trajectory]  # List of trajectories in this episode
    metrics: dict  # Optional metrics
    is_correct: bool  # Whether the episode succeeded
```

Each `Trajectory` contains:
- `steps`: List of `Step` objects
- `reward`: Reward for this trajectory

Each `Step` contains:
- `prompt_ids`: Token IDs for the prompt
- `response_ids`: Token IDs for the model's response
- `logprobs`: Log probabilities for each token in the response
- `reward`: Reward for this step (optional, can be trajectory-level)

This format is the **only** thing that connects inference and training. The inference engine produces episodes, and the training engine consumes them.

## Example: Training a Solver-Judge Program

Let's walk through a concrete example: training a solver-judge program for solving countdown problems. This demonstrates the complete separation of concerns.

### Defining the Program

We define a `SolverJudgeWorkflow` that orchestrates the coordination between solver and judge agents:

```python
class SolverJudgeWorkflow(Workflow):
    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        problem = task["question"]
        
        # Step 1: Generate multiple solutions in parallel
        solver_trajectories = await self.solver.generate_solutions(
            problem, self.n_solutions
        )
        
        # Step 2: Judge selects the best solution
        judge_trajectory = await self.judge.judge_solutions(problem, solutions)
        
        # Step 3: Return episode with all trajectories
        return Episode(
            id=uid,
            task=task,
            trajectories=[*solver_trajectories, judge_trajectory],
            ...
        )
```

This defines your program structure: how agents interact, how trajectories are generated, and how rewards are assigned. The program logic is completely independent of both inference and training infrastructure.

### Training the Workflow

To train the solver-judge workflow, simply:

```python
from rllm.trainer.tinker.tinker_workflow_trainer import TinkerWorkflowTrainer
from rllm.rewards.countdown_reward import countdown_reward_fn

trainer = TinkerWorkflowTrainer(
    workflow_class=SolverJudgeWorkflow,
    workflow_args={
        "n_solutions": 2,
        "reward_function": countdown_reward_fn,
    },
    config=config,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
)
trainer.fit_agent()
```

### What Happens During Training with GRPO

1. For each task, the agent workflow engine generates `group_size` rollouts. 
2. Trajectories are grouped by their names (e.g. "solver", "judge"), as well as step index for advantage computation. Each group are converted to a `Episode` object.
3. The `TinkerWorkflowTrainer` updates the policy which improves both the solver and judge.


## Conclusion

The architecture of rLLM's Tinker integration is built on **complete separation of concerns** between inference and training:

- **Inference** is handled by `AgentWorkflowEngine` (or `AsyncAgentExecutionEngine`)
- **Training** is handled by `TinkerPolicyTrainer`
- **The trainers** (`TinkerWorkflowTrainer`, `TinkerAgentTrainer`) are just glue layers that transform data between inference and training

This separation makes it easy to:
- Write your own inference engines and rollout logic
- Focus on your program logic while rLLM handles distributed training

The key insight is that inference and training are completely independent, connected only by the `Episode` format. As long as you can produce `Episode` objects, `TinkerPolicyTrainer` will handle all the complexity of distributed training infrastructure, advantage computation, and model updates.

## References

- Solver-Judge example: `examples/solver_judge_tinker/`
