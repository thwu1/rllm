"""
Solver-Judge workflow using @step and @trajectory decorators.

This is a cleaner version of simple_solver_judge_flow.py that demonstrates
how the new decorators streamline the workflow implementation.

Key improvements:
- @step decorator eliminates manual session management
- Automatic StepView creation with result field
- Cleaner code with less boilerplate
- Same functionality, better ergonomics
"""

import asyncio
import re
import uuid

from rllm.agents.agent import Episode, Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.sdk import get_chat_client_async, step, trajectory
from rllm.sdk.protocol import TrajectoryView
from rllm.workflows.workflow import Workflow


class Solver:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(
            base_url="http://localhost:4000/v1",
            api_key="EMPTY",
            model="Qwen/Qwen3-4B-Instruct-2507"
        )

    @step(name="solver")
    async def generate_solution(self, problem: str):
        """
        Generate a solution using @step decorator.

        The decorator:
        - Creates a session internally
        - Tracks LLM calls automatically
        - Returns StepView with result field set to the return value
        """
        messages = [
            {
                "role": "user",
                "content": f"{problem}. Output the final answer within <answer>...</answer>"
            }
        ]
        response = await self.client.chat.completions.create(
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content

        # Parse and return the answer (matches original behavior)
        return self._parse_solver_response(response_text)

    async def generate_solutions(self, problem: str, n_solutions: int = 2):
        """Generate multiple solutions in parallel."""
        tasks = [
            asyncio.create_task(self.generate_solution(problem))
            for _ in range(n_solutions)
        ]
        # Returns list of StepView objects
        return await asyncio.gather(*tasks)

    def _parse_solver_response(self, response: str) -> str:
        """Extract answer from response."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return f"<answer>{answer_match.group(1).strip()}</answer>"
        else:
            return "No solution found"


class Judge:
    def __init__(self, **kwargs):
        self.client = get_chat_client_async(
            base_url="http://localhost:4000/v1",
            api_key="EMPTY",
            model="Qwen/Qwen3-4B-Instruct-2507"
        )

    @step(name="judge")
    async def judge_solutions(self, problem: str, solutions: list[str]):
        """
        Judge solutions using @step decorator.

        Returns StepView with the selected solution in the result field.
        """
        messages = [
            {
                "role": "user",
                "content": self._create_judge_prompt(problem, solutions)
            }
        ]
        response = await self.client.chat.completions.create(
            messages=messages,
            temperature=1,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content

        # Parse and return the selected solution (matches original behavior)
        return self._parse_judge_response(response_text, solutions)

    def _parse_judge_response(self, response: str, solutions: list[str]) -> str:
        """Parse judge response to get selected solution."""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            try:
                solution_index = int(answer_text)
                return solutions[solution_index - 1]
            except (ValueError, IndexError):
                return ""
        return ""

    def _create_judge_prompt(self, problem: str, solutions: list[str]) -> str:
        """Create a prompt for the judge to evaluate solutions."""
        prompt = f"""You are an expert verifier. Given a countdown problem and multiple solution attempts, select a correct solution.
Problem:
{problem}
Solutions to evaluate:
"""
        for i, solution in enumerate(solutions, 1):
            prompt += f"\nSolution {i}:\n{solution}\n"

        prompt += """
A correct solution must satisfy the following criteria:
1. The solution uses only the given numbers.
2. Each number is used exactly once.
3. Only basic arithmetic operations (+, -, *, /) are used.
4. The calculation results in the target number.
5. The final answer is clearly marked within <answer>...</answer> tags.
Output the index of your selected solution within <answer>...</answer> tags, e.g., <answer>1</answer> for the first solution, <answer>2</answer> for the second solution, etc. If multiple solutions are correct, output the index of the first correct solution."""
        return prompt


class SolverJudgeWorkflowDecorated(Workflow):
    """
    Solver-Judge workflow using decorators.

    This shows how @step simplifies the workflow implementation:
    - No manual session management
    - Cleaner code with automatic StepView creation
    - Easy access to results via .result field
    - Delayed reward assignment still works
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_solutions: int = 2,
        reward_function: RewardFunction = None,
        **kwargs
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver()
        self.judge = Judge()

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Run the solver-judge workflow.

        Demonstrates:
        1. Using @step decorated methods
        2. Accessing results via .result field
        3. Delayed reward assignment
        4. Manual TrajectoryView construction
        """
        self.reset(task, uid)
        problem = task["question"]

        # Step 1: Solver generates multiple solutions in parallel
        # Each call returns a StepView
        solver_steps = await self.solver.generate_solutions(problem, self.n_solutions)

        # Assign rewards to solver trajectories (delayed reward assignment)
        solutions = []
        for solver_step in solver_steps:
            # The parsed answer is already in .result (from @step decorator)
            parsed_answer = solver_step.result

            # Set action to the parsed answer
            solver_step.action = parsed_answer
            solutions.append(solver_step.action)

            # Delayed reward assignment
            solver_step.reward = self.reward_function(task, solver_step.action).reward

        # Step 2: Judge selects the best solution
        # Returns StepView with selected solution in .result (already parsed)
        judge_step = await self.judge.judge_solutions(problem, solutions)

        # The selected solution is already in .result (from @step decorator)
        selected_solution = judge_step.result
        judge_step.action = selected_solution

        # Evaluate the selected solution and set reward
        reward_result = self.reward_function(task, judge_step.action)
        judge_step.reward = reward_result.reward

        # Construct trajectory views manually (same as before)
        # Each solver step becomes its own trajectory
        solver_trajectories = [
            TrajectoryView(name="solver", steps=[solver_step])
            for solver_step in solver_steps
        ]

        # Judge gets its own trajectory
        judge_trajectory = TrajectoryView(name="judge", steps=[judge_step])

        return solver_trajectories + [judge_trajectory]


# Alternative: Using @trajectory decorator for even cleaner code
class SolverJudgeWorkflowFullyDecorated(Workflow):
    """
    Fully decorated version using both @step and @trajectory.

    This is the cleanest approach - let decorators handle everything.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_solutions: int = 2,
        reward_function: RewardFunction = None,
        **kwargs
    ):
        super().__init__(rollout_engine, **kwargs)
        self.n_solutions = n_solutions
        self.reward_function = reward_function
        self.solver = Solver()
        self.judge = Judge()

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Run workflow - manually construct trajectories for now.

        Note: We could use @trajectory decorator on helper methods,
        but the workflow engine expects a specific return format,
        so we construct TrajectoryViews manually here.
        """
        self.reset(task, uid)
        problem = task["question"]

        # Generate solutions
        solver_steps = await self.solver.generate_solutions(problem, self.n_solutions)

        # Process solutions
        solutions = []
        for solver_step in solver_steps:
            # Result is already parsed by the @step decorator
            solver_step.action = solver_step.result
            solver_step.reward = self.reward_function(task, solver_step.action).reward
            solutions.append(solver_step.action)

        # Judge solutions
        judge_step = await self.judge.judge_solutions(problem, solutions)
        # Result is already parsed by the @step decorator
        judge_step.action = judge_step.result
        judge_step.reward = self.reward_function(task, judge_step.action).reward

        # Return trajectory views
        return [
            TrajectoryView(name="solver", steps=[solver_step])
            for solver_step in solver_steps
        ] + [TrajectoryView(name="judge", steps=[judge_step])]


# Example standalone workflow using @trajectory decorator
@trajectory(name="solver_judge_pipeline", reward_mode="sum")
async def run_solver_judge_pipeline(problem: str, reward_fn, n_solutions: int = 2):
    """
    Standalone workflow using @trajectory decorator.

    This is the most streamlined approach for ad-hoc workflows.
    Returns TrajectoryView automatically with all steps collected.
    """
    solver = Solver()
    judge = Judge()

    # Generate solutions
    solver_steps = await solver.generate_solutions(problem, n_solutions)

    # Process and score solutions
    solutions = []
    for solver_step in solver_steps:
        solver_step.action = solver._parse_solver_response(solver_step.result)
        solver_step.reward = reward_fn({"question": problem}, solver_step.action).reward
        solutions.append(solver_step.action)

    # Judge solutions
    judge_step = await judge.judge_solutions(problem, solutions)
    judge_step.action = judge._parse_judge_response(judge_step.result, solutions)
    judge_step.reward = reward_fn({"question": problem}, judge_step.action).reward

    # Return 0 - actual reward comes from reward_mode="sum"
    return 0.0
