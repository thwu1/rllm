"""OpenAI-based version of train_decorator.py for fast testing with real proxy.

This tests the full LiteLLM proxy integration with inexpensive OpenAI models.

Prerequisites:
1. Start LiteLLM proxy with OpenAI config
2. Set OPENAI_API_KEY environment variable

Usage:
    python -m examples.omni_trainer.solver_judge_workflow.train_decorator_openai \
        data.train_batch_size=4 \
        trainer.total_epochs=1
"""

import hydra

from examples.omni_trainer.solver_judge_workflow.solver_judge_flow_decorator_openai import SolverJudgeWorkflowDecoratedOpenAI
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.countdown_reward import countdown_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


async def run_workflow(**kwargs) -> float:
    """Run workflow with OpenAI via LiteLLM proxy."""
    task = kwargs
    workflow = SolverJudgeWorkflowDecoratedOpenAI(rollout_engine=None, executor=None, n_solutions=2, reward_function=countdown_reward_fn)
    return await workflow.run(task, "")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("countdown", "train")
    test_dataset = DatasetRegistry.load_dataset("countdown", "test")

    trainer = AgentTrainer(
        agent_run_func=run_workflow,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
