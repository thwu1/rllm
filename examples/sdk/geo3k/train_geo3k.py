import hydra

from examples.sdk.geo3k.geo3k_workflow import Geo3KWorkflowSdk
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


async def run_workflow(**kwargs) -> float:
    task = kwargs
    workflow = Geo3KWorkflowSdk(rollout_engine=None, executor=None, reward_function=math_reward_fn)
    return await workflow.run(task, "")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("geo3k", "train")
    test_dataset = DatasetRegistry.load_dataset("geo3k", "test")

    assert train_dataset, "Train dataset not found. Please run examples/sdk/geo3k/preprocess_geo3k.py first."
    assert test_dataset, "Test dataset not found. Please run examples/sdk/geo3k/preprocess_geo3k.py first."

    trainer = AgentTrainer(
        agent_run_func=run_workflow,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
