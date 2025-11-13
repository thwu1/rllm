import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.sdk.shortcuts import get_chat_client
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    # Define run function that recreates the client inside to avoid closure capture
    # This ensures the function is fully serializable for Ray
    def rollout(
        question: str,
        ground_truth: str,
        base_url: str = "http://localhost:4000/v1",
        api_key: str = "EMPTY",
        **kwargs,
    ):
        # Recreate the client inside the function to avoid serialization issues
        # This ensures the function doesn't capture non-serializable objects
        client = get_chat_client(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model="vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            messages=[
                {"role": "user", "content": question},
            ],
        )
        response_text = response.choices[0].message.content
        reward = math_reward_fn({"response": response_text, "ground_truth": ground_truth}, response_text).reward
        return reward

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=rollout,
    )
    trainer.train()


if __name__ == "__main__":
    main()
