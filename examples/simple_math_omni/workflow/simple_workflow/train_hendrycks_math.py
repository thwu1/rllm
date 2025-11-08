import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.simple_workflow import SimpleWorkflow

from rllm.sdk import RLLMClient
@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    # Define run function with client config as default arguments to avoid closure capture
    # This ensures the function is fully serializable for Ray
    def run(
        question: str,
        ground_truth: str,
        base_url: str = "http://localhost:4000/v1",
        api_key: str = "EMPTY",
        project: str = "rllm-agent-omni-engine",
        cs_endpoint: str = "http://localhost:8000",
        cs_api_key: str = "your-api-key-here",
        **kwargs,
    ):
        # Recreate the client inside the function to avoid serialization issues
        # This ensures the function doesn't capture non-serializable objects
        rllm_client = RLLMClient(
            base_url=base_url,
            api_key=api_key,
            project=project,
            cs_endpoint=cs_endpoint,
            cs_api_key=cs_api_key,
        )
        client = rllm_client.get_chat_client()
        
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
        workflow_class=SimpleWorkflow,
        workflow_args={
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=run,
    )
    trainer.train()


if __name__ == "__main__":
    main()
