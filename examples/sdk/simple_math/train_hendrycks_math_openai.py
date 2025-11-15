"""OpenAI-based testing version of train_hendrycks_math.py

This version uses OpenAI models via the LiteLLM proxy to enable fast testing
while still exercising the full proxy integration including:
- Proxy routing and metadata injection
- Trace collection through the proxy
- Proxy flush mechanisms
- Session tracking via proxy

This is the RECOMMENDED approach for validating your setup before deploying vLLM.

Cost: ~$0.01-0.10 for a quick test run with gpt-3.5-turbo
Time: 2-5 minutes (vs hours for vLLM deployment)

Prerequisites:
1. Start LiteLLM proxy with OpenAI config:
   ```bash
   export OPENAI_API_KEY="sk-..."

   python -m rllm.sdk.proxy.litellm_server \
     --config /tmp/litellm_openai_config.yaml \
     --host 127.0.0.1 \
     --port 4000 \
     --state-dir /tmp/litellm_proxy \
     --db-path /tmp/rllm_test.db \
     --project rllm-test \
     --admin-token my-secret
   ```

2. Create /tmp/litellm_openai_config.yaml:
   ```yaml
   model_list:
     - model_name: gpt-3.5-turbo
       litellm_params:
         model: gpt-3.5-turbo
         api_key: ${OPENAI_API_KEY}
   ```

Usage:
    python -m examples.omni_trainer.simple_math.train_hendrycks_math_openai \
        data.train_batch_size=4 \
        trainer.total_epochs=1
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.sdk.shortcuts import get_chat_client
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hendrycks_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    # Define run function that uses OpenAI via LiteLLM proxy
    def rollout(**kwargs):
        """Rollout function using OpenAI via LiteLLM proxy.

        This tests the full proxy integration including:
        - Proxy routing
        - Trace collection through proxy
        - Proxy flush mechanisms
        - Metadata injection
        """
        ground_truth = kwargs["ground_truth"]
        question = kwargs["question"]

        # Use real LiteLLM proxy (same as production, just different model)
        client = get_chat_client(base_url="http://localhost:4000/v1", api_key="EMPTY")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Fast, cheap OpenAI model
            messages=[
                {"role": "user", "content": question},
            ],
        )
        response_text = response.choices[0].message.content

        # Reward calculation is the same as in real version
        reward = math_reward_fn({"response": response_text, "ground_truth": ground_truth}, response_text).reward
        return reward * 1.0

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_run_func=rollout,
    )
    trainer.train()


if __name__ == "__main__":
    main()
