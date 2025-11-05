import asyncio

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.sdk.client import RLLMClient
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    agent_args = {"tools": ["python"], "parser_name": "qwen", "system_prompt": "You are a math assistant that can write python to solve math problems. Let's think step by step and put the final answer within \\boxed{}."}
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": "http://localhost:30000/v1", "api_key": "None", "model": model_name},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=4096,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )
    rllm = RLLMClient(
        base_url="http://localhost:30000/v1",
        api_key="None",
        project="sft-firework",
        cs_endpoint="http://localhost:8000",
        cs_api_key="your-api-key-here",
    )

    # Override the openai client with the RLLM OpenAI client for auto tracking responses
    engine.rollout_engine.client = rllm.get_chat_client_async()

    test_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    print(len(test_dataset))

    tasks = test_dataset.repeat(n=1)  # repeat to evaluate pass@k
    tasks = tasks[:1000]

    with rllm.session(f"batch-{model_name}-first1000"):
        results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
