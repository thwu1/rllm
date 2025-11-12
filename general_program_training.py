from dataclasses import make_dataclass
from secrets import token_urlsafe
from rllm.sdk import RLLMClient
from rllm.agents.agent import Episode

rllm = RLLMClient()
client = rllm.get_chat_client(provider="openai")

def execute_tool_calls(tool_calls: list[dict]):
    pass

def calculate_reward(final_answer: str, solution: str):
    pass

def rollout(task: dict):
    problem = task["problem"]
    solution = task["solution"]

    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": problem}]

    while True:
        with rllm.session(agent_name="solver"):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )

        messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        if response.choices[0].tool_calls:
            tool_call_result = execute_tool_calls(response.choices[0].tool_calls)
            messages.append({"role": "tool", "content": tool_call_result})
        else:
            break
 
    final_answer = response.choices[0].message.content

    reward = calculate_reward(final_answer, solution)
    return reward

# trainer = AgentTrainer(
#     agent_class=MathAgent,
#     agent_args={},
#     env_args=env_args,
#     env_class=SingleTurnEnvironment,
#     config=config,
#     train_dataset=train_dataset,
#     val_dataset=test_dataset,
# )

traject:
tokens
masks


class SolverWor

def assemble_trajectories(traces: list[dict]):
    trace: logprobs, propt toekns, respose tokens, metadata {"agent_name": "solver"}
    solver_trace : list

    # given a list of (prompt, response) pairs, assemble the trajectories
    # All traces are comming from the same session (same rollout)
    pass

def process_fn(traces: list[dict]):
    trajectories = assemble_trajectories(traces)
    return Episode(
        trajectories=trajectories,
    )

rllm.train(
    entrypoint=rollout,
    process_fn=process_fn,
    dataset="gsm8k",
    num_epochs=10,
)